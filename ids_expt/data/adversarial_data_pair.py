from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from ids_expt.core.defs import DataType
import cv2
import torch
import pandas as pd
from loguru import logger
import numpy as np
from pathlib import Path


class AdversarialDataPairConfig(BaseModel):
    # in HPC, its tempdir
    data_dir: Path
    # npz_samples_dir={data_dir}/adv_samples/{adv_alg}_eps_{epsilon}/{train/val}/{label}_{batch_idx}_{item_idx}.npz
    # npz has: input_image: np.ndarray, target_image: np.ndarray, label str
    # here, str will be {adv_alg}_{epsilon}
    adversarial_type_selection_rate: list[tuple[str, float]] = Field(
        default=[
            ("basiciterativemethod_eps_0.1", 0.25),
            ("fastgradientmethod_eps_0.1", 0.25),
            ("fastgradientmethod_eps_0.01", 0.25),
            ("fastgradientmethod_eps_0.01", 0.25),
        ],
        description="List of tuples with adversarial type and its selection rate. ",
    )
    clean_selection_rate: float = Field(
        default=0.5,
        description="Selection rate for clean images",
    )

    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    num_samples_per_epoch: int = Field(
        default=10000,
        description="Number of samples to use per epoch for training. -1 means use all available data.",
    )


class AdversarialDataPair:
    def __init__(self, config: AdversarialDataPairConfig):
        self.config = config
        self.clean_selection_rate = config.clean_selection_rate
        self.random_seed = config.random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.data_type = None
        self.adversarial_type_selection_rate = config.adversarial_type_selection_rate
        self.adversarial_type_npz_files = {
            adv_type: [] for adv_type, _ in self.adversarial_type_selection_rate
        }

    def load_data(self):
        # load file names
        data_dir = self.config.data_dir
        # fill adversarial type npz files for training
        for adv_type, selection_rate in self.adversarial_type_selection_rate:
            adv_npz_files = list(data_dir.glob(f"adv_samples/{adv_type}/train/*.npz"))

            if not adv_npz_files:
                logger.warning(f"No files found for adversarial type: {adv_type}")
                continue
            # apply this in __getitem__
            # selected_files = self.random_state.choice(
            #     adv_npz_files,
            #     size=int(len(adv_npz_files) * selection_rate),
            #     replace=False,
            # )
            self.adversarial_type_npz_files[adv_type].extend(adv_npz_files)

        logger.info(
            f"Loaded adversarial files from {len(self.adversarial_type_npz_files)} types."
        )

        # make train data pair
        train_pair = AdversarialDataPair(
            config=self.config,
        )
        train_pair.data_type = DataType.TRAIN
        train_pair.adversarial_type_npz_files = self.adversarial_type_npz_files

        # fill adversarial type npz files for validation
        self.adversarial_type_npz_files = {
            adv_type: [] for adv_type, _ in self.adversarial_type_selection_rate
        }
        for adv_type, selection_rate in self.adversarial_type_selection_rate:
            adv_npz_files = list(data_dir.glob(f"adv_samples/{adv_type}/train/*.npz"))
            if not adv_npz_files:
                logger.warning(f"No files found for adversarial type: {adv_type}")
                continue
            self.adversarial_type_npz_files[adv_type].extend(adv_npz_files)
        # make validation data pair
        val_pair = AdversarialDataPair(
            config=self.config,
        )
        val_pair.data_type = DataType.VALIDATION
        val_pair.adversarial_type_npz_files = self.adversarial_type_npz_files

        return train_pair, val_pair

    def __len__(self):
        return self.config.num_samples_per_epoch

    def __getitem__(self, idx):
        # randomly select an adversarial type based on selection rates
        adv_type = self.random_state.choice(
            [adv_type for adv_type, _ in self.adversarial_type_selection_rate],
            p=[rate for _, rate in self.adversarial_type_selection_rate],
        )
        # randomly select a file from the selected adversarial type
        selected_files = self.adversarial_type_npz_files[adv_type]
        if not selected_files:
            logger.warning(f"No files available for adversarial type: {adv_type}")
            raise ValueError(f"No files available for adversarial type: {adv_type}")

        selected_file = self.random_state.choice(selected_files)
        # load the npz file
        npz_data = np.load(selected_file)
        input_image = npz_data["inputs"]
        adversarial_image = npz_data["adversarial"]
        label = npz_data["label_str"].item()

        # our input will be adversarial image and target will be clean image
        target_img = input_image.copy()
        input_image = adversarial_image.copy()

        if (
            self.config.clean_selection_rate > 0
            and self.random_state.rand() < self.config.clean_selection_rate
        ):
            # select clean image as input
            input_image = target_img.copy()
        # normalize by 255
        input_image = input_image.astype(np.float32) / 255.0
        target_img = target_img.astype(np.float32) / 255.0
        return input_image, target_img


class TorchPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: AdversarialDataPair):
        self.dataset = dataset
        self.config = dataset.config
        self.num_classes = -1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_img, target_img = self.dataset[idx]
        return (
            torch.tensor(input_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target_img, dtype=torch.float32).unsqueeze(0),
        )


if __name__ == "__main__":
    from ids_expt.data.adversarial_data_pair import (
        AdversarialDataPairConfig,
        AdversarialDataPair,
    )
    import os

    config = AdversarialDataPairConfig(
        data_dir=Path(os.environ.get("DATA_DIR", "data/adv_samples"))
    )
    dataset = AdversarialDataPair(config)
    train_dataset, test_dataset = dataset.load_data()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of getting an item
    input_img, target_img = train_dataset[0]
    print(
        f"Input image shape: {input_img.shape}, Target image shape: {target_img.shape}"
    )
