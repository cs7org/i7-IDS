from pydantic import BaseModel, Field
from ids_expt.core.defs import DataType
import torch
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
            ("basiciterativemethod_eps_0.1", 0.5),
            ("fastgradientmethod_eps_0.1", 0.2),
            ("basiciterativemethod_eps_0.01", 0.15),
            ("fastgradientmethod_eps_0.01", 0.15),
        ],
        description="List of tuples with adversarial type and its selection rate. ",
    )
    clean_selection_rate: float = Field(
        default=0.3,
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
    model_name: str = Field(
        default="original_cnn",
        description="Name of the model to be used for training.",
    )
    labels: list[str] = Field(
        default=[
            "REPLAY",
            "DNP3_INFO",
            "DNP3_ENUMERATE",
            "STOP_APP",
            "NORMAL",
            "INIT_DATA",
            "COLD_RESTART",
            "WARM_RESTART",
            "DISABLE_UNSOLICITED",
        ],
        description="List of labels to be used for classification.",
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
        self.label_counts = {}
        self.label_encoding = {
            label: idx for idx, label in enumerate(config.labels)
        }

    def load_data(self):
        # load file names
        data_dir = self.config.data_dir
        # fill adversarial type npz files for training
        # get label counts from file name as: "{lbl_str}_{batch_idx}_{i}.npz"
        for adv_type, selection_rate in self.adversarial_type_selection_rate:
            adv_npz_files = list(data_dir.rglob(f"{adv_type}/train/*.npz"))
            adv_npz_files = [
                file for file in adv_npz_files if self.config.model_name in str(file)
            ]
            if not adv_npz_files:
                logger.warning(f"No train files found for adversarial type: {adv_type}")
                continue

            self.adversarial_type_npz_files[adv_type] = [
                np.load(file) for file in adv_npz_files
            ]
            logger.info(
                f"Adversarial Type: {adv_type}, Selection Rate: {selection_rate}"
            )

        logger.info(
            f"Loaded adversarial files from {len(self.adversarial_type_npz_files)} types."
        )

        # make train data pair
        train_pair = AdversarialDataPair(
            config=self.config,
        )
        train_pair.data_type = DataType.TRAIN
        train_pair.adversarial_type_npz_files = self.adversarial_type_npz_files
        train_pair.label_counts = self.label_counts

        # fill adversarial type npz files for validation
        self.adversarial_type_npz_files = {
            adv_type: [] for adv_type, _ in self.adversarial_type_selection_rate
        }

        for adv_type, selection_rate in self.adversarial_type_selection_rate:
            adv_npz_files = list(data_dir.rglob(f"{adv_type}/validation/*.npz"))
            adv_npz_files = [
                file for file in adv_npz_files if self.config.model_name in str(file)
            ]
            if not adv_npz_files:
                logger.warning(f"No valid files found for adversarial type: {adv_type}")
                continue
            self.adversarial_type_npz_files[adv_type] = [
                np.load(file) for file in adv_npz_files
            ]

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
        npz_data = self.adversarial_type_npz_files[adv_type][0]

        # load the npz file
        input_image = npz_data["inputs"]
        adversarial_image = npz_data["adversarial"]
        labels = npz_data["labels"]
        selected_idx = self.random_state.randint(0, len(input_image))

        # our input will be adversarial image and target will be clean image
        target_img = input_image.copy()[selected_idx]
        input_image = adversarial_image.copy()[selected_idx]
        if "fastgradientmethod" in adv_type:
            eps = adv_type.split("_")[-1]
            self.data_kind = f"FGSSM({eps})"
        else:
            eps = adv_type.split("_")[-1]
            self.data_kind = f"BIGM({eps})"

        if (
            self.config.clean_selection_rate > 0
            and self.random_state.rand() < self.config.clean_selection_rate
        ):
            # select clean image as input
            input_image = target_img.copy()
            self.data_kind = "Clean"

        label = labels[selected_idx]
        input_image = input_image.astype(np.float32)
        target_img = target_img.astype(np.float32)
        return input_image, target_img, label


class TorchPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: AdversarialDataPair):
        self.dataset = dataset
        self.config = dataset.config
        self.num_classes = -1
        self.current_label = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_img, target_img, label = self.dataset[idx]
        self.current_label = label

        self.data_kind = self.dataset.data_kind
        return (
            torch.from_numpy(input_img).float(),
            torch.from_numpy(target_img).float(),
            torch.tensor(label).to(torch.float32),
        )


if __name__ == "__main__":
    from ids_expt.data.adversarial_tabular_data_pair import (
        AdversarialDataPairConfig,
        AdversarialDataPair,
    )
    import os

    config = AdversarialDataPairConfig(
        data_dir=Path(
            os.environ.get(
                "DATA_DIR",
                r"/home/hpc/iwi7/iwi7101h/i7-IDS/results/adversarial_attacks/tabular/original_cnn",
            )
        )
    )
    dataset = AdversarialDataPair(config)
    train_dataset, test_dataset = dataset.load_data()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of getting an item
    input_img, target_img, label = train_dataset[0]
    print(
        f"Input image shape: {input_img.shape}, Target image shape: {target_img.shape}"
    )

    train_dataset = TorchPairDataset(train_dataset)
    test_dataset = TorchPairDataset(test_dataset)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    input_img, target_img = train_dataset[0]
    print(
        f"Input image shape: {input_img.shape}, Target image shape: {target_img.shape}"
    )
    print("Done!!")
