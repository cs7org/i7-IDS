from pydantic import BaseModel, Field
from ids_expt.core.defs import DataType, SamplingMethod
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
            ("basiciterativemethod_eps_0.1", 0.4),
            ("fastgradientmethod_eps_0.1", 0.3),
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
    validation_ratio: float = Field(
        default=0.15,
        description="Ratio of the dataset to be used for validation. If -1, no validation set will be created.",
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.OVERSAMPLE,
        description="Sampling method to use for selecting samples.",
    )
    data_labels: list[str] = Field(
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
        description="List of labels to use for training. If empty, all labels will be used.",
    )
    apply_noise_rate: float = Field(
        default=0.7,
        description="Rate of applying noise to the input images. 0 means no noise.",
    )
    noise_range: tuple[float, float] = Field(
        default=(-0.01, 0.01),
        description="Range of noise to apply to the input images. Used only if apply_noise_rate > 0.",
    )
    combine_attacks: bool = Field(
        default=False,
        description="If True, combine all attacks into one class. If False, keep them separate.",
    )
    attack_only: bool = Field(
        default=False,
        description="If True, use only attack samples for training. If False, use both normal and attack samples.",
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

        if self.config.attack_only:
            self.config.data_labels = [
                l for l in self.config.data_labels if l != "NORMAL"
            ]
        if self.config.combine_attacks:
            self.config.data_labels = ["ATTACK", "NORMAL"]
        self.label_counts = {}
        self.label_encoding = {
            label: idx for idx, label in enumerate(config.data_labels)
        }
        for label in self.label_encoding.keys():
            lbl = [0] * len(self.label_encoding)
            idx = self.label_encoding[label]
            lbl[idx] = 1
            self.label_encoding[label] = lbl
        logger.info(f"Label encoding: {self.label_encoding}")

    def load_data(self):
        # load file names
        data_dir = self.config.data_dir
        # fill adversarial type npz files for training
        # get label counts from file name as: "{lbl_str}_{batch_idx}_{i}.npz"
        for adv_type, selection_rate in self.adversarial_type_selection_rate:
            adv_npz_files = list(data_dir.glob(f"adv_samples/{adv_type}/train/*.npz"))
            if not adv_npz_files:
                logger.warning(f"No files found for adversarial type: {adv_type}")
                continue
            label_files = {}
            for file in adv_npz_files:
                for label_str in self.config.data_labels:
                    if self.config.combine_attacks:
                        if "NORMAL" in file.stem:
                            # this file belongs to normal label
                            if "NORMAL" not in label_files:
                                label_files["NORMAL"] = []
                            label_files["NORMAL"].append(file)
                            break
                        else:
                            # this file belongs to attack label
                            if "ATTACK" not in label_files:
                                label_files["ATTACK"] = []
                            label_files["ATTACK"].append(file)
                            break

                    else:
                        if label_str in file.stem:
                            # this file belongs to this label
                            if label_str not in label_files:
                                label_files[label_str] = []
                            label_files[label_str].append(file)
                            break

            label_counts = {label: len(files) for label, files in label_files.items()}
            logger.info(f"Label counts: {label_counts} for {adv_type}")
            max_label_count = max(label_counts.values())
            new_label_counts = {
                label: len(files) for label, files in label_files.items()
            }
            self.adversarial_type_npz_files[adv_type] = []
            logger.info(
                f"Adversarial Type: {adv_type}, Selection Rate: {selection_rate}"
            )

            if self.config.sampling_method == SamplingMethod.OVERSAMPLE:
                # oversample to max_label_count
                for label, files in label_files.items():
                    if len(files) < max_label_count:
                        # oversample
                        files = self.random_state.choice(
                            files, size=max_label_count, replace=True
                        )
                    self.adversarial_type_npz_files[adv_type].extend(files)
                    new_label_counts[label] = len(files)
            elif self.config.sampling_method == SamplingMethod.UNDER_SAMPLE:
                # under sample to min_label_count
                min_label_count = min(label_counts.values())
                for label, files in label_files.items():
                    if len(files) > min_label_count:
                        # under sample
                        files = self.random_state.choice(
                            files, size=min_label_count, replace=False
                        )
                    self.adversarial_type_npz_files[adv_type].extend(files)
                    new_label_counts[label] = len(files)
            else:
                # do notthing
                logger.warning(
                    f"Unknown sampling method: {self.config.sampling_method}. Using no sampling."
                )
                self.adversarial_type_npz_files[adv_type].extend(adv_npz_files)
            logger.info(
                f"Applied {self.config.sampling_method} and now new counts: {new_label_counts}."
            )
            for label, count in new_label_counts.items():
                if label not in self.label_counts:
                    self.label_counts[label] = 0
                self.label_counts[label] += count

        logger.info(
            f"Loaded adversarial files from {len(self.adversarial_type_npz_files)} types."
        )

        if self.config.attack_only:
            self.config.data_labels = [
                l for l in self.config.data_labels if l != "NORMAL"
            ]

        # make train data pair
        train_pair = AdversarialDataPair(
            config=self.config.copy(),
        )
        train_pair.data_type = DataType.TRAIN
        train_pair.adversarial_type_npz_files = self.adversarial_type_npz_files
        train_pair.label_counts = self.label_counts

        # fill adversarial type npz files for validation
        self.adversarial_type_npz_files = {
            adv_type: [] for adv_type, _ in self.adversarial_type_selection_rate
        }
        self.label_counts = {}
        for adv_type, selection_rate in self.adversarial_type_selection_rate:
            adv_npz_files = list(data_dir.glob(f"adv_samples/{adv_type}/val/*.npz"))
            if not adv_npz_files:
                logger.warning(f"No files found for adversarial type: {adv_type}")
                continue
            self.adversarial_type_npz_files[adv_type].extend(adv_npz_files)
            label_files = {}
            for file in adv_npz_files:
                for label_str in self.config.data_labels:
                    if self.config.combine_attacks:
                        if "NORMAL" in file.stem:
                            # this file belongs to normal label
                            if "NORMAL" not in label_files:
                                label_files["NORMAL"] = []
                            label_files["NORMAL"].append(file)
                            break
                        else:
                            # this file belongs to attack label
                            if "ATTACK" not in label_files:
                                label_files["ATTACK"] = []
                            label_files["ATTACK"].append(file)
                            break

                    else:
                        if label_str in file.stem:
                            # this file belongs to this label
                            if label_str not in label_files:
                                label_files[label_str] = []
                            label_files[label_str].append(file)
                            break
            label_counts = {label: len(files) for label, files in label_files.items()}
            logger.info(f"Validation Label counts: {label_counts} for {adv_type}")
            self.label_counts = {
                label: self.label_counts.get(label, 0) + count
                for label, count in label_counts.items()
            }

        # make validation data pair
        val_pair = AdversarialDataPair(
            config=self.config.copy(),
        )
        val_pair.data_type = DataType.VALIDATION
        val_pair.adversarial_type_npz_files = self.adversarial_type_npz_files
        val_pair.label_counts = self.label_counts

        return train_pair, val_pair

    def __len__(self):
        # if self.data_type == DataType.TRAIN:
        return self.config.num_samples_per_epoch
        # else:
        #     return int(self.config.validation_ratio * self.config.num_samples_per_epoch)

    def __getitem__(self, idx):
        # randomly select an adversarial type based on selection rates
        adv_type = self.random_state.choice(
            [adv_type for adv_type, _ in self.adversarial_type_selection_rate],
            p=[rate for _, rate in self.adversarial_type_selection_rate],
        )
        selected_files = self.adversarial_type_npz_files[adv_type]
        if not selected_files:
            raise ValueError(f"No files found for adversarial type: {adv_type}")

        selected_file = self.random_state.choice(selected_files)
        # logger.debug(f"Selected file: {adv_type}, {selected_file}")
        # load the npz file
        npz_data = np.load(selected_file)
        input_image = npz_data["inputs"]
        adversarial_image = npz_data["adversarial"]
        label = npz_data["label_str"].item()

        if label not in self.label_encoding:
            return self.__getitem__(self.random_state.randint(len(self)))

        if self.config.combine_attacks:
            if "NORMAL" in label:
                label = "NORMAL"
            else:
                label = "ATTACK"

        # our input will be adversarial image and target will be clean image
        target_img = input_image.copy()
        input_image = adversarial_image.copy()
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
        if (
            # self.data_type == DataType.TRAIN
            # and
            self.config.apply_noise_rate > 0
            and self.random_state.rand() < self.config.apply_noise_rate
        ):
            noise_level = self.config.noise_range[1]
            mean_pix = input_image.mean()
            noise = self.random_state.normal(0, noise_level, input_image.shape) * 255
            noise = np.clip(noise, -mean_pix, mean_pix)
            # print(noise.min(), noise.max())

            input_image = input_image.astype(float)
            input_image += noise
            input_image = np.clip(input_image, 0, 255)
            input_image = input_image.astype(np.uint8)
        # normalize by 255
        input_image = input_image.astype(np.float32) / 255.0
        target_img = target_img.astype(np.float32) / 255.0
        return input_image, target_img, label


class TorchPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: AdversarialDataPair):
        self.dataset = dataset
        self.config = dataset.config
        self.num_classes = len(dataset.label_encoding)
        self.current_label = None
        self.label_counts = dataset.label_counts
        self.label_encoding = dataset.label_encoding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_img, target_img, label = self.dataset[idx]
        self.current_label = label
        lbl_encoding = self.dataset.label_encoding[label]
        label_tensor = torch.tensor(lbl_encoding, dtype=torch.float)

        self.data_kind = self.dataset.data_kind
        return (
            torch.tensor(input_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target_img, dtype=torch.float32).unsqueeze(0),
            label_tensor,
        )


if __name__ == "__main__":
    from ids_expt.data.adversarial_data_pair import (
        AdversarialDataPairConfig,
        AdversarialDataPair,
    )
    import os

    config = AdversarialDataPairConfig(
        data_dir=Path(
            os.environ.get("DATA_DIR", "data/adv_samples"),
        ),
        attack_only=True,
        # combine_attacks=True,
    )
    dataset = AdversarialDataPair(config)
    train_dataset, test_dataset = dataset.load_data()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example of getting an item
    input_img, target_img, label_tensor = TorchPairDataset(train_dataset)[0]
    print(
        f"Input image shape: {input_img.shape}, Target image shape: {target_img.shape}, Label: {label_tensor.shape}"
    )
