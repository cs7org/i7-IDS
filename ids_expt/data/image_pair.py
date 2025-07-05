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
    session_images_dir: Path = Field(
        default=Path(
            r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions\session_images"
        ),
        description="Root directory of original images",
    )
    labelled_csv: Path = Field(
        default=Path(
            r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions\labelled_sessions.csv"
        ),
        description="CSV file with labelled sessions",
    )
    adversarial_images_dir: list[Path, float] = Field(
        default=[
            (
                Path(
                    r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions\adversarial\normal\fastgradientmethod_eps_0.1"
                ),
                0.25,
            ),
            (
                Path(
                    r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions\adversarial\normal\fastgradientmethod_eps_0.01"
                ),
                0.25,
            ),
            (
                Path(
                    r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions\adversarial\normal\basiciterativemethod_eps_0.1"
                ),
                0.25,
            ),
            (
                Path(
                    r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions\adversarial\normal\basiciterativemethod_eps_0.01"
                ),
                0.25,
            ),
        ],
        description="List of tuples containing adversarial image directories and their selection rate",
    )
    clean_selection_rate: float = Field(
        default=0.5,
        description="Selection rate for clean images",
    )
    train_ratio: float = Field(
        default=0.8,
        description="Ratio of data to use for training",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    num_pkts: int = Field(
        default=192,
        description="Number of packets to consider for each session (Height)",
    )
    byte_length: int = Field(
        default=256,
        description="Byte length of each packet (Width)",
    )
    max_data: int = Field(
        default=-1,
        description="Maximum number of data points to load. -1 means no limit.",
    )
    is_normalized: bool = Field(
        default=False,
        description="If using normalized images",
    )
    min_num_pkts: int = Field(
        default=1,
        description="Minimum number of packets required to include a session in the dataset",
    )
    num_samples_per_epoch: int = Field(
        default=-1,
        description="Number of samples to use per epoch for training. -1 means use all available data.",
    )


class AdversarialDataPair:
    def __init__(self, config: AdversarialDataPairConfig = AdversarialDataPairConfig()):
        self.config = config
        self.labelled_csv = config.labelled_csv
        self.clean_selection_rate = config.clean_selection_rate
        self.random_seed = config.random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.num_pkts = config.num_pkts
        self.byte_length = config.byte_length
        self.train_ratio = self.config.train_ratio
        self.data_type = None
        self.data_df = None
        self.train_df = None
        self.test_df = None
        self.adv_not_found_images = []  # to keep track of adversarial images not found

    def load_data(self):
        data_df = pd.read_csv(self.labelled_csv)
        data_df = data_df.query(f"total_matched_pkts>={self.config.min_num_pkts}")

        self.data_df = data_df.copy()
        self.data_df["label"] = self.data_df.flow_label
        # it does notr have file_path col
        if self.config.is_normalized:
            logger.info("Using normalized images for training")
            self.data_df["file_path"] = self.data_df.apply(
                lambda row: self.config.session_images_dir
                / f"{row.session_file_name.replace('.pcap', '_normalized.png')}",
                axis=1,
            )
        else:
            self.data_df["file_path"] = self.data_df.apply(
                lambda row: self.config.session_images_dir
                / f"{row.session_file_name.replace('.pcap', '.png')}",
                axis=1,
            )

        if self.config.max_data > 0:
            logger.info(f"Limiting dataset to {self.config.max_data} samples per class")
            self.data_df = self.data_df.groupby("label").head(self.config.max_data)
        logger.info(
            f"Final dataset size: {len(self.data_df)} entries after applying max_data limit"
        )
        train_df, test_df = train_test_split(
            self.data_df,
            train_size=self.train_ratio,
            stratify=self.data_df["label"],
            random_state=self.config.random_seed,
        )
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        logger.info(
            f"Split data into {len(self.train_df)} training and {len(self.test_df)} testing samples"
        )
        train_dataset = AdversarialDataPair(self.config)
        train_dataset.data_df = self.train_df
        train_dataset.data_type = DataType.TRAIN

        test_dataset = AdversarialDataPair(self.config)
        test_dataset.data_df = self.test_df
        test_dataset.data_type = DataType.VALIDATION
        logger.info(f"Created datasets: {train_dataset} and {test_dataset}")
        logger.info(
            f"{train_dataset.data_type}: {train_dataset.data_df.label.value_counts()}"
        )
        return train_dataset, test_dataset

    def __len__(self):
        if self.data_df is not None:
            if self.config.num_samples_per_epoch > 0:
                return min(len(self.data_df), self.config.num_samples_per_epoch)
            else:
                return len(self.data_df)

    def __getitem__(self, idx):
        if self.data_df is not None:
            row = self.data_df.iloc[idx]
        image_path = Path(row["file_path"])
        label = row["label"]
        lbl_string = row.label

        if image_path.exists():
            img = cv2.imread(str(image_path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray_img.shape
            # check num pkts
            if h < self.config.num_pkts:
                # pad with zeros on bottom
                pad_height = self.config.num_pkts - h
                gray_img = cv2.copyMakeBorder(
                    gray_img, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0
                )

            elif h > self.config.num_pkts:
                gray_img = gray_img[: self.config.num_pkts, :]
            # check byte length but not for normalized bcz normalized has only 256 cols
            if not self.config.is_normalized:
                if w < self.config.byte_length:
                    # pad with zeros
                    pad_width = self.config.byte_length - w
                    gray_img = cv2.copyMakeBorder(
                        gray_img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0
                    )
                elif w > self.config.byte_length:
                    gray_img = gray_img[:, : self.config.byte_length]
            target_image = gray_img.copy()
            # above image is the target image now find the input image
            input_img = target_image.copy()
            if (
                self.config.clean_selection_rate > self.random_state.rand()
                or image_path.name in self.adv_not_found_images
            ):
                # select original image
                pass
            else:
                # select adversarial image
                adv_roots = [
                    adv_dir for adv_dir, _ in self.config.adversarial_images_dir
                ]
                probabilities = [prob for _, prob in self.config.adversarial_images_dir]

                adv_img_path = self.random_state.choice(adv_roots, p=probabilities)
                adv_img_path = adv_img_path / image_path.name
                if adv_img_path.exists():
                    input_img = cv2.imread(str(adv_img_path), cv2.IMREAD_GRAYSCALE)

                else:
                    # try to find image in any adversarial directory
                    input_img = None
                    for adv_dir in adv_roots:
                        temp_path = adv_dir / image_path.name
                        if temp_path.exists():
                            input_img = cv2.imread(str(temp_path), cv2.IMREAD_GRAYSCALE)
                            break
                    if input_img is None:
                        # use the original image if adversarial image not found
                        logger.warning(
                            f"Adversarial image not found for {image_path}. Using original image."
                        )
                        self.adv_not_found_images.append(image_path.name)
                        input_img = gray_img.copy()
            input_img = input_img / 255
            target_image = target_image / 255
            return input_img, target_image

        raise IndexError("DataFrame is empty or not loaded.")


class TorchPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: AdversarialDataPair):
        self.dataset = dataset
        self.config = dataset.config
        self.num_classes = len(dataset.data_df.label.unique())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_img, target_img = self.dataset[idx]
        return (
            torch.tensor(input_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target_img, dtype=torch.float32).unsqueeze(0),
        )
