from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from ids_expt.core.defs import Session
from scapy.all import raw
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset as TorchDataset
import torch
from pydantic import BaseModel, Field
from pathlib import Path
from ids_expt.core.defs import DataType, SamplingMethod


class PCAPImageDataConfig(BaseModel):
    data_root: Path = Field(
        default=Path(r"E:\MSc Works\IDS\notebooks\output"),
        description="Root directory for storing PCAP images.",
    )
    train_ratio: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for training.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    labels: list[str] = Field(
        default=[],
        description="List of labels to be used for classification.",
    )
    normal_label: str = Field(
        default="NORMAL",
        description="Label for normal data samples.",
    )
    combine_attacks: bool = Field(
        default=False,
        description="Whether to combine all attack types into a single label.",
    )
    max_data: int = Field(
        default=-1,
        description="Maximum number of samples to be used from each class.",
    )
    byte_length: int = Field(
        default=8 * 32,
        ge=1,
        description="Length of byte sequences to be used for image generation.",
    )
    num_pkts: int = Field(
        default=6 * 32,
        ge=1,
        description="Number of packets to consider for each sample.",
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.NONE,
        description="Method to use for sampling data.",
    )


def image_normalize(image):
    """Normalize image pixel values to [0, 1] range."""
    return image / 255.0


class DFDataSet:
    def __init__(self, config: PCAPImageDataConfig):
        self.config = config
        self.data_root = config.data_root
        self.train_ratio = config.train_ratio
        self.data_df = None
        self.data_type = None
        self.label_encoding = None
        self.scaler = image_normalize

    def read_statistics(self):
        self.all_sessions = list(self.data_root.rglob("*.pkl"))
        logger.info(f"Found {len(self.all_sessions)} session files in {self.data_root}")
        stats = []
        session_index = 0
        for session_file in tqdm(self.all_sessions):
            # session_file = self.all_sessions[43]
            with open(session_file, "rb") as f:
                try:
                    sessions: list[Session] = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error reading {session_file}: {e}")
                    continue
            for sidx, session in enumerate(sessions):
                session_pkts = session.packets
                num_pkts = len(session_pkts)
                label = session.label
                duration = session.duration
                pkt_lens = []
                for pkt in session_pkts:
                    pkt_raw = raw(pkt)
                    pkt_lens.append(len(pkt_raw))
                min_len = min(pkt_lens) if pkt_lens else 0
                max_len = max(pkt_lens) if pkt_lens else 0
                stats.append(
                    {
                        "session_file": session_file.name,
                        "session_index": session_index,
                        "num_packets": num_pkts,
                        "label": label,
                        "duration": duration,
                        "min_packet_length": min_len,
                        "max_packet_length": max_len,
                    }
                )
                session_index += 1
        self.stats_df = pd.DataFrame(stats)
        return self.stats_df

    def load_data(self):
        self.all_files = list(self.data_root.rglob("*.png"))
        logger.info(f"Found {len(self.all_files)} files in {self.data_root}")
        self.all_files = [
            f for f in self.all_files if f.is_file() and "hot" not in f.stem
        ]
        logger.info(
            f"Filtered to {len(self.all_files)} files after removing 'hot' images"
        )
        data = []
        for file in self.all_files:
            label = file.stem.replace("session_", "").split("_")[1:]
            label = "_".join(label)
            data.append({"file_path": str(file), "label": label})
        self.data_df = pd.DataFrame(data)
        if self.config.combine_attacks:
            self.data_df["label"] = self.data_df["label"].apply(
                lambda x: (
                    self.config.normal_label
                    if x == self.config.normal_label
                    else "ATTACK"
                )
            )
            logger.info("Combined all attack types into a single label")
        logger.info(f"Data loaded into DataFrame with {len(self.data_df)} entries")
        logger.info(f"Label distribution:\n{self.data_df['label'].value_counts()}")

        if self.config.max_data > 0:
            logger.info(f"Limiting dataset to {self.config.max_data} samples per class")
            self.data_df = self.data_df.groupby("label").head(self.config.max_data)
        logger.info(
            f"Final dataset size: {len(self.data_df)} entries after applying max_data limit"
        )
        labels = self.data_df["label"].unique().tolist()
        self.label_encoding = {label: idx for idx, label in enumerate(labels)}
        for label in self.label_encoding.keys():
            lbl = [0] * len(self.label_encoding)
            idx = self.label_encoding[label]
            lbl[idx] = 1
            self.label_encoding[label] = lbl

        train_df, test_df = train_test_split(
            self.data_df,
            train_size=self.train_ratio,
            stratify=self.data_df["label"],
            random_state=self.config.random_seed,
        )
        min_labels = train_df["label"].value_counts().min()
        max_labels = train_df["label"].value_counts().max()
        if self.config.sampling_method == SamplingMethod.UNDERSAMPLE:
            logger.info(f"Undersampling to {min_labels} samples per class for training")
            train_df = (
                train_df.groupby("label")
                .apply(
                    lambda x: x.sample(min_labels, random_state=self.config.random_seed)
                )
                .reset_index(drop=True)
            )
        elif self.config.sampling_method == SamplingMethod.OVERSAMPLE:
            logger.info(f"Oversampling to {max_labels} samples per class for training")
            new_df = train_df.query(f"label == '{self.config.normal_label}'")
            for label in train_df["label"].unique():
                if label == self.config.normal_label:
                    continue
                label_df = train_df.query(f"label == '{label}'")
                if len(label_df) < max_labels:
                    num_samples = max_labels
                    sampled_df = label_df.sample(
                        num_samples, replace=True, random_state=self.config.random_seed
                    )
                    new_df = pd.concat([new_df, sampled_df], ignore_index=True)
            train_df = new_df
        elif self.config.sampling_method == SamplingMethod.NONE:
            logger.info("No sampling applied, using original training data")
        else:
            raise ValueError(
                f"Unsupported sampling method: {self.config.sampling_method}"
            )

        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        logger.info(
            f"Split data into {len(self.train_df)} training and {len(self.test_df)} testing samples"
        )
        train_dataset = DFDataSet(self.config)
        train_dataset.data_df = self.train_df
        train_dataset.data_type = DataType.TRAIN
        train_dataset.label_encoding = self.label_encoding

        test_dataset = DFDataSet(self.config)
        test_dataset.data_df = self.test_df
        test_dataset.data_type = DataType.VALIDATION
        test_dataset.label_encoding = self.label_encoding
        logger.info(f"Created datasets: {train_dataset} and {test_dataset}")
        logger.info(
            f"{train_dataset.data_type}: {train_dataset.data_df.label.value_counts()}"
        )

        return train_dataset, test_dataset

    def __len__(self):
        if self.data_df is not None:
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
            if h < self.config.num_pkts:
                # pad with zeros
                pad_height = self.config.num_pkts - h
                gray_img = cv2.copyMakeBorder(
                    gray_img, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0
                )
            elif h > self.config.num_pkts:
                gray_img = gray_img[: self.config.num_pkts, :]
            if w < self.config.byte_length:
                # pad with zeros
                pad_width = self.config.byte_length - w
                gray_img = cv2.copyMakeBorder(
                    gray_img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0
                )
            elif w > self.config.byte_length:
                gray_img = gray_img[:, : self.config.byte_length]
            label = self.label_encoding[label] if self.label_encoding else label
            return gray_img, label, lbl_string

        raise IndexError("DataFrame is empty or not loaded.")

    def __repr__(self):
        return f"DataSet(data_root={self.data_root}, train_ratio={self.train_ratio})"


class TorchImageDataset(TorchDataset):
    def __init__(self, dataset: DFDataSet):
        self.dataset = dataset
        self.config = dataset.config
        self.data = dataset.data_df
        self.label_encoding = dataset.label_encoding
        self.data_type = dataset.data_type
        self.num_classes = len(self.label_encoding)
        # Get class counts
        class_counts = self.data["label"].value_counts().to_dict()

        # Compute inverse frequency weights
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}

        # Order weights according to label encoding
        weights_list = [class_weights[label] for label in self.label_encoding.keys()]

        # Convert to tensor (no need to normalize)
        self.class_weights = torch.tensor(weights_list, dtype=torch.float32)
        self.class_weights = (
            self.class_weights * len(self.class_weights) / self.class_weights.sum()
        )
        self.label_counts = class_counts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, _ = self.dataset[idx]
        tensor = torch.from_numpy(image).float() / 255
        tensor = tensor.unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return tensor, label_tensor

    def __repr__(self):
        return f"TorchImageDataset(dataset={self.dataset})"
