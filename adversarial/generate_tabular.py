from pathlib import Path
from loguru import logger

import torch
from ids_expt.adversarial.adversarial_experiment import AdversarialExperiment, ClfModel
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod,MomentumIterativeMethod
from art.estimators.classification import PyTorchClassifier
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
import argparse
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ids_expt.models.cnn import CNN1D
from ids_expt.models.ffnn import FFNN, DNP3DNN
from ids_expt.core.defs import TOP_CIC_FEATURES,TOP_FEATURES,DataType
from ids_expt.data.dataset import (
        DataSetConfig,
        SamplingMethod,
        CLFDataSet,
        DFDataSet,
    )
from sklearn.model_selection import train_test_split

# Argument parser for project dir, max_data, and model paths
parser = argparse.ArgumentParser(
    description="Adversarial Tabular Data Generation Configuration"
)
parser.add_argument(
    "--project_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code",
    help="Directory containing the project files.",
)
parser.add_argument(
    "--max_data",
    type=int,
    default=-1000,
    help="Maximum number of data points to use. Use -ve for all data.",
)
parser.add_argument(
    "--models",
    type=str,
    default=[
        "fnn,cnn",
    ],
    help="Model names in comma separated.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for adversarial attack generation.",
)
parser.add_argument(
    "--data_type",
    type=str,
    choices=["original", "synthetic"],
    default="original",
    help="Type of data to use for training. Options are 'original' or 'synthetic'.",
)


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels: list[str],
        labels_key: str = "Label",
        split: str = "train",
        random_state: int = 42,
        max_data: int = -1,
        train_size: float = 0.75,
    ):
        self.train_size = train_size
        self.df = df
        class_names = labels
        self.class_names = class_names
        self.class_encoder = {name: i for i, name in enumerate(class_names)}

        self.df = self.df.sample(
            frac=1.0, random_state=random_state, replace=False
        ).reset_index(drop=True)

        if max_data > 0:
            self.df = self.df.iloc[:max_data]

    
        self.split = split
        self.random_state = random_state
        self.labels_key = labels_key
        self._split_data()
        self._one_hot_encode_labels()

    def _split_data(self):
        train_df, val_df = train_test_split(
            self.df,
            test_size=1 - self.train_size,
            random_state=self.random_state,
            stratify=self.df[self.labels_key],
        )
        # scale the features
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(train_df.drop(columns=[self.labels_key]))

        if self.split == "train":
            self.features = scaler.transform(train_df.drop(columns=[self.labels_key]))
            self.labels = train_df[self.labels_key].values
        else:
            self.features = scaler.transform(val_df.drop(columns=[self.labels_key]))
            self.labels = val_df[self.labels_key].values

    def _one_hot_encode_labels(self):
        labels = np.zeros((self.labels.shape[0], len(self.class_names)))
        for i, label in enumerate(self.labels):
            labels[i, self.class_encoder[label]] = 1
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx == len(self.labels) - 1:
            idxs = np.arange(len(self.labels))
            np.random.shuffle(idxs)
            self.features = self.features[idxs]
            self.labels = self.labels[idxs]

        return torch.from_numpy(self.features[idx]).to(torch.float32), torch.from_numpy(
            self.labels[idx]
        ).to(torch.float32)



args = parser.parse_args()
project_dir = Path(args.project_dir)
if not project_dir.exists():
    logger.error(f"Project directory {project_dir} does not exist.")
    exit(1)
model_names = args.models.split(",")
if not model_names:
    logger.error("No model names provided. Please specify at least one model.")
else:
    logger.info(f"Using models: {model_names}")

model_paths = [
    Path(project_dir)
    / f"results/tabular_classification/{model_name}/best_model_full.pth"
    for model_name in model_names
]
max_data = args.max_data
batch_size = args.batch_size

# If using synthetic data, use the CTGAN generated data.
if args.data_type == "synthetic":
    csv_file = project_dir / "data/cic_ctgan_merged_synthetic_data.csv"
    use_synthetic = True
else:
    csv_file = project_dir / "data/cicflow_combined.csv"
    use_synthetic = False
# overridding it fir now
csv_file = Path(args.project_dir) / "data/cicflow_combined.csv"
# csv_file = Path(args.project_dir) / "data/cic_ctgan_merged_synthetic_data.csv"
# combined_df = pd.read_csv(f'{args.project_dir}/data/combined_120_timeout.csv')
combined_df = pd.read_csv(csv_file)
combined_df.columns = combined_df.columns.str.strip()
labels = [
            "REPLAY",
            "DNP3_INFO",
            "DNP3_ENUMERATE",
            "STOP_APP",
            "NORMAL",
            "INIT_DATA",
            "COLD_RESTART",
            "WARM_RESTART",
            "DISABLE_UNSOLICITED",
        ]

combined_df = combined_df.query("Label in @labels")
logger.info(f"Initial DataFrame shape: {combined_df.shape}")
ignore_columns = [
    "File",
    "flow ID",
    "binary_label",
    "Timestamp",
    "source IP",
    "destination IP",
    "date",
    "Unnamed: 0",
    "Unnamed: 0.1",
    "firstPacketDIR",
]
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ignore_columns]+["Label"]
combined_df = combined_df[numeric_cols]
# remove nan, inf,-inf rows
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna()
logger.info(f"DataFrame shape after cleaning: {combined_df.shape}")
df = combined_df.copy()
logger.info(f"Filtered DataFrame shape: {df.shape}")
# df = oversample_class(df, "Label")
max_data = args.max_data 
train_dataset = CustomDataset(df, split="train", max_data=max_data, labels=labels)
val_dataset = CustomDataset(df, split="test", max_data=max_data, labels=labels)


for model_path in model_paths:
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist.")
        exit(1)

    val_dataset = val_dataset
    val_dataset.num_classes = len(labels)
    train_dataset.num_classes = len(labels)
    val_dataset.data_type = DataType.VALIDATION
    train_dataset.data_type = DataType.TRAIN
    train_dataset.label_encoding = {label: i for i, label in enumerate(labels)}
    val_dataset.label_encoding = {label: i for i, label in enumerate(labels)}
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )

    logger.info(f"Running adversarial attacks on model: {model_path.name}")
    epsilons = [0.001,0.01,0.1,0.2,0.3,0.5]
    iterations = 10
    input_shape = len(TOP_CIC_FEATURES)

    attacks = [
        FastGradientMethod(
            estimator=PyTorchClassifier(
                model=ClfModel(model),
                loss=torch.nn.CrossEntropyLoss(),
                clip_values=(0, 1),
                input_shape=input_shape,
                nb_classes=len(labels),
                optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            ),
            eps=eps,
        )
        for eps in epsilons
    ]
    attacks.extend(
        [
            BasicIterativeMethod(
                estimator=PyTorchClassifier(
                    model=ClfModel(model),
                    loss=torch.nn.CrossEntropyLoss(),
                    clip_values=(0, 1),
                    input_shape=input_shape,
                    nb_classes=len(labels),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                ),
                eps=eps,
                max_iter=iterations,
                verbose=False,
            )
            for eps in epsilons
        ]
    )
    attacks.extend(
        [
            MomentumIterativeMethod(
                estimator=PyTorchClassifier(
                    model=ClfModel(model),
                    loss=torch.nn.CrossEntropyLoss(),
                    clip_values=(0, 1),
                    input_shape=input_shape,
                    nb_classes=len(labels),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                ),
                eps=eps,
                max_iter=iterations,
                verbose=False,
            )
            for eps in epsilons
        ]
    )
    adv = AdversarialExperiment(
        model=model,
        model_name=model_path.parent.name,
        attacks=attacks,
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        output_dir=project_dir / "results" / "adversarial_attacks",
    )

    adv.run(
        results_dir=project_dir
        / "results"
        / "adversarial_attacks"
        / model_path.parent.name
    )

    logger.info("Adversarial attacks completed successfully.")
    logger.info("Generating adversarial attack data...")

    selected_attacks = [atk for atk in attacks]

    for attack in selected_attacks:
        logger.info(
            f"Generating adversarial data for attack: {attack.__class__.__name__} with eps: {attack.eps}"
        )
        out_folder = attack.__class__.__name__.lower() + f"_eps_{attack.eps}"
        adv.generate(attack, out_folder=out_folder, is_image_dataset=False)
        logger.info(f"Adversarial data generated and saved in {out_folder} folder.")
    logger.info("All adversarial data generation completed.")
