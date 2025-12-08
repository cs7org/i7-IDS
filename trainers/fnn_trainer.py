import torch
import torch.nn as nn
import torch.optim as optim
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
from ids_expt.models.ffnn import FFNN
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Train a DNP3 DNN model")
parser.add_argument(
    "--data_type",
    type=str,
    choices=["synthetic", "original"],
    default="original",
    help="Type of data to use for training.",
)
parser.add_argument(
    "--max_data",
    type=int,
    default=-1,
    help="Maximum number of data points to use. Use -1 for all data.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1000,
    help="Number of epochs for training.",
)
parser.add_argument(
    "--project_dir",
    type=str,
    default="/home/hpc/iwi7/iwi7101h/i7-IDS",
    help="Directory containing the project files.",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["cnn", "fnn"],
    default="fnn",
    help="Model type to use for training. Options are 'cnn' or 'fnn'.",
)

args = parser.parse_args()
result_dir = Path(f"{args.project_dir}/results/tabular_classification/original_{args.model}")
if not result_dir.exists():
    result_dir.mkdir(parents=True, exist_ok=True)
logger.add(result_dir / "trainer.log", rotation="1 MB", level="INFO")

def oversample_class(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Oversample the specified class in the DataFrame.
    """
    logger.info(f"Before oversampling: {df[label].value_counts()}")
    label_counts = df[label].value_counts()
    max_count = label_counts.max()
    for lbl in label_counts.index:
        count = label_counts[lbl]
        if count < max_count:
            needed = max_count - count
            logger.info(f"Label {lbl} needs {needed} samples")
            oversample_df = df[df[label] == lbl].sample(needed, replace=True)
            df = pd.concat([df, oversample_df], ignore_index=True)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)
    logger.info(f"After oversampling: {df[label].value_counts()}")
    return df


def undersample_class(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Undersample the specified class in the DataFrame.
    """
    logger.info(f"Before undersampling: {df[label].value_counts()}")
    label_counts = df[label].value_counts()
    min_count = label_counts.min()
    for lbl in label_counts.index:
        count = label_counts[lbl]
        if count > min_count:
            needed = count - min_count
            logger.info(f"Label {lbl} needs to be reduced by {needed} samples")
            undersample_df = df[df[label] == lbl].sample(needed, replace=False)
            df = df.drop(undersample_df.index)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)
    logger.info(f"After undersampling: {df[label].value_counts()}")
    return df

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

csv_file = Path(args.project_dir) / "data/cicflow_combined.csv"
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
batch_size = args.batch_size
class_weights = {
    "REPLAY": 2.3268327713012695,
    "DNP3_INFO": 0.6492278575897217,
    "DNP3_ENUMERATE": 0.5834585428237915,
    "STOP_APP": 2.2812085151672363,
    "NORMAL": 0.10969416797161102,
    "INIT_DATA": 2.241649866104126,
    "COLD_RESTART": 0.26930931210517883,
    "WARM_RESTART": 0.26930931210517883,
    "DISABLE_UNSOLICITED": 0.26930931210517883,
}
class_weights = {k: 1 for k, v in class_weights.items()}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size)
for features, labels in train_loader:
    logger.info(f"Features: {features.shape}, Labels: {labels}")
    # logger.info(f"Encoding labels: {train_dataset.class_encoder}")
    break


input_dim = train_dataset.features.shape[1]
# input_dim = len(train_dataset.config.features)
# output_dim = train_dataset.labels.shape[1]
output_dim = len(class_weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "cnn":
    model = CNN1D(input_size=input_dim, output_size=output_dim)
elif args.model == "fnn":
    model = FFNN(input_size=input_dim, output_size=output_dim)

model.to(device)

# --- Training setup ---
criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights.values()), device=device)) 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)
num_epochs = 2000
log_every = 1
# --- Training loop ---
train_losses = []
train_accs = []

test_losses = []
test_accs = []
metrics = dict(epoch=[], train_loss=[], train_acc=[], test_loss=[], test_acc=[], test_f1=[], train_f1=[])
min_loss = float('inf')
max_epoch = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    acc = 0.0
    f1 = 0.0
    batch_outputs = []
    batch_targets = []
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs,_ = model(features)
        batch_outputs.extend(outputs.cpu().detach().numpy().argmax(axis=1).tolist())
        batch_targets.extend(labels.cpu().detach().numpy().argmax(axis=1).tolist())
        # acc += (
        #     (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float().sum()
        # )

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss = running_loss / len(train_loader.dataset)
    acc = np.mean(np.array(batch_outputs) == np.array(batch_targets))
    f1 = f1_score(batch_targets, batch_outputs, average='macro')
    metrics['epoch'].append(epoch)
    metrics['train_loss'].append(loss)
    metrics['train_acc'].append(acc)
    metrics['train_f1'].append(f1)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

    # --- Evaluation ---
    model.eval()
    batch_outputs = []
    batch_targets = []
    running_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs,_ = model(features)
            batch_outputs.extend(outputs.cpu().detach().numpy().argmax(axis=1).tolist())
            batch_targets.extend(labels.cpu().detach().numpy().argmax(axis=1).tolist())
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    loss = running_loss / len(test_loader.dataset)
    acc = np.mean(np.array(batch_outputs) == np.array(batch_targets))
    f1 = f1_score(batch_targets, batch_outputs, average='macro')
    
    scheduler.step(loss)
    logger.info(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    if loss < min_loss:
        min_loss = loss
        torch.save(model.state_dict(), result_dir / "best_model.pth")
        logger.info(f"New best model saved at {result_dir} epoch {epoch} with F1 {f1:.4f}")
        torch.save(model, result_dir / "best_model_full.pth")
    torch.save(model.state_dict(), result_dir / "last_model.pth")
    torch.save(model, result_dir / "last_model_full.pth")
    metrics['test_loss'].append(loss)
    metrics['test_acc'].append(acc)
    metrics['test_f1'].append(f1)

    df = pd.DataFrame(metrics)
    df.to_csv(result_dir / "metrics.csv", index=False)
logger.info(f"Training complete. Metrics saved to {result_dir / 'metrics.csv'}")

