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
from ids_expt.models.ffnn import FFNN, DNP3DNN
from ids_expt.core.defs import TOP_CIC_FEATURES,TOP_FEATURES
from ids_expt.data.dataset import (
        DataSetConfig,
        SamplingMethod,
        CLFDataSet,
        DFDataSet,
    )
from sklearn.model_selection import train_test_split
import os
from ids_expt.data.adversarial_tabular_data_pair import (
    AdversarialDataPairConfig,
    TorchPairDataset,
    AdversarialDataPair,
)
from tqdm import tqdm
import sys

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
    default=12800,
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
    "--data_dir",
    type=str,
    default=os.environ.get("DATA_DIR", "/home/hpc/iwi7/iwi7101h/i7-IDS/results/adversarial_attacks/original_cnn"),
    help="Path to the data directory.",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["cnn", "fnn"],
    default="cnn",
    help="Model type to use for training. Options are 'cnn' or 'fnn'.",
)

args = parser.parse_args()
batch_size = args.batch_size
result_dir = Path(f"{args.project_dir}/results/tabular_classification/original_adv_{args.model}")
if not result_dir.exists():
    result_dir.mkdir(parents=True, exist_ok=True)
logger.add(result_dir / "trainer.log", rotation="1 MB", level="INFO")



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

data_config = AdversarialDataPairConfig(
data_dir=args.data_dir, 
num_samples_per_epoch=args.max_data
)
train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()
val_ds.config.num_samples_per_epoch = int(args.max_data * 0.15)

train_dataset=TorchPairDataset(train_ds)
val_dataset=TorchPairDataset(val_ds)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size)
for features,_, labels in train_loader:
    logger.info(f"Features: {features.shape}, Labels: {labels}")
    # logger.info(f"Encoding labels: {train_dataset.class_encoder}")
    break


input_dim = train_dataset[0][0].shape[0]
# input_dim = len(train_dataset.config.features)
output_dim = train_dataset[0][2].shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "cnn":
    model = CNN1D(input_size=input_dim, output_size=output_dim)
elif args.model == "fnn":
    model = FFNN(input_size=input_dim, output_size=output_dim)
model.to(device)


# --- Training setup ---
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)
num_epochs = 1000
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
    # for features,_, labels in train_loader:
    for features, _, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
    disable=not sys.stdout.isatty(), ):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs,_ = model(features)
        batch_outputs.extend(outputs.cpu().detach().numpy().argmax(axis=1).tolist())
        batch_targets.extend(labels.cpu().detach().numpy().argmax(axis=1).tolist())

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
        for features, _, labels in test_loader:
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

    metrics['test_loss'].append(loss)
    metrics['test_acc'].append(acc)
    metrics['test_f1'].append(f1)

    df = pd.DataFrame(metrics)
    df.to_csv(result_dir / "metrics.csv", index=False)
logger.info(f"Training complete. Metrics saved to {result_dir / 'metrics.csv'}")

