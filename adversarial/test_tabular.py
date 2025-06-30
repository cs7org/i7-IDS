import numpy as np
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ids_expt.core.defs import TOP_CIC_FEATURES
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from pathlib import Path
import torch
from ids_expt.models.ffnn import FFNN
from ids_expt.utils.confusion_matrix import get_confusion_matrix
from tqdm import tqdm

model_path = Path(
    r"C:\Users\Viper\Desktop\thesis_code\results\cic_fnn\ctgan_oversampling\best_model.pth"
)


class ClfModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ClfModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


EPSILONS = [0.0001, 0.001, 0.01, 0.1]
train_dataset, val_dataset = DFDataSet(
    config=DataSetConfig(
        csv_path=Path(r"E:\MSc Works\IDS\data\cicflow_combined.csv"),
        features=TOP_CIC_FEATURES,
        sampling_method=SamplingMethod.NONE,
        max_data=-1,
        train_ratio=0.8,
    )
).get_datasets()

model = FFNN(
    input_size=len(TOP_CIC_FEATURES),
    hidden_layers=[90] * 9,
    output_size=val_dataset.data.Label.nunique(),
    use_batchnorm=False,
)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model = ClfModel(model)
model.to("cuda" if torch.cuda.is_available() else "cpu")
# model.eval()

train_df = pd.read_csv(r"E:\MSc Works\IDS\cic_merged_train_data.csv")
train_df.columns = train_df.columns.str.strip()
val_df = pd.read_csv(r"E:\MSc Works\IDS\cic_merged_test_data.csv")
val_df.columns = val_df.columns.str.strip()
X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = val_df.drop(columns=["Label"])
y_val = val_df["Label"]
X_val = scaler.transform(X_val)
train_dataset.data = pd.DataFrame(X_train, columns=train_df.columns[:-1])
train_dataset.data["Label"] = y_train.values
val_dataset.data = pd.DataFrame(X_val, columns=val_df.columns[:-1])
val_dataset.data["Label"] = y_val.values

# x_test = X_val.astype(np.float32).reshape(
#     len(X_val), 1, len(TOP_CIC_FEATURES)
# )  # Reshape for PyTorch input
# y_test = np.array([val_dataset.label_encoding[label] for label in y_val]).astype(
#     np.float32
# )
val_dataset = DataSet(val_dataset)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

predictions = []
targets = []

for x, y in tqdm(val_loader):
    logits = model(x.to("cuda"))
    probs = torch.nn.functional.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    predictions.extend(preds.cpu().numpy())
    targets.extend(y.argmax(dim=1).cpu().numpy())
get_confusion_matrix(
    predictions,
    targets,
    label_keys=list(val_dataset.label_encoding.keys()),
    out_file=model_path.parent / "confusion_matrix.png",
)

for epsilon in EPSILONS:
    logger.info(f"Evaluating adversarial examples with epsilon: {epsilon}")
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(1, len(TOP_CIC_FEATURES)),
        nb_classes=val_dataset.data.Label.nunique(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )
    attack = FastGradientMethod(
        estimator=classifier,
        eps=epsilon,
    )
    adv_predictions = []
    adv_targets = []
    for x, y in tqdm(val_loader):
        adv_x = attack.generate(x=x.numpy())
        adv_logits = model(torch.tensor(adv_x).to("cuda"))
        adv_probs = torch.nn.functional.softmax(adv_logits, dim=1)
        adv_preds = torch.argmax(adv_probs, dim=1)
        adv_predictions.extend(adv_preds.cpu().numpy())
        adv_targets.extend(y.argmax(dim=1).cpu().numpy())

    get_confusion_matrix(
        adv_predictions,
        adv_targets,
        label_keys=list(val_dataset.label_encoding.keys()),
        out_file=model_path.parent / f"confusion_matrix_adv_{epsilon}.png",
        eps=epsilon,
    )
