import numpy as np
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ids_expt.core.defs import TOP_CIC_FEATURES
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch
from ids_expt.models.ffnn import FFNN
from ids_expt.utils.confusion_matrix import get_confusion_matrix
from tqdm import tqdm


class ClfModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ClfModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


model_paths = [
    Path(
        r"C:\Users\Viper\Desktop\thesis_code\results\cic_cnn\ctgan_oversampling\best_model_full.pth"
    ),
    Path(
        r"C:\Users\Viper\Desktop\thesis_code\results\cic_fnn\ctgan_oversampling\best_model_full.pth"
    ),
]
project_dir = Path(r"C:\Users\Viper\Desktop\thesis_code")

for model_path in model_paths:
    batch_size = 256
    train_ratio = 0.8

    EPSILONS = [0.0001, 0.001, 0.01, 0.1]
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=project_dir / "data/cic_ctgan_merged_synthetic_data.csv",
            features=TOP_CIC_FEATURES,
            sampling_method=SamplingMethod.NONE,
            max_data=-1,
            train_ratio=train_ratio,
        )
    ).get_datasets()

    val_dataset = DataSet(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )
    full_model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )
    model = ClfModel(full_model).to("cuda")

    predictions = []
    targets = []

    for x, y in tqdm(val_loader):
        with torch.no_grad():
            probs = full_model(x.to("cuda"))[1]
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(y.argmax(dim=1).cpu().numpy())
    get_confusion_matrix(
        predictions,
        targets,
        label_keys=list(val_dataset.label_encoding.keys()),
        out_file=model_path.parent / "confusion_matrix.png",
    )
    # calculate accuracy for each labels
    accuracy_per_label = {}
    for label in val_dataset.label_encoding.keys():
        label_idx = np.array(val_dataset.label_encoding[label]).argmax()
        label_preds = np.array(predictions)[np.array(targets) == label_idx]
        accuracy_per_label[label] = np.mean(label_preds == label_idx)
    logger.info(f"Accuracy per label: {accuracy_per_label}")
    with open(model_path.parent / "accuracy_per_label.txt", "w") as f:
        for label, acc in accuracy_per_label.items():
            f.write(f"{label}: {acc:.4f}\n")
    logger.info(f"Running adversarial attacks on model: {model_path.name}")

    for epsilon in EPSILONS:
        logger.info(f"Evaluating adversarial examples with epsilon: {epsilon}")
        classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(1, len(TOP_CIC_FEATURES)),
            nb_classes=val_dataset.data.Label.nunique(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
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
            out_file=model_path.parent / f"confusion_matrix_adv_fgsm_{epsilon}.png",
            eps=epsilon,
        )
        # calculate accuracy for each labels
        accuracy_per_label = {}
        for label in val_dataset.label_encoding.keys():
            label_idx = np.array(val_dataset.label_encoding[label]).argmax()
            label_preds = np.array(adv_predictions)[np.array(adv_targets) == label_idx]
            accuracy_per_label[label] = np.mean(label_preds == label_idx)
        logger.info(f"Accuracy per label for epsilon {epsilon}: {accuracy_per_label}")
        with open(
            model_path.parent / f"accuracy_per_label_fgsm_{epsilon}.txt", "w"
        ) as f:
            for label, acc in accuracy_per_label.items():
                f.write(f"{label}: {acc:.4f}\n")

    logger.info("Adversarial evaluation completed successfully.")
