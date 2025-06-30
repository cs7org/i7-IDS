from ids_expt.models.cnn import CNN2D as SimpleCNN, BiggerCNN2D
from ids_expt.data.session_image_dataset import (
    SessionImageDataConfig,
    DFDataSet,
    TorchImageDataset,
    SamplingMethod,
)
from pathlib import Path
from loguru import logger
import torch
from torchmetrics import F1Score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from ids_expt.utils.confusion_matrix import get_confusion_matrix


class ClfModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ClfModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


EPSILONS = [0.0001, 0.001, 0.01, 0.1]
model_path = Path(
    r"C:\Users\Viper\Desktop\thesis_code\results\image_classification\bigger_cnn2d_normalized\best_model.pth"
)
config = SessionImageDataConfig(
    max_data=-100,
    session_images_dir=Path(
        r"C:\Users\Viper\Desktop\thesis_code\notebooks\120_timeout_dnp3_sessions\session_images"
    ),
    labels_file=Path(
        r"C:\Users\Viper\Desktop\thesis_code\notebooks\120_timeout_dnp3_sessions\labelled_sessions.csv"
    ),
    sampling_method=SamplingMethod.OVERSAMPLE,
    use_normalized=True,
)
train_ds, test_ds = DFDataSet(config=config).load_data()

model = BiggerCNN2D(
    in_channel=1,
    num_classes=len(train_ds.label_encoding),
    dropout_rate=0.1,
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(
    torch.load(
        model_path,
        map_location=torch.device("cuda"),
    )
)


testds = TorchImageDataset(test_ds)
test_loader = torch.utils.data.DataLoader(
    testds,
    batch_size=32,
    shuffle=False,
)
f1_score = F1Score(task="multiclass", num_classes=len(test_ds.label_encoding))

predictions = []
targets = []

for images, labels in tqdm(test_loader):
    images = images.to(torch.float32)
    logits, proba = model(images.to("cuda"))
    preds = torch.argmax(proba, dim=1)
    predictions.extend(preds.cpu().numpy())
    targets.extend(labels.argmax(dim=1).cpu().numpy())

get_confusion_matrix(
    predictions,
    targets,
    label_keys=list(test_ds.label_encoding.keys()),
    out_file=model_path.parent / "confusion_matrix.png",
)

for epsilon in EPSILONS:
    logger.info(f"Evaluating adversarial examples with epsilon: {epsilon}")
    x_min, x_max = (0, 1)
    clip_values = (x_min, x_max)
    classifier = PyTorchClassifier(
        model=ClfModel(model),
        clip_values=clip_values,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(1, 6 * 32, 8 * 32),
        nb_classes=test_ds.data_df.label.nunique(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )
    attack = FastGradientMethod(
        estimator=classifier,
        eps=epsilon,
    )
    adv_predictions = []
    adv_targets = []
    for images, labels in tqdm(test_loader):
        images = images.to(torch.float32)
        adv_images = attack.generate(x=images.numpy())
        adv_logits, adv_proba = model(torch.tensor(adv_images).to("cuda"))
        adv_preds = torch.argmax(adv_proba, dim=1)
        adv_predictions.extend(adv_preds.cpu().numpy())
        adv_targets.extend(labels.argmax(dim=1).cpu().numpy())
    get_confusion_matrix(
        adv_predictions,
        adv_targets,
        label_keys=list(test_ds.label_encoding.keys()),
        out_file=model_path.parent / f"confusion_matrix_adv_{epsilon}.png",
        eps=epsilon,
    )
