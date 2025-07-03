from pathlib import Path
from loguru import logger
from ids_expt.data.session_image_dataset import (
    SessionImageDataConfig,
    DFDataSet,
    TorchImageDataset,
    SamplingMethod,
)
from ids_expt.models.cnn import BiggerCNN2D
import torch
from ids_expt.adversarial.adversarial_experiment import AdversarialExperiment, ClfModel
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from ids_expt.utils.confusion_matrix import get_confusion_matrix

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


logger.info(f"Running adversarial attacks on model: {model_path.name}")
epsilons = [0.0001, 0.001, 0.01, 0.1]
iterations = 10
input_shape = (1, 6 * 32, 256)

attacks = [
    FastGradientMethod(
        estimator=PyTorchClassifier(
            model=ClfModel(model),
            loss=torch.nn.CrossEntropyLoss(),
            clip_values=(0, 1),
            input_shape=input_shape,
            nb_classes=len(train_ds.label_encoding),
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
                nb_classes=len(train_ds.label_encoding),
                optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            ),
            eps=eps,
            eps_step=eps / 10,
            max_iter=iterations,
        )
        for eps in epsilons
    ]
)
adv = AdversarialExperiment(
    model=model,
    model_name=model_path.parent.name,
    attacks=attacks,
    train_dataset=TorchImageDataset(train_ds),
    test_dataset=TorchImageDataset(test_ds),
)
for attack in attacks:
    logger.info(f"Running attack: {attack.__class__.__name__} with eps: {attack.eps}")
    adv.run()

logger.info("Adversarial attacks completed successfully.")
logger.info("Generating adversarial attack data...")

selected_attacks = [atk for atk in attacks if atk.eps in [0.1, 0.01]]

for attack in selected_attacks:
    logger.info(
        f"Generating adversarial data for attack: {attack.__class__.__name__} with eps: {attack.eps}"
    )
    out_folder = attack.__class__.__name__.lower() + f"_eps_{attack.eps}"
    adv.generate(attack, out_folder=out_folder)
