from pathlib import Path
from loguru import logger
from ids_expt.data.session_image_dataset import (
    SessionImageDataConfig,
    DFDataSet,
    TorchImageDataset,
)
import torch
from ids_expt.adversarial.adversarial_experiment import AdversarialExperiment, ClfModel
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
import os

project_dir = Path(os.environ.get("PROJECT_DIR", r"C:\Users\Viper\Desktop\thesis_code"))
data_dir = Path(
    os.environ.get(
        "DATA_DIR", r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions"
    )
)
model_names = [
    "resnet18_normalized_nosampling",
    "mobilenetv3_large_normalized_nosampling",
]

# !!!IMPORTANT: full model might not be usable when package is not installed
model_paths = [
    project_dir / "results" / "image_classification" / name / "best_model_full.pth"
    for name in model_names
]
batch_size = int(os.environ.get("BATCH_SIZE", 128))


epsilons = [0.0001, 0.001, 0.01, 0.1]
for model_path in model_paths:
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    logger.info(f"Model path exists: {model_path}")
    config = SessionImageDataConfig(
        max_data=-10,
        session_images_dir=data_dir / "session_images",
        labels_file=data_dir / "labelled_sessions.csv",
        use_normalized=True,
    )
    train_ds, test_ds = DFDataSet(config=config).load_data()
    model_path = model_path.resolve()
    # this might fail if package is not installed
    logger.info(f"Loading model from: {model_path}")
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )

    logger.info(f"Running adversarial attacks on model: {model_path.name}")
    epsilons = [0.0001, 0.001, 0.01, 0.1]
    iterations = 10
    input_shape = (1, config.num_pkts, config.byte_length)

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
        input_shape=input_shape,
        output_dir=data_dir / "adversarial_attacks" / model_path.parent.name,
        batch_size=batch_size,
    )
    adv.run(results_dir=model_path.parent)

    logger.info("Adversarial attacks completed successfully.")
    logger.info("Generating adversarial attack data...")

    selected_attacks = [atk for atk in attacks if atk.eps in [0.1, 0.01]]

    for attack in selected_attacks:
        logger.info(
            f"Generating adversarial data for attack: {attack.__class__.__name__} with eps: {attack.eps}"
        )
        out_folder = attack.__class__.__name__.lower() + f"_eps_{attack.eps}"
        copy_compressed_to = (
            project_dir
            / "results"
            / "adversarial_attacks"
            / model_path.parent.name
            / out_folder
        )
        adv.generate(
            attack, out_folder=out_folder, copy_compressed_to=copy_compressed_to
        )
