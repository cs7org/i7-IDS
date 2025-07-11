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
import argparse

# parser for model_names by comma separated, batch_size
# data_dir, project_dir, image_type
parser = argparse.ArgumentParser(
    description="Adversarial Image Generation Configuration"
)
parser.add_argument(
    "--model_names",
    type=str,
    default="resnet18_nosampling,mobilenet_v3_large_nosampling",
    help="Comma-separated list of model names to use for adversarial attacks.",
)
parser.add_argument(
    "--image_type",
    type=str,
    choices=["normal", "normalized"],
    help="Type of images to use for training. 'normal' for raw images, 'normalized' for normalized images.",
)
parser.add_argument(
    "--max_data",
    type=int,
    default=-100,
    help="Maximum number of data points to use. Use -ve for all data.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for adversarial attack generation.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions",
    help="Directory containing the dataset.",
)
parser.add_argument(
    "--project_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code",
    help="Directory containing the project files.",
)
parser.add_argument(
    "--epsilons",
    type=str,
    default="0.0001,0.001,0.01,0.1",
    help="Comma-separated list of epsilon values for adversarial attacks.",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=10,
    help="Number of iterations for iterative adversarial attacks.",
)

args = parser.parse_args()

# Parse arguments
model_names = [m.strip() for m in args.model_names.split(",")]
project_dir = Path(args.project_dir)
data_dir = Path(args.data_dir)
if not data_dir.exists():
    logger.error(f"Data directory does not exist: {data_dir}")
    raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
if not project_dir.exists():
    logger.error(f"Project directory does not exist: {project_dir}")
    raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
epsilons = [float(eps.strip()) for eps in args.epsilons.split(",") if eps.strip()]
batch_size = args.batch_size


# !!!IMPORTANT: full model might not be usable when package is not installed
model_paths = [
    project_dir / "results" / "image_classification" / name / "best_model_full.pth"
    for name in model_names
]
iterations = args.iterations
max_data = args.max_data
use_normalized = args.image_type.lower() == "normalized"

for model_path in model_paths:
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    logger.info(f"Model path exists: {model_path}")
    config = SessionImageDataConfig(
        max_data=max_data,
        session_images_dir=data_dir / "session_images",
        labels_file=data_dir / "labelled_sessions.csv",
        use_normalized=use_normalized,
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
                verbose=False,
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
    adv.run(
        results_dir=(
            project_dir / "results" / "adversarial_attacks" / model_path.parent.name
        )
    )

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

    logger.info("Adversarial image generation completed successfully.")
