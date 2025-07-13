from pathlib import Path
from loguru import logger

import torch
from ids_expt.adversarial.adversarial_experiment import AdversarialExperiment, ClfModel
from ids_expt.core.defs import TOP_CIC_FEATURES
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
import argparse

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
    default="synthetic",
    help="Type of data to use for training. Options are 'original' or 'synthetic'.",
)

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
    / f"results/tabular_classification/{model_name}/last_model_full.pth"
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

for model_path in model_paths:
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist.")
        exit(1)

    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=csv_file,
            features=TOP_CIC_FEATURES,
            max_data=max_data,
            train_ratio=0.8,
            # labels=[
            #     "ARP_POISONING",
            #     "COLD_RESTART",
            #     "DISABLE_UNSOLICITED",
            #     "DNP3_ENUMERATE",
            #     "DNP3_INFO",
            #     "INIT_DATA",
            #     "MITM_DOS",
            #     "NORMAL",
            #     "REPLAY",
            #     "STOP_APP",
            #     "WARM_RESTART",
            # ],
        )
    ).get_datasets()

    val_dataset.data = val_dataset.data.query("is_synthetic != True")

    # write val data
    # val_dataset.data.to_csv(
    #     project_dir / "data/val_data2.csv", index=False, header=True
    # )

    val_dataset = DataSet(val_dataset)
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
    epsilons = [0.0001, 0.001, 0.01, 0.1]
    iterations = 10
    input_shape = len(TOP_CIC_FEATURES)

    attacks = [
        FastGradientMethod(
            estimator=PyTorchClassifier(
                model=ClfModel(model),
                loss=torch.nn.CrossEntropyLoss(),
                clip_values=(0, 1),
                input_shape=input_shape,
                nb_classes=len(train_dataset.label_encoding),
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
                    nb_classes=len(train_dataset.label_encoding),
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
        train_dataset=DataSet(train_dataset),
        test_dataset=DataSet(val_dataset),
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

    # selected_attacks = [atk for atk in attacks if atk.eps in [0.1, 0.01]]

    # for attack in selected_attacks:
    #     logger.info(
    #         f"Generating adversarial data for attack: {attack.__class__.__name__} with eps: {attack.eps}"
    #     )
    #     out_folder = attack.__class__.__name__.lower() + f"_eps_{attack.eps}"
    #     adv.generate(attack, out_folder=out_folder, is_image_dataset=False)
    #     logger.info(f"Adversarial data generated and saved in {out_folder} folder.")
    # logger.info("All adversarial data generation completed.")
