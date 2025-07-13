from ids_expt.models.image_model import ImageClfModel
from ids_expt.data.adversarial_data_pair import (
    AdversarialDataPairConfig,
    TorchPairDataset,
    AdversarialDataPair,
)
from ids_expt.models.unet import UnetAE
from ids_expt.models.trainer_adv import AdvTrainer, NNTrainerConfig
import torch
from pathlib import Path
import os
from loguru import logger
import argparse


# Argument parser for project dir, data dir, ae_type, batch_size, num_samples_per_epoch
parser = argparse.ArgumentParser(description="Session AE Trainer Configuration")
parser.add_argument(
    "--project_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code",
    help="Directory containing the project files.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions",
    help="Directory containing the dataset.",
)
parser.add_argument(
    "--backbone",
    type=str,
    default="resnet18",
    help="Backbone model to use for image classification.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for training.",
)
parser.add_argument(
    "--num_samples_per_epoch",
    type=int,
    default=10000,
    help="Number of samples per epoch for training.",
)
parser.add_argument(
    "--data_type",
    type=str,
    choices=["normal", "normalized"],
    default="normal",
    help="Type of data to use for training. 'normal' for raw images, 'normalized' for normalized images.",
)
args = parser.parse_args()
project_dir = Path(args.project_dir)
data_dir = Path(args.data_dir)
num_samples_per_epoch = args.num_samples_per_epoch
normalized_str = "_normalized_" if args.data_type == "normalized" else "_"
sampling_method = "nosampling"

if __name__ == "__main__":
    data_config = AdversarialDataPairConfig(
        data_dir=data_dir,
        num_samples_per_epoch=num_samples_per_epoch,
        clean_selection_rate=0.5,
    )
    train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()

    run_name = f"{args.backbone}{normalized_str}{sampling_method}"
    model = ImageClfModel(
        in_channel=1,
        num_classes=len(train_ds.label_encoding),
        backbone=args.backbone,
    )

    norm_ = "_normalized_" if args.data_type == "normalized" else "_"

    logger.info(
        f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )

    val_ds.config.num_samples_per_epoch = int(num_samples_per_epoch * 0.15)
    trainer = AdvTrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name="image_classification",
            run_name=run_name + "_adv",
            epochs=1000,
            batch_size=args.batch_size,
            learning_rate=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=100,
            weighted_loss=False,
            log_mlflow=False,
            weight_decay=1e-5,
            optimizer="adamw",
        ),
        model=model,
        train_dataset=TorchPairDataset(train_ds),
        val_dataset=TorchPairDataset(val_ds),
    )
    trainer.train()
    trainer.plot_metrics()
