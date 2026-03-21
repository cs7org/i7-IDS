from ids_expt.models.image_model import ImageClfModel
from ids_expt.data.adversarial_data_pair import (
    AdversarialDataPairConfig,
    TorchPairDataset,
    AdversarialDataPair,
)
from ids_expt.models.trainer_adv import AdvTrainer, NNTrainerConfig
import torch
from pathlib import Path
from loguru import logger
import argparse
import cv2
import numpy as np


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
    default=128,
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
parser.add_argument(
    "--clf_mode",
    type=str,
    choices=["binary", "multiclass"],
    default="multiclass",
    help="Classification mode to use. 'binary' for binary classification, 'multiclass' for multiclass classification.",
)
parser.add_argument("--attack_only", action="store_true", default=False, help="Use only attack samples for training.")


args = parser.parse_args()
project_dir = Path(args.project_dir)
data_dir = Path(args.data_dir)
num_samples_per_epoch = args.num_samples_per_epoch
normalized_str = "_normalized_" if args.data_type == "normalized" else "_"
sampling_method = "nosampling"

combine_attacks = True if args.clf_mode == "binary" else False
attack_only = args.attack_only
if attack_only:
    suffix = "_attack_only"
else:
    suffix = ""

if __name__ == "__main__":
    data_config = AdversarialDataPairConfig(
        data_dir=data_dir,
        num_samples_per_epoch=num_samples_per_epoch,
        clean_selection_rate=0.7,
        combine_attacks=combine_attacks,
        attack_only=attack_only,
        apply_noise_rate=0.2

    )
    logger.info(f"Data Config: {data_config}")
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
    expt_dir = project_dir / "results" / "image_classification" / f"{run_name}_adv"
    inp, out, lbl = val_ds[0]
    img = np.hstack([inp, out])
    cv2.imwrite(str(expt_dir / f"{run_name}.png"), (img * 255).astype(np.uint8))
    run_name = f"{args.backbone}{normalized_str}{sampling_method}_{args.clf_mode}{suffix}"

    trainer = AdvTrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name="image_classification",
            run_name=run_name + "_adv",
            epochs=200,
            batch_size=args.batch_size,
            learning_rate=0.0001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=20,
            weighted_loss=False,
            log_mlflow=False,
            weight_decay=1e-5,
            optimizer="adamw",
            number_of_workers=8
        ),
        model=model,
        train_dataset=TorchPairDataset(train_ds),
        val_dataset=TorchPairDataset(val_ds),
    )
    trainer.train()
    trainer.plot_metrics()
