from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
from ids_expt.data.session_image_dataset import (
    SessionImageDataConfig,
    DFDataSet,
    TorchImageDataset,
    SamplingMethod,
)
from ids_expt.models.cnn import CNN2D as BiggerCNN2D
from ids_expt.models.image_model import ImageClfModel
import torch
from pathlib import Path
from loguru import logger
import os
import argparse
import cv2
import numpy as np

# argument parser to accept: image_type[normalized,normal], backbone, max_data
parser = argparse.ArgumentParser(description="Session Image Trainer Configuration")
parser.add_argument(
    "--image_type",
    type=str,
    choices=["normal", "normalized"],
    help="Type of images to use for training. 'normal' for raw images, 'normalized' for normalized images.",
)
parser.add_argument(
    "--backbone",
    type=str,
    default="resnet18",
    help="Backbone model to use for image classification.",
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
    help="Batch Size.",
)
parser.add_argument(
    "--sampling_method",
    type=str,
    choices=["nosampling", "oversampling", "undersampling"],
    default="nosampling",
    help="Sampling method to use for the dataset.",
)
parser.add_argument(
    "--num_samples_per_epoch",
    type=int,
    default=10000,
    help="Number of samples to use per epoch. Default is 10000.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions",
)
parser.add_argument(
    "--project_dir",
    type=str,
    default=r"C:\Users\Viper\Desktop\thesis_code",
    help="Project directory where results will be saved.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1000,
    help="Number of epochs to train the model.",
)

args = parser.parse_args()
batch_size = args.batch_size
use_normalized = args.image_type.lower() == "normalized"
normalized_str = "_normalized_" if use_normalized else "_"
sampling_method = args.sampling_method
data_dir = Path(args.data_dir)
project_dir = Path(args.project_dir)

logger.info(f"Args: {args}")
expt_name = "image_classification2"

if __name__ == "__main__":
    # Configuration parameters
    config = SessionImageDataConfig(
        max_data=args.max_data,
        session_images_dir=data_dir / "session_images",
        labels_file=data_dir / "labelled_sessions.csv",
        sampling_method=sampling_method.lower(),
        use_normalized=use_normalized,
    )

    # Load the dataset
    train_ds, test_ds = DFDataSet(config=config).load_data()
    img, lbl, lbl_str = test_ds[0]

    expt_dir = project_dir / "results" / expt_name
    if not expt_dir.exists():
        expt_dir.mkdir(parents=True)

    run_name = f"{args.backbone}{normalized_str}{sampling_method}"
    model = ImageClfModel(
        in_channel=1,
        num_classes=len(train_ds.label_encoding),
        backbone=args.backbone,
    )
    cv2.imwrite(str(expt_dir / f"{run_name}.png"), (img * 255).astype(np.uint8))
    # Initialize the trainer
    trainer = NNTrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name=expt_name,
            run_name=run_name,
            epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=0.0001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=100,
            log_mlflow=False,
            weight_decay=1e-5,
            optimizer="adamw",
        ),
        model=model,
        train_dataset=TorchImageDataset(train_ds),
        val_dataset=TorchImageDataset(test_ds),
    )
    # Train the model
    trainer.train()
    # Plot the training metrics
    trainer.plot_metrics()
    logger.stop()
