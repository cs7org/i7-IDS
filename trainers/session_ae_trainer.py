from ids_expt.models.cnn_ae import DDSA_CNN
from ids_expt.models.resnet_vae import VAE
from ids_expt.data.adversarial_data_pair import (
    AdversarialDataPairConfig,
    TorchPairDataset,
    AdversarialDataPair,
)
from ids_expt.models.unet import UnetAE
from ids_expt.models.rdunet import RDUNet
from ids_expt.models.trainer_ae import AETrainer, NNTrainerConfig
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
    "--ae_type",
    type=str,
    choices=["ddsa", "vae", "unet", "rdunet"],
    default="ddsa",
    help="Type of autoencoder to use. Options are 'ddsa' or 'cnn'.",
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
parser.add_argument(
    "--num_workers",
    type=int,
    default=os.cpu_count(),
    help="Number of workers for data loading. Default is the number of CPU cores.",
)
parser.add_argument(
    "--use_clf_model",
    action="store_true",
    help="Whether to use a pre-trained classifier model for training the autoencoder.",
)
parser.add_argument(
    "--expt_name",
    type=str,
    default="autoencoder2",
    help="Name of the experiment for logging purposes.",
)
# default false
parser.add_argument(
    "--drop_encoder_decoder_connnection",
    action="store_true",
    default=True,
    help="Whether to drop the encoder-decoder connection in the UnetAE model.",
)
args = parser.parse_args()
project_dir = Path(args.project_dir)
data_dir = Path(args.data_dir)
num_samples_per_epoch = args.num_samples_per_epoch
ae_type = args.ae_type.lower()


if __name__ == "__main__":
    clf_model = None
    suffix = ""
    if ae_type == "ddsa":
        model = DDSA_CNN()
        logger.info("Using DDSA_CNN model for autoencoder.")
    elif ae_type == "vae":
        model = VAE()
        logger.info("Using VAE model for autoencoder.")
    elif ae_type == "unet":
        model = UnetAE(
            encoder_name="resnet34",
            encoder_weights=None,
            activation="sigmoid",
            drop_first_n_encoder_connections=1,
            drop_encoder_decoder_connnection=args.drop_encoder_decoder_connnection,
        )
        logger.info("Using UnetAE model for autoencoder.")
        logger.info(
            f"Drop encoder-decoder connection: {args.drop_encoder_decoder_connnection}"
        )
        suffix = "_custom" if args.drop_encoder_decoder_connnection else ""
    elif ae_type == "rdunet":
        model = RDUNet(base_filters=64)
    else:
        logger.error(f"Invalid ae_type: {ae_type}. Choose 'ddsa' or 'vae'.")
        exit(1)

    norm_ = "_normalized_" if args.data_type == "normalized" else "_"
    clf_model = None
    clf_str = "_no_clf"
    if args.use_clf_model:
        clf_model_path = (
            project_dir
            / f"results/image_classification/resnet18{norm_}nosampling/best_model_full.pth"
        )
        clf_model = torch.load(
            clf_model_path,
            weights_only=False,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        clf_model.eval()
        logger.info(f"Loaded classifier model from {clf_model_path}")
        # set all parameters to not require gradients
        for param in clf_model.parameters():
            param.requires_grad = False
        clf_str = ""

    logger.info(
        f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )

    data_config = AdversarialDataPairConfig(
        data_dir=data_dir,
        num_samples_per_epoch=num_samples_per_epoch,
        clean_selection_rate=0.5,
    )
    train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()
    val_ds.config.num_samples_per_epoch = int(num_samples_per_epoch * 0.15)
    num_workers = max(args.num_workers // 2, 1)
    logger.info(f"Number of workers for data loading: {num_workers}")
    trainer = AETrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name=args.expt_name,
            run_name=f"{ae_type}{suffix}_{args.data_type}{clf_str}",
            epochs=1000,
            batch_size=args.batch_size,
            learning_rate=0.0001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=100,
            metrics=[],
            weighted_loss=False,
            log_mlflow=False,
            weight_decay=1e-5,
            optimizer="adamw",
            number_of_workers=num_workers,
        ),
        model=model,
        train_dataset=TorchPairDataset(train_ds),
        val_dataset=TorchPairDataset(val_ds),
        # criterion=torch.nn.L1Loss(reduction="mean"),
        # criterion=torch.nn.BCELoss(),
        ae_type=ae_type,
        clf_model=clf_model,
        clf_loss_weight=0.5,
    )
    trainer.train()
    trainer.plot_metrics()
