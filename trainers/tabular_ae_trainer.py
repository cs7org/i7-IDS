from ids_expt.models.ae import DDSA_FFNN
from ids_expt.data.adversarial_tabular_data_pair import (
    AdversarialDataPairConfig,
    TorchPairDataset,
    AdversarialDataPair,
)
from ids_expt.models.trainer_ae import AETrainer, NNTrainerConfig
import torch
from pathlib import Path
import os
from loguru import logger
import argparse

parser = argparse.ArgumentParser(description="Train DDSA FFNN model on adversarial tabular data.")
parser.add_argument(
    "--project_dir",
    type=str,
    default=os.environ.get("PROJECT_DIR", "/home/hpc/iwi7/iwi7101h/i7-IDS/"),
    help="Path to the project directory.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=os.environ.get("DATA_DIR", "/home/hpc/iwi7/iwi7101h/i7-IDS/results/adversarial_attacks/original_cnn"),
    help="Path to the data directory.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=int(os.environ.get("BATCH_SIZE", 128)),
    help="Batch size for training.",
)
parser.add_argument(
    "--num_samples_per_epoch",
    type=int,
    default=int(os.environ.get("NUM_SAMPLES_PER_EPOCH", 12800)),
    help="Number of samples per epoch.",
)
args = parser.parse_args()

project_dir = args.project_dir
data_dir = args.data_dir
batch_size = args.batch_size
num_samples_per_epoch = args.num_samples_per_epoch

project_dir = Path(project_dir)
if not project_dir.exists():
    logger.error(f"Project directory {project_dir} does not exist. Please check the path.")
    exit(1)
data_dir = Path(data_dir)
if not data_dir.exists():
    logger.error(f"Data directory {data_dir} does not exist. Please check the path.")
    exit(1)

if __name__ == "__main__":
    
    data_config = AdversarialDataPairConfig(
        data_dir=data_dir, num_samples_per_epoch=num_samples_per_epoch
    )
    train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()
    val_ds.config.num_samples_per_epoch = int(num_samples_per_epoch * 0.15)
    model = DDSA_FFNN(input_size=train_ds[0][0].shape[0])
    trainer = AETrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name="session_ae_experiment",
            run_name="ddsa_ffnn",
            epochs=1000,
            batch_size=128,
            learning_rate=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=500,
            metrics=[],
            weighted_loss=False,
            log_mlflow=False,
        ),
        model=model,
        train_dataset=TorchPairDataset(train_ds),
        val_dataset=TorchPairDataset(val_ds),
        criterion=torch.nn.MSELoss(reduction="mean"),

    )
    # no need to plot at end of epoch, as we will plot after training
    trainer.at_epoch_end = lambda: None
    trainer.train()
    trainer.plot_metrics()
