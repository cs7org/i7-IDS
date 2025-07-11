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

project_dir = os.environ.get("PROJECT_DIR")
if project_dir is None:
    logger.warning(
        "PROJECT_DIR environment variable not set, using default project directory."
    )
    project_dir = Path(r"C:\Users\Viper\Desktop\thesis_code")
else:
    project_dir = Path(project_dir)
data_dir = os.environ.get("DATA_DIR")
if data_dir is None:
    logger.warning(
        "DATA_DIR environment variable not set, using default data directory."
    )
    data_dir = Path(r"C:\Users\Viper\Desktop\thesis_code\results\adversarial_attacks")
else:
    data_dir = Path(data_dir)
batch_size = int(os.environ.get("BATCH_SIZE", 4))
num_samples_per_epoch = int(os.environ.get("NUM_SAMPLES_PER_EPOCH", 10000))
if __name__ == "__main__":
    model = DDSA_FFNN()
    data_config = AdversarialDataPairConfig(
        data_dir=data_dir, num_samples_per_epoch=num_samples_per_epoch
    )
    train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()
    val_ds.config.num_samples_per_epoch = int(num_samples_per_epoch * 0.15)
    trainer = AETrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name="session_ae_experiment",
            run_name="ddsa_ffnn",
            epochs=1000,
            batch_size=32,
            learning_rate=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_patience=50,
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
