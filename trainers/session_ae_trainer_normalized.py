from ids_expt.models.cnn_ae import DDSA_CNN
from ids_expt.data.adversarial_data_pair import (
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
    data_dir = Path(
        r"C:\Users\Viper\Desktop\thesis_code\data\120_timeout_dnp3_sessions"
    )
else:
    data_dir = Path(data_dir)
batch_size = int(os.environ.get("BATCH_SIZE", 4))
num_samples_per_epoch = int(os.environ.get("NUM_SAMPLES_PER_EPOCH", 10000))
if __name__ == "__main__":
    model = DDSA_CNN()
    data_config = AdversarialDataPairConfig(
        data_dir=data_dir,
        num_samples_per_epoch=num_samples_per_epoch,
        adversarial_type_selection_rate=[
            ("basiciterativemethod_eps_0.1", 0.25),
            ("fastgradientmethod_eps_0.1", 0.25),
            ("fastgradientmethod_eps_0.01", 0.25),
            ("fastgradientmethod_eps_0.01", 0.25),
        ],
    )
    train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()
    # only 150% of num_samples_per_epoch for val
    val_ds.config.num_samples_per_epoch = int(num_samples_per_epoch * 0.15)
    trainer = AETrainer(
        config=NNTrainerConfig(
            result_dir=project_dir / "results",
            expt_name="session_ae_experiment_normalized",
            run_name="ddsa_cnn",
            epochs=1000,
            batch_size=1,
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
    trainer.train()
    trainer.plot_metrics()
