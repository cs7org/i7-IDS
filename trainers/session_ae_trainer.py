from ids_expt.models.cnn_ae import UNetAutoencoderNoSkip
from ids_expt.data.image_pair import (
    AdversarialDataPairConfig,
    TorchPairDataset,
    AdversarialDataPair,
)
from ids_expt.models.trainer_ae import AETrainer, NNTrainerConfig
import torch
from pathlib import Path

max_data = -100
num_samples_per_epoch = 100
if __name__ == "__main__":
    model = UNetAutoencoderNoSkip(
        encoder_name="resnet18",
        in_channels=1,
        out_channels=1,
        encoder_weights=None,
    )
    data_config = AdversarialDataPairConfig(
        max_data=max_data, num_samples_per_epoch=num_samples_per_epoch, min_num_pkts=1
    )
    train_ds, val_ds = AdversarialDataPair(config=data_config).load_data()
    trainer = AETrainer(
        config=NNTrainerConfig(
            result_dir=Path(r"C:\Users\Viper\Desktop\thesis_code\results"),
            expt_name="session_ae_experiment",
            run_name="unet_ae",
            epochs=1000,
            batch_size=16,
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
