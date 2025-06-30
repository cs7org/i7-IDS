from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
from ids_expt.data.session_image_dataset import (
    SessionImageDataConfig,
    DFDataSet,
    TorchImageDataset,
    SamplingMethod,
)
from ids_expt.models.cnn import CNN2D as SimpleCNN, BiggerCNN2D
import torch
from pathlib import Path
from loguru import logger


if __name__ == "__main__":
    batch_size = 64
    # Configuration parameters
    config = SessionImageDataConfig(
        max_data=-100,
        session_images_dir=Path(
            r"C:\Users\Viper\Desktop\thesis_code\notebooks\120_timeout_dnp3_sessions\session_images"
        ),
        labels_file=Path(
            r"C:\Users\Viper\Desktop\thesis_code\notebooks\120_timeout_dnp3_sessions\labelled_sessions.csv"
        ),
        sampling_method=SamplingMethod.OVERSAMPLE,
    )

    # Load the dataset
    train_ds, test_ds = DFDataSet(config=config).load_data()

    models = [
        (
            "bigger_cnn2d",
            BiggerCNN2D(
                in_channel=1,
                num_classes=len(train_ds.label_encoding),
                dropout_rate=0.1,
            ),
        ),
        # (
        #     "simple_cnn",
        #     SimpleCNN(
        #         in_channel=1,
        #         num_classes=len(train_ds.label_encoding),
        #     ),
        # ),
    ]
    for run_name, model in models:
        # Initialize the trainer
        trainer = NNTrainer(
            config=NNTrainerConfig(
                result_dir=Path(r"C:\Users\Viper\Desktop\thesis_code\results"),
                expt_name="image_classification",
                run_name=run_name,
                epochs=10000,
                batch_size=batch_size,
                learning_rate=0.0001,
                device="cuda" if torch.cuda.is_available() else "cpu",
                early_stopping_patience=1000,
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
