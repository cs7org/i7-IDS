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
project_dir = os.environ.get("PROJECT_DIR")
if project_dir is None:
    logger.warning(
        "PROJECT_DIR environment variable not set, using default project directory."
    )
    project_dir = Path(r"C:\Users\Viper\Desktop\thesis_code")
else:
    project_dir = Path(project_dir)

if __name__ == "__main__":
    batch_size = 256
    # Configuration parameters
    config = SessionImageDataConfig(
        max_data=-100,
        session_images_dir=data_dir / "session_images",
        labels_file=data_dir / "labelled_sessions.csv",
        sampling_method=SamplingMethod.NONE,
    )

    # Load the dataset
    train_ds, test_ds = DFDataSet(config=config).load_data()

    models = [
        (
            # "bigger_cnn2d_nosampling",
            # BiggerCNN2D(
            #     in_channel=1,
            #     num_classes=len(train_ds.label_encoding),
            # ),
            "resnet18",
            ImageClfModel(
                in_channel=1,
                num_classes=len(train_ds.label_encoding),
            ),
        ),
    ]
    for run_name, model in models:
        # Initialize the trainer
        trainer = NNTrainer(
            config=NNTrainerConfig(
                result_dir=project_dir / "results",
                expt_name="image_classification",
                run_name=run_name,
                epochs=1000,
                batch_size=batch_size,
                learning_rate=0.0001,
                device="cuda" if torch.cuda.is_available() else "cpu",
                early_stopping_patience=100,
                log_mlflow=False,
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
