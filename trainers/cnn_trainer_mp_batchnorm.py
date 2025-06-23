import multiprocessing
from pathlib import Path
from loguru import logger
from ids_expt.models.cnn import CNN1D
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
from datetime import datetime
import torch


# Define run_training OUTSIDE the main guard
def run_training(config, model, sampling_method):
    """Wrapper function for parallel execution"""
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            sampling_method=sampling_method,
            max_data=-100,
            train_ratio=0.8,
        )
    ).get_datasets()

    trainer = NNTrainer(
        config=config,
        model=model,
        train_dataset=DataSet(train_dataset),
        val_dataset=DataSet(val_dataset),
    )
    trainer.train()
    trainer.plot_metrics()
    logger.info(f"Trained successfully: {config.run_name}")
    return None


if __name__ == "__main__":
    # Configuration parameters
    epochs = 10000
    batch_size = 16

    configs = []
    for name, layers, method in [
        ("no_sampling", 9, SamplingMethod.NONE),
        # ("undersampling", 9, SamplingMethod.UNDERSAMPLE),
        ("oversampling", 9, SamplingMethod.OVERSAMPLE),
    ]:
        model = CNN1D(input_size=46)
        # model.load_state_dict(
        #     torch.load(
        #         rf"E:\MSc Works\IDS\results\cnn1D\{name}_bnorm\best_model.pth",
        #     )
        # )
        configs.append(
            (
                NNTrainerConfig(
                    result_dir=Path("results"),
                    expt_name="cnn1D",
                    run_name=name
                    + "_bnorm_"
                    + datetime.now().date().strftime("%Y%m%d"),
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=0.001,
                    early_stopping_patience=1000,
                ),
                model,
                method,
            )
        )

    # Create pool and run in parallel
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.starmap(run_training, configs)

    logger.info("All training jobs completed")
