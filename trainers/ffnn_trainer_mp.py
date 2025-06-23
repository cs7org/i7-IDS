import multiprocessing
from pathlib import Path
from loguru import logger
from ids_expt.models.ffnn import FFNN
from ids_expt.models.cnn import CNN1D
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
from ids_expt.models.trainer import NNTrainer, NNTrainerConfig, Optimizer


# Define run_training OUTSIDE the main guard
def run_training(config, model, sampling_method):
    """Wrapper function for parallel execution"""
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            sampling_method=sampling_method,
            max_data=-100,
            train_ratio=0.75,
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
    epochs = 1000
    batch_size = 32

    # Create configurations and models
    configs = [
        (
            NNTrainerConfig(
                result_dir=Path("results"),
                expt_name="reproduce_ffnn",
                run_name=name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.0001,
                early_stopping_patience=1000,
                optimizer=Optimizer.ADAM,
            ),
            FFNN(
                input_size=46,
                hidden_layers=[46 * i for i in [2, 3, 4, 5, 4, 3, 2]],
                output_size=9,
                dropout_rate=0.0,
            ),
            method,
        )
        for name, layers, method in [
            ("no_sampling", 9, SamplingMethod.NONE),
            ("undersampling", 9, SamplingMethod.UNDERSAMPLE),
            ("oversampling", 9, SamplingMethod.OVERSAMPLE),
        ]
    ]

    # Create pool and run in parallel
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(run_training, configs)

    logger.info("All training jobs completed")
