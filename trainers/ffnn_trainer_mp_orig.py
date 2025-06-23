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
from ids_expt.core.defs import TOP_CIC_FEATURES


# Define run_training OUTSIDE the main guard
def run_training(config, model, sampling_method):
    """Wrapper function for parallel execution"""
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=Path(r"E:\MSc Works\IDS\data\cicflow_combined.csv"),
            sampling_method=sampling_method,
            max_data=-100,
            train_ratio=0.75,
            features=TOP_CIC_FEATURES,
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
                expt_name="cic_fnn",
                run_name=name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.0001,
                early_stopping_patience=1000,
                optimizer=Optimizer.ADAM,
            ),
            FFNN(
                input_size=len(TOP_CIC_FEATURES),
                hidden_layers=[90] * layers,
                output_size=11,
                dropout_rate=0.0,
            ),
            method,
        )
        for name, layers, method in [
            ("no_sampling", 9, SamplingMethod.NONE),
            # ("undersampling", 9, SamplingMethod.UNDERSAMPLE),
            ("smote_oversampling", 9, SamplingMethod.OVERSAMPLE),
        ]
    ]

    # Create pool and run in parallel
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(run_training, configs)

    logger.info("All training jobs completed")
