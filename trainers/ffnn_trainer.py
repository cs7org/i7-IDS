if __name__ == "__main__":
    from ids_expt.models.ffnn import FFNN
    from ids_expt.data.dataset import (
        DataSetConfig,
        SamplingMethod,
        CLFDataSet as DataSet,
        DFDataSet,
    )
    from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
    from pathlib import Path
    from loguru import logger

    epochs = 100
    max_data = -100
    batch_size = 512
    train_ratio = 0.8

    trainer_cfg = NNTrainerConfig(
        result_dir=Path("results"),
        expt_name="reproduce_ffnn",
        run_name="no_sampling",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.0001,
    )
    model = FFNN(input_size=46, hidden_layers=[90] * 5, output_size=9)
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            sampling_method=SamplingMethod.NONE,
            max_data=max_data,
            train_ratio=train_ratio,
        )
    ).get_datasets()

    trainer = NNTrainer(
        config=trainer_cfg,
        model=model,
        train_dataset=DataSet(train_dataset),
        val_dataset=DataSet(val_dataset),
    )
    trainer.train()
    trainer.plot_metrics()
    logger.info("Trained successfully.")

    trainer_cfg = NNTrainerConfig(
        result_dir=Path("results"),
        expt_name="reproduce_ffnn",
        run_name="undersampling",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.0001,
    )
    model = FFNN(input_size=46, hidden_layers=[90] * 9, output_size=9)
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            sampling_method=SamplingMethod.UNDERSAMPLE,
            max_data=max_data,
            train_ratio=train_ratio,
        )
    ).get_datasets()

    trainer = NNTrainer(
        config=trainer_cfg,
        model=model,
        train_dataset=DataSet(train_dataset),
        val_dataset=DataSet(val_dataset),
    )
    trainer.train()
    trainer.plot_metrics()
    logger.info("Trained successfully.")

    trainer_cfg = NNTrainerConfig(
        result_dir=Path("results"),
        expt_name="reproduce_ffnn",
        run_name="oversampling",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.0001,
    )
    model = FFNN(input_size=46, hidden_layers=[90] * 9, output_size=9)
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            sampling_method=SamplingMethod.OVERSAMPLE,
            max_data=max_data,
            train_ratio=train_ratio,
        )
    ).get_datasets()

    trainer = NNTrainer(
        config=trainer_cfg,
        model=model,
        train_dataset=DataSet(train_dataset),
        val_dataset=DataSet(val_dataset),
    )
    trainer.train()
    trainer.plot_metrics()
    logger.info("Trained successfully.")
