if __name__ == "__main__":
    from ids_expt.models.cnn import CNN1D
    from ids_expt.data.dataset import (
        DataSetConfig,
        SamplingMethod,
        CLFDataSet as DataSet,
        DFDataSet,
    )
    from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
    from pathlib import Path
    from loguru import logger
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from ids_expt.core.defs import TOP_CIC_FEATURES

    project_dir = Path(r"C:\Users\Viper\Desktop\thesis_code")

    epochs = 2000
    max_data = -100
    batch_size = 256
    train_ratio = 0.8

    trainer_cfg = NNTrainerConfig(
        result_dir=project_dir / "results",
        expt_name="cic_cnn",
        run_name="ctgan_oversampling",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.0001,
        early_stopping_patience=500,
        log_mlflow=False,
        # best_model_metric="f1_score",
        # best_model_metric_greater=True,
    )

    # just initialize the object.
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=project_dir / "data/cic_ctgan_merged_synthetic_data.csv",
            features=TOP_CIC_FEATURES,
            sampling_method=SamplingMethod.NONE,
            max_data=max_data,
            train_ratio=train_ratio,
        )
    ).get_datasets()

    model = CNN1D(
        input_size=len(TOP_CIC_FEATURES),
        output_size=val_dataset.data.Label.nunique(),
        use_batchnorm=True,
        dropout_rate=0.0,
    )

    trainer = NNTrainer(
        config=trainer_cfg,
        model=model,
        train_dataset=DataSet(train_dataset),
        val_dataset=DataSet(val_dataset),
    )
    trainer.train()
    trainer.plot_metrics()
    logger.info("Trained successfully.")
