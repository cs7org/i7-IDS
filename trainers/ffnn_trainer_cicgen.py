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
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from ids_expt.core.defs import TOP_CIC_FEATURES

    project_dir = Path(r"C:\Users\Viper\Desktop\thesis_code")

    epochs = 1000
    max_data = -100
    batch_size = 32
    train_ratio = 0.8

    trainer_cfg = NNTrainerConfig(
        result_dir=project_dir / "results",
        expt_name="cic_fnn",
        run_name="ctgan_oversampling",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.0001,
        early_stopping_patience=1000,
        log_mlflow=False,
    )

    # just initialize the object.
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=project_dir / "data/cicflow_combined.csv",
            features=TOP_CIC_FEATURES,
            sampling_method=SamplingMethod.NONE,
            max_data=max_data,
            train_ratio=train_ratio,
        )
    ).get_datasets()

    train_df = pd.read_csv(project_dir / "data/cic_merged_train_data.csv")
    train_df.columns = train_df.columns.str.strip()
    val_df = pd.read_csv(project_dir / "data/cic_merged_test_data.csv")
    val_df.columns = val_df.columns.str.strip()
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = val_df.drop(columns=["Label"])
    y_val = val_df["Label"]
    X_val = scaler.transform(X_val)
    train_dataset.data = pd.DataFrame(X_train, columns=train_df.columns[:-1])
    train_dataset.data["Label"] = y_train.values
    val_dataset.data = pd.DataFrame(X_val, columns=val_df.columns[:-1])
    val_dataset.data["Label"] = y_val.values
    model = FFNN(
        input_size=len(TOP_CIC_FEATURES),
        hidden_layers=[90] * 9,
        output_size=val_dataset.data.Label.nunique(),
        use_batchnorm=False,
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
