if __name__ == "__main__":
    from ids_expt.models.cnn import CNN1D
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
    from ids_expt.core.defs import TOP_FEATURES
    import argparse

    # Argument parser for project dir, max_data, and model paths
    parser = argparse.ArgumentParser(
        description="Train CNN/FNN model on CIC dataset with CTGAN oversampling."
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=r"C:\Users\Viper\Desktop\thesis_code",
        help="Directory containing the project files.",
    )
    parser.add_argument(
        "--max_data",
        type=int,
        default=-1000,
        help="Maximum number of data points to use. Use -ve for all data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for adversarial attack generation.",
    )
    # model either cnn or fnn from options
    parser.add_argument(
        "--model",
        choices=["cnn", "fnn"],
        type=str,
        default="cnn",
        help="Model type to use for training. Options are 'cnn' or 'fnn'.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["original", "synthetic"],
        default="synthetic",
        help="Type of data to use for training. Options are 'original' or 'synthetic'.",
    )

    args = parser.parse_args()
    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        logger.error(f"Project directory {project_dir} does not exist.")
        exit(1)

    model = args.model
    max_data = args.max_data
    batch_size = args.batch_size

    epochs = 2000
    train_ratio = 0.8

    trainer_cfg = NNTrainerConfig(
        result_dir=project_dir / "results",
        expt_name="tabular_classification",
        run_name=f"{model}",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.0001,
        weight_decay=1e-5,
        early_stopping_patience=500,
        log_mlflow=False,
        optimizer="adam",
        
        # best_model_metric="f1_score",
        # best_model_metric_greater=True,
    )
    # If using synthetic data, use the CTGAN generated data.
    if args.data_type == "synthetic":
        csv_file = project_dir / "data/cic_ctgan_merged_synthetic_data.csv"
        use_synthetic = True
    else:
        csv_file = project_dir / "data/cicflow_combined.csv"        
        csv_file = project_dir / "data/combined_120_timeout.csv"
        use_synthetic = False

    trainer_cfg.run_name = f"{model}_{'synthetic' if use_synthetic else 'original'}"
    # just initialize the object.
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=csv_file,
            # features=TOP_CIC_FEATURES,
            features=TOP_FEATURES,
            # features=[],
            sampling_method=SamplingMethod.NONE,
            max_data=max_data,
            train_ratio=train_ratio,
            has_synthetic=use_synthetic,
        )
    ).get_datasets()
    input_size = len(train_dataset.config.features)
    logger.info(f"Input size: {input_size}, Model: {model}, Data type: {args.data_type}")
    if model == "cnn":
        model = CNN1D(
            # input_size=len(TOP_CIC_FEATURES),
            input_size=input_size,
            output_size=val_dataset.data.Label.nunique(),
            use_batchnorm=True,
            dropout_rate=0.0,
        )
    elif model == "fnn":
        model = FFNN(
            # input_size=len(TOP_CIC_FEATURES),
            input_size=input_size,
            # hidden_layers=[32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32],
            output_size=val_dataset.data.Label.nunique(),
            use_batchnorm=False,
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
