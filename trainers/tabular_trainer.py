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
    from ids_expt.core.defs import TOP_CIC_FEATURES
    import argparse
    import pandas as pd

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
        default=256,
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

    val_dataset.data = val_dataset.data.query("is_synthetic != True")

    if model == "cnn":
        model = CNN1D(
            input_size=len(TOP_CIC_FEATURES),
            output_size=val_dataset.data.Label.nunique(),
            use_batchnorm=True,
            dropout_rate=0.0,
        )
    elif model == "fnn":

        model = FFNN(
            input_size=len(TOP_CIC_FEATURES),
            hidden_layers=[90] * 10,
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
