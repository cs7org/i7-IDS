from ids_expt.data.dataset import DFDataSet, CLFDataSet as DataSet
from ids_expt.core.configs import NNTrainerConfig, Optimizer

from loguru import logger
from tqdm import tqdm
import torch
from torch import nn
from pathlib import Path
import joblib
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


class NNTrainer:
    def __init__(
        self,
        config: NNTrainerConfig,
        model: nn.Module,
        train_dataset: DataSet,
        val_dataset: DataSet,
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ):

        self.model = model
        self.config = config
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_file != "":
            if (self.config.run_dir / self.config.log_file).exists():
                (self.config.run_dir / self.config.log_file).unlink()
            logger.add(
                self.config.run_dir / self.config.log_file,
                rotation="1 day",
                retention="7 days",
                level="INFO",
                format="{time} | {level} | {message}",
            )
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        logger.info(f"Model attribs: {self.model.__dict__}")

        if self.config.optimizer == Optimizer.ADAM:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate
            )
        elif self.config.optimizer == Optimizer.SGD:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=config.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        self.criterion = criterion
        self._metrics = config.metrics
        self.metric_history = defaultdict(list)
        self.patience_counter = 0
        self.started_mlflow = False

        # Create directories for results

        # write scaler if it exists
        if hasattr(train_dataset.dataset, "scaler") and train_dataset.dataset.scaler:
            logger.info("Saving scaler to disk.")
            joblib.dump(
                train_dataset.dataset.scaler,
                self.config.run_dir / "scaler.pkl",
            )

        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Configuration: {self.config.model_dump_json(indent=2)}")
        logger.info(f"Data Config: {train_dataset.config.model_dump_json(indent=2)}")
        logger.info(
            f"Training dataset: {len(train_dataset)} samples, "
            f"Validation dataset: {len(val_dataset)} samples."
        )
        # save config
        with open(self.config.run_dir / "train_config.json", "w") as f:
            f.write(self.config.model_dump_json(indent=2))
        # save dataset config
        with open(self.config.run_dir / "dataset_config.json", "w") as f:
            f.write(train_dataset.config.model_dump_json(indent=2))
        self.logger = logger
        self.logger.info("Trainer initialized successfully.")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.number_of_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.number_of_workers,
        )
        self.metrics = self.get_metrics()
        if self.config.weighted_loss:
            self.update_criterion(self.criterion)

    def update_criterion(self, criterion: nn.Module):
        """Update the loss criterion."""
        if isinstance(criterion, nn.CrossEntropyLoss):
            self.criterion = nn.CrossEntropyLoss(
                weight=self.train_dataset.class_weights.to(self.device)
            )
            logger.info(f"Label Counts: {self.train_dataset.label_counts}")
            logger.info(
                f"Criterion updated to CrossEntropyLoss with class weights: {list(self.train_dataset.label_encoding.keys())}:{self.train_dataset.class_weights}."
            )

        else:
            logger.warning(
                "Criterion is not CrossEntropyLoss. Using the provided criterion."
            )

    def get_metrics(self):
        """Get the metrics based on the configuration."""
        num_classes = self.train_dataset.num_classes

        if num_classes == 2:
            # Binary classification
            metrics = MetricCollection(
                {
                    "accuracy": Accuracy(task="binary"),
                    "precision": Precision(task="binary", average="macro"),
                    "recall": Recall(task="binary", average="macro"),
                    "f1_score": F1Score(task="binary", average="macro"),
                    "roc_auc": AUROC(task="binary"),  # Binary AUROC
                }
            )
        else:
            # Multiclass classification
            metrics = MetricCollection(
                {
                    "accuracy": Accuracy(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ),
                    "precision": Precision(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ),
                    "recall": Recall(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ),
                    "f1_score": F1Score(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ),
                }
            )

        return metrics.to(self.device)

    def forward_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        logits, probs = self.model(inputs)
        loss = self.criterion(logits, labels.argmax(dim=1))

        # Update metrics
        self.metrics.update(probs.argmax(dim=1), labels.argmax(dim=1))
        metrics = self.metrics.compute()

        return probs, loss, metrics

    def run_epoch(self, dataloader, is_train=True):
        epoch_loss = 0.0
        epoch_metrics = {metric: 0.0 for metric in self.metrics}
        epoch_metrics["loss"] = 0.0
        pbar = tqdm(
            dataloader,
            desc="Training" if is_train else "Validation",
            unit="batch",
        )
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for i, batch in enumerate(pbar):
            if is_train:
                self.optimizer.zero_grad()
            outputs, loss, metrics = self.forward_step(batch)
            if is_train:
                loss.backward()
                self.optimizer.step()
            # Update metrics
            for metric in self.metrics:
                epoch_metrics[metric] += metrics[metric].item()

            # add loss as first metric
            epoch_metrics["loss"] += loss.item()
            if is_train:
                pbar.set_description(f"Training - Epoch {i + 1}/{len(dataloader)}")
            else:
                pbar.set_description(f"Validation - Epoch {i + 1}/{len(dataloader)}")

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    **{
                        metric: epoch_metrics[metric] / (i + 1)
                        for metric in self.metrics
                    },
                }
            )
        epoch_loss /= len(dataloader)
        for metric in self.metrics:
            epoch_metrics[metric] /= len(dataloader)
        epoch_metrics["loss"] = epoch_loss
        return outputs, epoch_loss, epoch_metrics

    def train(self):
        if self.config.log_mlflow:
            # if mlflow object is not None, log metrics to mlflow
            if not self.started_mlflow:
                mlflow.set_tracking_uri("http://localhost:5000")
                mlflow.set_experiment(self.config.expt_name)
                mlflow.start_run(
                    run_name=self.config.run_name,
                    nested=False,
                )
                logger.info("MLflow run started.")
                self.started_mlflow = True
        self.model.train()
        best_metric_value = (
            float("-inf") if self.config.best_model_metric_greater else float("inf")
        )
        best_model_path = self.config.run_dir / self.config.best_model_name

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            # Training step
            outputs, epoch_loss, epoch_metrics = self.run_epoch(
                self.train_loader, is_train=True
            )

            # Validation step
            with torch.no_grad():
                outputs, val_loss, val_metrics = self.run_epoch(
                    self.val_loader, is_train=False
                )
            logger.info(
                f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}, "
                + ", ".join(
                    f"{metric}: {epoch_metrics[metric]:.4f}" for metric in self.metrics
                )
            )
            logger.info(
                f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, "
                + ", ".join(
                    f"{metric}: {val_metrics[metric]:.4f}" for metric in self.metrics
                )
            )
            self.system_usage()

            # Save the best model
            if (
                self.config.best_model_metric_greater
                and val_metrics[self.config.best_model_metric] > best_metric_value
            ) or (
                not self.config.best_model_metric_greater
                and val_metrics[self.config.best_model_metric] < best_metric_value
            ):
                best_metric_value = val_metrics[self.config.best_model_metric]
                torch.save(self.model.state_dict(), best_model_path)
                torch.save(
                    self.model,
                    self.config.run_dir
                    / self.config.best_model_name.replace(".pth", "_full.pth"),
                )
                logger.info(f"Best model saved at {best_model_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement in {self.config.best_model_metric}. "
                    f"Patience counter: {self.patience_counter}/{self.config.early_stopping_patience}"
                )
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered. No improvement for "
                        f"{self.config.early_stopping_patience} epochs."
                    )
                    break
            # Save the optimizer state
            torch.save(
                self.optimizer.state_dict(),
                self.config.run_dir / "optimizer_state.pth",
            )
            # update metric history
            for metric in epoch_metrics:
                self.metric_history[metric].append(epoch_metrics[metric])
            for metric in val_metrics:
                self.metric_history[f"val_{metric}"].append(val_metrics[metric])
            # Log metrics
            self.log_metrics(
                epoch,
                epoch_loss,
                epoch_metrics,
                val_loss,
                val_metrics,
            )
        logger.info("Training completed.")
        if self.config.log_mlflow:
            mlflow.end_run()
            logger.info("MLflow run ended.")
        return best_model_path

    def log_metrics(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        """
        Log the training and validation metrics.
        """
        # Save metrics to file: epoch, train_loss, train_metrics, val_loss, val_metrics
        metrics_file = self.config.run_dir / self.config.metric_file
        # if it is a first epoch and the file exists, write header
        if epoch == 0:
            with open(metrics_file, "w") as f:
                header = (
                    "epoch,train_loss,"
                    + ",".join(self.metrics)
                    + ",val_loss,"
                    + ",".join(f"val_{metric}" for metric in self.metrics)
                )
                f.write(header + "\n")
        with open(metrics_file, "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},"
                + ",".join(f"{train_metrics[metric]:.4f}" for metric in self.metrics)
                + f",{val_loss},"
                + ",".join(f"{val_metrics[metric]:.4f}" for metric in self.metrics)
                + "\n"
            )
        logger.info(f"Metrics logged to {metrics_file}")
        if self.config.log_mlflow:
            # if mlflow object is not None, log metrics to mlflow
            if not self.started_mlflow:
                mlflow.set_tracking_uri("http://localhost:5000")
                mlflow.set_experiment(self.config.expt_name)
                mlflow.start_run(
                    run_name=self.config.run_name,
                    nested=False,
                )
                logger.info("MLflow run started.")
            self.started_mlflow = True
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for metric in self.metrics:
                mlflow.log_metric(f"train_{metric}", train_metrics[metric], step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for metric in self.metrics:
                mlflow.log_metric(f"val_{metric}", val_metrics[metric], step=epoch)
            logger.info("Metrics logged to MLflow.")

    def system_usage(self):
        """
        Log the system usage statistics.
        """
        import psutil

        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        gpu = None
        if torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / (1024**2)
        logger.info(
            f"System Usage - CPU: {cpu}%, Memory: {memory.percent}%, " f"GPU: {gpu} MB"
            if gpu is not None
            else "GPU: Not available"
        )
        return {
            "cpu": cpu,
            "memory": memory.percent,
            "gpu": gpu if gpu is not None else "Not available",
        }

    def plot_metrics(self):
        plot_data = {
            "epoch": list(range(1, len(self.metric_history["loss"]) + 1)),
            "train_loss": self.metric_history["loss"],
            "val_loss": self.metric_history["val_loss"],
        }
        for metric in self.metrics:
            plot_data[metric] = self.metric_history[metric]
            plot_data[f"val_{metric}"] = self.metric_history[f"val_{metric}"]

        sns.set(style="whitegrid")
        df = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 8))
        for metric in self.metrics:
            plt.plot(
                df["epoch"],
                df[metric],
                label=metric,
                marker="o",
                linestyle="-",
            )
            plt.plot(
                df["epoch"],
                df[f"val_{metric}"],
                label=f"val_{metric}",
                marker="x",
                linestyle="--",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training and Validation Metrics")
        plt.legend()
        plt.grid()
        plt.savefig(self.config.run_dir / "metrics_plot.png")
        plt.close()

        logger.info(f"Metrics plot saved at {self.config.run_dir / 'metrics_plot.png'}")

        # plot losses
        plt.figure(figsize=(12, 6))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(self.config.run_dir / "loss_plot.png")
        plt.close()
        logger.info(f"Loss plot saved at {self.config.run_dir / 'loss_plot.png'}")


if __name__ == "__main__":
    from ids_expt.models.ffnn import FFNN
    from ids_expt.data.dataset import (
        DataSetConfig,
        SamplingMethod,
        CLFDataSet as DataSet,
    )

    trainer_cfg = NNTrainerConfig(
        result_dir=Path("results"),
        expt_name="ffnn",
        run_name="run_1",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
    )
    model = FFNN(input_size=46, hidden_layers=[90] * 9, output_size=9)
    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(sampling_method=SamplingMethod.UNDERSAMPLE, max_data=100)
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
