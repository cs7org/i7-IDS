# read AE model here
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from loguru import logger
from art.estimators.classification import PyTorchClassifier
from ids_expt.utils.confusion_matrix import get_confusion_matrix
from pathlib import Path
from ids_expt.data.session_image_dataset import (
    TorchImageDataset,
)
from torchmetrics.functional.classification import multiclass_f1_score
import cv2
import numpy as np
import torch
from tqdm import tqdm
import sys


class ClfModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ClfModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


class AdversarialBlockingExperiment:
    def __init__(
        self,
        model: torch.nn.Module,
        blocking_model: torch.nn.Module,
        model_name: str,
        attacks: list[ProjectedGradientDescent],
        test_dataset: TorchImageDataset,
        input_shape=(1, 6 * 32, 8 * 32),
        loss=torch.nn.CrossEntropyLoss(),
        output_dir: Path = Path(
            r"C:\Users\Viper\Desktop\thesis_code\results\adversarial_blocking"
        ),
        batch_size: int = 64,
    ):
        self.model = model
        self.blocking_model = blocking_model
        self.attacks = attacks
        self.test_dataset = test_dataset
        self.input_shape = input_shape
        self.loss = loss
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.blocking_model.to(self.device)
        self.blocking_model.eval()

    def run(self, results_dir: Path = None):
        if results_dir is None:
            results_dir = self.output_dir
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to {results_dir}")
        results_dict = {}
        logger.info("Running no attack evaluation")

        predictions = []
        blocked_predictions = []
        targets = []
        mse_losses = []
        with torch.no_grad():
            for images, labels in tqdm(
                torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                ),
                disable=not sys.stdout.isatty(),
            ):
                images = images.to(torch.float32)
                recon_image = self.blocking_model(images.to(self.device))
                mse_loss = torch.nn.functional.mse_loss(
                    recon_image, images.to(self.device)
                )
                logits, proba = self.model(images.to("cuda"))
                recon_logits, recon_proba = self.model(recon_image)
                preds = torch.argmax(proba, dim=1)
                recon_preds = torch.argmax(recon_proba, dim=1)

                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.argmax(dim=1).cpu().numpy())
                blocked_predictions.extend(recon_preds.cpu().numpy())
                mse_losses.append(mse_loss.item())
        mse_loss = torch.tensor(mse_losses).mean()

        non_blocking_f1 = multiclass_f1_score(
            torch.tensor(predictions),
            torch.tensor(targets),
            num_classes=self.test_dataset.num_classes,
            average="macro",
        )
        blocking_f1 = multiclass_f1_score(
            torch.tensor(blocked_predictions),
            torch.tensor(targets),
            num_classes=self.test_dataset.num_classes,
            average="macro",
        )
        cm = get_confusion_matrix(
            predictions,
            targets,
            self.test_dataset.label_encoding,
            out_file=results_dir / "no_attack.png",
        )
        cm2 = get_confusion_matrix(
            blocked_predictions,
            targets,
            self.test_dataset.label_encoding,
            out_file=results_dir / "no_attack_blocked.png",
        )
        results_dict["no_attack"] = {
            "f1_score": non_blocking_f1.item(),
            "blocked_f1_score": blocking_f1.item(),
            "mse_loss": mse_loss.item(),
            "confusion_matrix": cm.tolist(),
            "blocked_confusion_matrix": cm2.tolist(),
        }
        logger.info(
            f"No attack F1 Score: {non_blocking_f1.item()}, "
            f"Blocked F1 Score: {blocking_f1.item()}"
        )
        logger.info(f"Confusion Matrix: {cm}")
        logger.info(f"Blocked Confusion Matrix: {cm2}")

        for attack in self.attacks:
            attack_name = attack.__class__.__name__ + f"_eps_{attack.eps}"
            logger.info(f"Running attack: {attack_name}")
            predictions = []
            blocked_predictions = []
            targets = []
            mse_losses = []
            for images, labels in tqdm(
                torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                ),
                disable=not sys.stdout.isatty(),
            ):
                images = images.to(torch.float32)
                adv_images = attack.generate(x=images.numpy())
                adv_images = torch.tensor(adv_images, dtype=torch.float32)
                recon_image = self.blocking_model(adv_images.to(self.device))
                mse_loss = torch.nn.functional.mse_loss(
                    recon_image, images.to(self.device)
                )
                mse_losses.append(mse_loss.item())

                logits, proba = self.model(adv_images.to(self.device))
                recon_logits, recon_proba = self.model(recon_image)

                preds = torch.argmax(proba, dim=1)
                recon_preds = torch.argmax(recon_proba, dim=1)

                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.argmax(dim=1).cpu().numpy())
                blocked_predictions.extend(recon_preds.cpu().numpy())
            mse_loss = torch.tensor(mse_losses).mean()
            non_blocking_f1 = multiclass_f1_score(
                torch.tensor(predictions),
                torch.tensor(targets),
                num_classes=self.test_dataset.num_classes,
                average="macro",
            )
            blocking_f1 = multiclass_f1_score(
                torch.tensor(blocked_predictions),
                torch.tensor(targets),
                num_classes=self.test_dataset.num_classes,
                average="macro",
            )
            atk_name = attack.__class__.__name__ + f"_eps_{attack.eps}"
            cm = get_confusion_matrix(
                predictions,
                targets,
                self.test_dataset.label_encoding,
                out_file=results_dir / f"{atk_name}_attack.png",
            )
            cm2 = get_confusion_matrix(
                blocked_predictions,
                targets,
                self.test_dataset.label_encoding,
                out_file=results_dir / f"{atk_name}_blocked.png",
            )
            results_dict[attack_name] = {
                "f1_score": non_blocking_f1.item(),
                "blocked_f1_score": blocking_f1.item(),
                "confusion_matrix": cm.tolist(),
                "blocked_confusion_matrix": cm2.tolist(),
                "mse_loss": mse_loss.item(),
            }
            logger.info(
                f"{atk_name} F1 Score: {non_blocking_f1.item()}, "
                f"Blocked F1 Score: {blocking_f1.item()}"
            )
            logger.info(f"Confusion Matrix: {cm}")
            logger.info(f"Blocked Confusion Matrix: {cm2}")
        # Save results
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            import json

            json.dump(results_dict, f, indent=4)
        logger.info(f"Results saved to {results_file}")
        return results_dict
