# read AE model here
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from loguru import logger
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
        adv_trained_model: torch.nn.Module = None,
        input_shape=(1, 6 * 32, 8 * 32),
        loss=torch.nn.CrossEntropyLoss(),
        output_dir: Path = Path(
            r"C:\Users\Viper\Desktop\thesis_code\results\adversarial_blocking"
        ),
        batch_size: int = 64,
        targeted=True,
        num_workers: int = 5,
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
        self.adv_trained_model = adv_trained_model
        self.targeted = targeted
        self.num_workers = num_workers
        if self.adv_trained_model is not None:
            self.adv_trained_model.to(self.device).eval()

    def run(self, results_dir: Path = None):
        if results_dir is None:
            results_dir = self.output_dir
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to {results_dir}")
        results_dict = {}
        logger.info("Running no attack evaluation")

        for attack in self.attacks:
            # if atk has eps then use it in the name
            if not hasattr(attack, "eps"):
                attack.eps = -1
            attack_name = attack.__class__.__name__ + f"_eps_{attack.eps}"
            logger.info(f"Running attack: {attack_name} | Trageted: {self.targeted}")
            predictions = []
            blocked_predictions = []
            adv_trained_predictions = []
            targets = []
            mse_losses = []
            written_normal_image = False
            for images, labels in tqdm(
                torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
                disable=not sys.stdout.isatty(),
            ):
                images = images.to(torch.float32)
                if self.targeted:
                    # Generate false labels for targeted attacks
                    false_labels = labels.clone()
                    # for i in range(false_labels.size(0)):
                    #     idx = false_labels[i].argmax().item()
                    #     # Set the false label to a different class
                    #     false_labels[i, idx] = 0
                    #     random_idx = idx
                    #     while random_idx == idx:
                    #         random_idx = np.random.randint(
                    #             0, self.test_dataset.num_classes
                    #         )
                    #     false_labels[i, random_idx] = 1
                    # false_labels = false_labels.to(self.device)
                    adv_images = attack.generate(
                        x=images.numpy(), y=false_labels.cpu().numpy()
                    )
                else:
                    adv_images = attack.generate(x=images.numpy())

                adv_images = torch.tensor(adv_images, dtype=torch.float32)
                with torch.no_grad():
                    recon_image = self.blocking_model(adv_images.to(self.device))

                # write sample normal images to disk
                if not written_normal_image:
                    # loop through labels to find the normal image
                    lbl_idxs = labels.argmax(dim=1).cpu().numpy()
                    # it has idx, label name
                    idx_lbl = self.test_dataset.label_index
                    for idx, lbl in enumerate(lbl_idxs):
                        if idx_lbl[lbl] == "NORMAL":
                            # write normal, adversarial and blocked images
                            normal_img = images[idx].cpu().squeeze().numpy()
                            adv_img = adv_images[idx].cpu().squeeze().numpy()
                            blocked_img = (
                                recon_image[idx].detach().cpu().squeeze().numpy()
                            )

                            sample_dir = results_dir / "sample_images"
                            sample_dir.mkdir(parents=True, exist_ok=True)

                            # add atk_name in fname
                            cv2.imwrite(
                                sample_dir / f"normal_{attack_name}.png",
                                (normal_img * 255).astype(np.uint8),
                            )
                            cv2.imwrite(
                                sample_dir / f"adv_{attack_name}.png",
                                (adv_img * 255).astype(np.uint8),
                            )
                            cv2.imwrite(
                                sample_dir / f"recon_{attack_name}.png",
                                (blocked_img * 255).astype(np.uint8),
                            )
                            written_normal_image = True
                            logger.info(
                                f"Normal, adversarial and blocked images saved for {attack_name}"
                            )
                        if written_normal_image:
                            break

                mse_loss = torch.nn.functional.mse_loss(
                    recon_image, images.to(self.device)
                )
                mse_losses.append(mse_loss.item())
                if self.adv_trained_model is not None:
                    with torch.no_grad():
                        adv_trained_logits, adv_trained_proba = self.adv_trained_model(
                            adv_images.to(self.device)
                        )
                    adv_trained_preds = torch.argmax(adv_trained_proba, dim=1)
                    adv_trained_predictions.extend(adv_trained_preds.cpu().numpy())

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
            if self.adv_trained_model is not None:
                adv_trained_f1 = multiclass_f1_score(
                    torch.tensor(adv_trained_predictions),
                    torch.tensor(targets),
                    num_classes=self.test_dataset.num_classes,
                    average="macro",
                )
                _, cm_adv = get_confusion_matrix(
                    adv_trained_predictions,
                    targets,
                    self.test_dataset.label_encoding,
                    out_file=results_dir / f"{attack_name}_adv_trained.png",
                )
            atk_name = attack.__class__.__name__ + f"_eps_{attack.eps}"
            _, cm = get_confusion_matrix(
                predictions,
                targets,
                self.test_dataset.label_encoding,
                out_file=results_dir / f"{atk_name}_attack.png",
            )
            _, cm2 = get_confusion_matrix(
                blocked_predictions,
                targets,
                self.test_dataset.label_encoding,
                out_file=results_dir / f"{atk_name}_blocked.png",
            )
            res_dict = {
                "f1_score": non_blocking_f1.item(),
                "blocked_f1_score": blocking_f1.item(),
                "confusion_matrix": cm.tolist(),
                "blocked_confusion_matrix": cm2.tolist(),
                "mse_loss": mse_loss.item(),
            }

            if self.adv_trained_model is not None:
                res_dict["adv_trained_f1_score"] = adv_trained_f1.item()
                res_dict["adv_trained_confusion_matrix"] = cm_adv.tolist()
            results_dict[attack_name] = res_dict
            logger.info(
                f"{atk_name} F1 Score: {non_blocking_f1.item()}, "
                f"Blocked F1 Score: {blocking_f1.item()}"
                f", Adv Trained F1 Score: {adv_trained_f1.item() if adv_trained_f1 else 'N/A'}"
            )
            logger.info(f"Confusion Matrix: {cm}")
            logger.info(f"Blocked Confusion Matrix: {cm2}")

        predictions = []
        blocked_predictions = []
        adv_trained_predictions = []
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
                if self.adv_trained_model is not None:
                    adv_logits, adv_proba = self.adv_trained_model(
                        images.to(self.device)
                    )
                    adv_trained_preds = torch.argmax(adv_proba, dim=1)
                    adv_trained_predictions.extend(adv_trained_preds.cpu().numpy())
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
        adv_trained_f1 = None
        if self.adv_trained_model is not None:
            adv_trained_f1 = multiclass_f1_score(
                torch.tensor(adv_trained_predictions),
                torch.tensor(targets),
                num_classes=self.test_dataset.num_classes,
                average="macro",
            )
            _, cm_adv = get_confusion_matrix(
                adv_trained_predictions,
                targets,
                self.test_dataset.label_encoding,
                out_file=results_dir / "adv_trained.png",
            )
        _, cm = get_confusion_matrix(
            predictions,
            targets,
            self.test_dataset.label_encoding,
            out_file=results_dir / "no_attack.png",
        )
        _, cm2 = get_confusion_matrix(
            blocked_predictions,
            targets,
            self.test_dataset.label_encoding,
            out_file=results_dir / "no_attack_blocked.png",
        )
        res_dict = {
            "f1_score": non_blocking_f1.item(),
            "blocked_f1_score": blocking_f1.item(),
            "mse_loss": mse_loss.item(),
            "confusion_matrix": cm.tolist(),
            "blocked_confusion_matrix": cm2.tolist(),
        }
        if self.adv_trained_model is not None:
            res_dict["adv_trained_f1_score"] = adv_trained_f1.item()
            res_dict["adv_trained_confusion_matrix"] = cm_adv.tolist()
        results_dict["no_attack"] = res_dict
        logger.info(
            f"No attack F1 Score: {non_blocking_f1.item()}, "
            f"Blocked F1 Score: {blocking_f1.item()}"
            f", Adv Trained F1 Score: {adv_trained_f1.item() if adv_trained_f1 else 'N/A'}"
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
