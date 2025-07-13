from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from loguru import logger
from art.estimators.classification import PyTorchClassifier
from ids_expt.utils.confusion_matrix import get_confusion_matrix
from torchmetrics import F1Score
from torchmetrics.functional.classification import multiclass_f1_score
from pathlib import Path
from ids_expt.data.session_image_dataset import (
    TorchImageDataset,
)
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


class AdversarialExperiment:
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        attacks: list[ProjectedGradientDescent],
        train_dataset: TorchImageDataset,
        test_dataset: TorchImageDataset,
        input_shape=(1, 6 * 32, 8 * 32),
        loss=torch.nn.CrossEntropyLoss(),
        output_dir: Path = Path(
            r"C:\Users\Viper\Desktop\thesis_code\results\adversarial_attacks"
        ),
        batch_size: int = 64,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.attacks = attacks
        self.input_shape = input_shape

        self.output_dir = output_dir / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss = loss
        self.batch_size = batch_size

    def run(self, results_dir: Path = None):
        if results_dir is None:
            results_dir = self.output_dir
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to {results_dir}")
        results_dict = {}
        logger.info("Running no attack evaluation")

        predictions = []
        targets = []
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
                logits, proba = self.model(images.to("cuda"))
                preds = torch.argmax(proba, dim=1)
                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.argmax(dim=1).cpu().numpy())
        # Calculate F1 score

        f1 = multiclass_f1_score(
            torch.tensor(predictions),
            torch.tensor(targets),
            num_classes=self.test_dataset.num_classes,
            average="macro",
        )
        logger.info(f"F1 Score on Original Images: {f1:.4f}")
        cm = get_confusion_matrix(
            predictions,
            targets,
            self.train_dataset.label_encoding,
            out_file=results_dir / "no_attack.png",
        )
        logger.info(f"Confusion Matrix:\n{cm}")
        results_dict["no_attack"] = {
            "f1_score": f1.item(),
            "confusion_matrix": cm.tolist(),
        }
        logger.info("No attack evaluation completed.\n")

        for attack in self.attacks:
            logger.info(
                f"Running attack: {attack.__class__.__name__} with eps: {attack.eps}"
            )
            adv_predictions = []
            targets = []
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
                adv_logits, adv_proba = self.model(torch.tensor(adv_images).to("cuda"))
                adv_preds = torch.argmax(adv_proba, dim=1)
                adv_predictions.extend(adv_preds.cpu().numpy())
                targets.extend(labels.argmax(dim=1).cpu().numpy())
            f1 = multiclass_f1_score(
                torch.tensor(adv_predictions),
                torch.tensor(targets),
                num_classes=self.test_dataset.num_classes,
                average="macro",
            )

            logger.info(f"F1 Score on Adversarial Examples: {f1:.4f}")
            cm = get_confusion_matrix(
                adv_predictions,
                targets,
                self.train_dataset.label_encoding,
                out_file=results_dir / f"{attack.__class__.__name__}_{attack.eps}.png",
            )
            logger.info(f"Confusion Matrix:\n{cm}")
            atk_name = attack.__class__.__name__ + f"_eps_{attack.eps}"
            results_dict[atk_name] = {
                "f1_score": f1.item(),
                "confusion_matrix": cm.tolist(),  # Convert to list for JSON serialization
            }
        # Save results to a file
        results_file = results_dir / "adv_f1_scores.json"

        with open(results_file, "w") as f:
            import json

            json.dump(results_dict, f, indent=4)
        logger.info("Adversarial evaluation completed.")
        logger.info(f"Results saved to {results_file}")

    def _generate_image(
        self,
        attack,
        out_folder,
        dataset: TorchImageDataset,
        preserve_blank_areas=False,
        as_npz=True,
    ):
        sample_images = []
        saved_sample = False
        num_samples = len(dataset)
        result_dir = self.output_dir / out_folder / dataset.data_type.value
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating adversarial examples for {num_samples} samples")
        if as_npz:
            batch_idx = 0
            lbl_str_mapping = {
                np.array(v).argmax(): k
                for k, v in dataset.dataset.label_encoding.items()
            }
            for images, labels in tqdm(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                ),
                desc="Generating adversarial examples",
                disable=not sys.stdout.isatty(),
            ):
                adv_images = attack.generate(x=images.numpy())

                lbls_str = [
                    lbl_str_mapping[lbl] for lbl in labels.argmax(dim=1).numpy()
                ]
                for i in range(len(images)):
                    adv_img = adv_images[i].reshape(self.input_shape[1:]) * 255
                    input_img = images[i].reshape(self.input_shape[1:]).numpy() * 255
                    lbl_str = lbls_str[i]
                    # Save batch to npz file
                    adv_data_path = (
                        result_dir
                        / f"{lbl_str}_{batch_idx}_{i}_{self.output_dir.name}.npz"
                    )
                    np.savez_compressed(
                        adv_data_path,
                        inputs=input_img.astype(np.uint8),
                        adversarial=adv_img.astype(np.uint8),
                        label_str=lbl_str,
                    )
                    if not saved_sample:
                        # Save the first sample as an image
                        sample_images.append(input_img.astype(np.uint8))
                        sample_images.append(adv_img.astype(np.uint8))
                        saved_sample = True

        else:
            for sample_idx in tqdm(
                range(num_samples),
                desc="Generating adversarial examples",
                disable=not sys.stdout.isatty(),
            ):
                row = dataset.dataset.data_df.iloc[sample_idx]
                filename = Path(row["file_path"]).name
                adv_img_path = result_dir / filename
                if adv_img_path.exists():
                    logger.warning(f"Adversarial image {adv_img_path} already exists.")
                    continue

                # arr, arr, str
                image, label, label_str = dataset.dataset[sample_idx]
                adv_img = attack.generate(
                    x=image.reshape(1, *self.input_shape).astype(np.float32)
                )
                adv_img = adv_img.reshape(self.input_shape[1:])
                # reverse normalize
                adv_img = adv_img * 255
                adv_img = adv_img.astype(np.uint8)
                # write to file
                if preserve_blank_areas:
                    row_mask = np.all(image == 0, axis=1)
                    col_mask = np.all(image == 0, axis=0)

                    # Apply masks to adversarial image
                    adv_img[row_mask, :] = 0
                    adv_img[:, col_mask] = 0
                cv2.imwrite(str(adv_img_path), adv_img)
        logger.info("Done")
        return result_dir, sample_images if as_npz else None

    def _generate_tabular(self, attack, out_folder, dataset):
        result_dir = self.output_dir / out_folder / dataset.data_type.value
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        data_pairs = []
        for inp, lbl in tqdm(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
            ),
            desc="Generating adversarial examples",
            disable=not sys.stdout.isatty(),
        ):
            adv_x = attack.generate(x=inp.numpy())
            # inp, adv pair
            for i in range(len(inp)):
                data_pairs.append((inp[i], adv_x[i], lbl[i].numpy()))
        # Save adversarial examples as npz file
        adv_data_path = result_dir / "adversarial_data.npz"
        if adv_data_path.exists():
            logger.warning(
                f"Adversarial data {adv_data_path} already exists. Overwriting."
            )
        np.savez_compressed(
            adv_data_path,
            inputs=np.array([pair[0] for pair in data_pairs]),
            adversarial=np.array([pair[1] for pair in data_pairs]),
            labels=np.array([pair[2] for pair in data_pairs]),
        )

        logger.info(f"Generated {len(data_pairs)} adversarial examples")
        return result_dir

    def generate(
        self,
        attack: ProjectedGradientDescent,
        out_folder: str = "",
        preserve_blank_areas=False,
        is_image_dataset=True,
        as_npz=True,
        compress_out_folder: bool = True,
        copy_compressed_to: Path = None,
    ):
        logger.info(
            f"Generating adversarial examples for attack: {attack.__class__.__name__}"
        )
        samples = None
        if is_image_dataset:
            train_adv_dir, samples = self._generate_image(
                attack, out_folder, self.train_dataset, preserve_blank_areas, as_npz
            )
            logger.info(
                f"Train Adversarial examples saved to {self.output_dir / out_folder}"
            )
            val_adv_dir, samples = self._generate_image(
                attack, out_folder, self.test_dataset, preserve_blank_areas, as_npz
            )

        else:
            train_adv_dir = self._generate_tabular(
                attack, out_folder, self.train_dataset
            )
            logger.info(
                f"Train Adversarial examples saved to {self.output_dir / out_folder}"
            )
            val_adv_dir = self._generate_tabular(attack, out_folder, self.test_dataset)

        if compress_out_folder and copy_compressed_to is not None:
            # Compress the output folder
            import shutil

            if not copy_compressed_to.parent.exists():
                copy_compressed_to.parent.mkdir(parents=True, exist_ok=True)

            if samples is not None:
                # Save sample images to a single npz file
                inp = samples[0]
                adv = samples[1]
                inpsaved = cv2.imwrite(
                    str(copy_compressed_to / "sample_input.png"), inp
                )
                advsaved = cv2.imwrite(str(copy_compressed_to / "sample_adv.png"), adv)
            logger.info(
                f"Sample images saved {inpsaved} to {copy_compressed_to / 'sample_input.png'} and {copy_compressed_to / 'sample_adv.png'}"
            )

            # compress train adversarial examples and copy to the specified path
            shutil.make_archive(
                copy_compressed_to / f"{out_folder}_train",
                "zip",
                train_adv_dir,
            )
            shutil.make_archive(
                copy_compressed_to / f"{out_folder}_val",
                "zip",
                val_adv_dir,
            )
            logger.info(
                f"Compressed adversarial examples saved to {copy_compressed_to / f'{out_folder}_train.zip'} and {copy_compressed_to / f'{out_folder}_val.zip'}"
            )
        else:
            # do nothing
            logger.info("Skipping compression of output folder as per configuration.")
        logger.info("Validation adversarial examples saved.")
