from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from loguru import logger
from art.estimators.classification import PyTorchClassifier
from ids_expt.utils.confusion_matrix import get_confusion_matrix
from torchmetrics import F1Score
from pathlib import Path
from ids_expt.data.session_image_dataset import (
    TorchImageDataset,
)
import cv2
import numpy as np
import torch
from tqdm import tqdm


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
            r"C:\Users\Viper\Desktop\thesis_code\results\adversarial_experiment"
        ),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.attacks = attacks
        self.input_shape = input_shape
        self.f1_score = F1Score(
            task="multiclass",
            num_classes=train_dataset.data[train_dataset.config.label_column].nunique(),
            average="macro",
        )
        self.output_dir = output_dir / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        if Path(self.output_dir / "results.txt").exists():
            # delete it
            (self.output_dir / "results.txt").unlink()

            # no attack
        logger.info("Running no attack evaluation")
        no_attack_classifier = PyTorchClassifier(
            model=ClfModel(self.model),
            loss=torch.nn.CrossEntropyLoss(),
            clip_values=(0, 1),
            input_shape=self.input_shape,
            nb_classes=len(self.train_dataset.label_encoding),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001),
        )
        predictions = []
        targets = []
        for images, labels in tqdm(
            torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        ):
            images = images.to(torch.float32)
            logits, proba = self.model(images.to("cuda"))
            preds = torch.argmax(proba, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.argmax(dim=1).cpu().numpy())
        f1 = self.f1_score(torch.tensor(predictions), torch.tensor(targets))
        logger.info(f"F1 Score on Original Images: {f1:.4f}")
        cm = get_confusion_matrix(
            predictions,
            targets,
            self.train_dataset.label_encoding,
            out_file=self.output_dir / "no_attack.png",
        )
        logger.info(f"Confusion Matrix:\n{cm}")
        with open(self.output_dir / "results.txt", "a") as f:
            f.write(f"No Attack - F1 Score: {f1.item()}\n")
        logger.info("No attack evaluation completed.\n")

        for attack in self.attacks:
            logger.info(
                f"Running attack: {attack.__class__.__name__} with eps: {attack.eps}"
            )
            adv_predictions = []
            targets = []
            for images, labels in tqdm(
                torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=64, shuffle=False
                )
            ):
                images = images.to(torch.float32)
                adv_images = attack.generate(x=images.numpy())
                adv_logits, adv_proba = self.model(torch.tensor(adv_images).to("cuda"))
                adv_preds = torch.argmax(adv_proba, dim=1)
                adv_predictions.extend(adv_preds.cpu().numpy())
                targets.extend(labels.argmax(dim=1).cpu().numpy())
            f1 = self.f1_score(torch.tensor(adv_predictions), torch.tensor(targets))

            logger.info(f"F1 Score on Adversarial Examples: {f1:.4f}")
            cm = get_confusion_matrix(
                adv_predictions,
                targets,
                self.train_dataset.label_encoding,
                out_file=self.output_dir
                / f"{attack.__class__.__name__}_{attack.eps}.png",
            )
            logger.info(f"Confusion Matrix:\n{cm}")
            with open(self.output_dir / "results.txt", "a") as f:
                f.write(
                    f"{attack.__class__.__name__} - eps: {attack.eps}, F1 Score: {f1.item()}\n"
                )

    def _generate_image(self, attack, out_folder, dataset, preserve_blank_areas=False):
        num_samples = len(dataset)
        if not (self.output_dir / out_folder).exists():
            (self.output_dir / out_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating adversarial examples for {num_samples} samples")
        for sample_idx in tqdm(
            range(num_samples), desc="Generating adversarial examples"
        ):
            row = dataset.dataset.data_df.iloc[sample_idx]
            filename = Path(row["file_path"]).name
            adv_img_path = self.output_dir / out_folder / filename
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

    def _generate_tabular(self, attack, out_folder, dataset):

        if not (self.output_dir / out_folder).exists():
            (self.output_dir / out_folder).mkdir(parents=True, exist_ok=True)
        data_pairs = []
        for inp, lbl in tqdm(
            torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False),
            desc="Generating adversarial examples",
        ):
            adv_x = attack.generate(x=inp.numpy())
            # inp, adv pair
            for i in range(len(inp)):
                data_pairs.append((adv_x[i], lbl[i].numpy()))
        # Save adversarial examples as npz file
        adv_data_path = (
            self.output_dir
            / out_folder
            / f"adversarial_inp_{dataset.data_type.value}.npz"
        )
        np.savez_compressed(
            adv_data_path,
            inputs=np.array([pair[0] for pair in data_pairs]),
            adversarial=np.array([pair[1] for pair in data_pairs]),
        )

        logger.info(f"Generated {len(data_pairs)} adversarial examples")

    def generate(
        self,
        attack: ProjectedGradientDescent,
        out_folder: str = "",
        preserve_blank_areas=False,
        is_image_dataset=True,
    ):
        logger.info(
            f"Generating adversarial examples for attack: {attack.__class__.__name__}"
        )
        if is_image_dataset:
            self._generate_image(
                attack, out_folder, self.train_dataset, preserve_blank_areas
            )
            logger.info(
                f"Train Adversarial examples saved to {self.output_dir / out_folder}"
            )
            self._generate_image(
                attack, out_folder, self.test_dataset, preserve_blank_areas
            )
        else:
            self._generate_tabular(attack, out_folder, self.train_dataset)
            logger.info(
                f"Train Adversarial examples saved to {self.output_dir / out_folder}"
            )
            self._generate_tabular(attack, out_folder, self.test_dataset)
        logger.info("Validation adversarial examples saved.")
