import torch
import cv2
from pathlib import Path
from loguru import logger
import pandas as pd
import os
import argparse
import numpy as np
import tqdm
import sys
from ids_expt.utils.confusion_matrix import get_confusion_matrix


class DataSet(torch.utils.data.Dataset):

    def __init__(
        self,
        input_dir: Path,
        labels: list[str],
        data_split: str,
        adversarial_type: str,
        max_samples: int = -1000,
    ):

        self.input_dir = input_dir
        self.labels = labels
        self.data_split = data_split
        self.adversarial_type = adversarial_type
        self.npz_files = list(input_dir.rglob(f"{adversarial_type}/{data_split}/*.npz"))
        self.max_samples = max_samples

        if not self.npz_files:
            raise FileNotFoundError(
                f"No .npz files found in {input_dir / adversarial_type / data_split}"
            )

    def __len__(self):
        return (
            len(self.npz_files)
            if self.max_samples < 0
            else min(len(self.npz_files), self.max_samples)
        )

    def __getitem__(self, idx):
        selected_file = self.npz_files[idx]
        npz_data = np.load(selected_file)
        input_image = npz_data["inputs"]

        adversarial_image = npz_data["adversarial"]
        label = npz_data["label_str"].item()

        cv2.imwrite(
            "/home/hpc/iwi7/iwi7101h/i7-IDS/res.png",
            adversarial_image,
        )

        label_index = self.labels.index(label)
        img_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0) / 255
        adv_tensor = (
            torch.tensor(adversarial_image, dtype=torch.float32).unsqueeze(0) / 255
        )
        self.curr_label = label

        return img_tensor, adv_tensor, label_index


parser = argparse.ArgumentParser(
    description="Evaluate adversarial examples against a classifier and an autoencoder."
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing the input images.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for processing images.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/hpc/iwi7/iwi7101h/i7-IDS/results/adversarial_blocking",
    help="Directory to save the output results.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="/home/hpc/iwi7/iwi7101h/i7-IDS/results",
    help="Directory containing the project structure with models.",
)
parser.add_argument(
    "--data_type",
    type=str,
    default="normal",
    choices=["normal", "normalized"],
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=-1000,
    help="Maximum number of samples to process. Use -1000 for all samples.",
)
parser.add_argument(
    "--adv_gen_model",
    type=str,
    default="resnet18_nosampling",
    help="Adversarial generated model.",
)
parser.add_argument("--data_split", type=str, default="val")

args = parser.parse_args()
data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
data_split = args.data_split
data_type = args.data_type
batch_size = args.batch_size
max_samples = args.max_samples
adv_gen_model = args.adv_gen_model

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_dir = output_dir / adv_gen_model

# Set the project directory
results_dir = Path(args.results_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adversarial_names = [a.name for a in data_dir.iterdir()]

logger.info(f"Found {len(adversarial_names)} adversarial types.")
logger.info(f"\n{adversarial_names}")


def get_predictions(
    clf_model,
    atc_model,
    ae_model,
    input_batch,
    adversarial_batch,
):
    input_tensor = input_batch
    adversarial_tensor = adversarial_batch

    with torch.no_grad():
        # for clean data
        clean_clf_logit, proba = clf_model(input_tensor)
        clean_atc_logit, adv_proba = atc_model(input_tensor)

        # for adversarial data
        adv_clf_logit, adv_proba = clf_model(adversarial_tensor)
        adv_atc_logit, adv_adv_proba = atc_model(adversarial_tensor)

        # reconstructed input and adversarial images
        recon_input = ae_model(input_tensor)
        recon_adversarial = ae_model(adversarial_tensor)

        # block clean data
        recon_clean_clf_logit, _ = clf_model(recon_input)
        recon_clean_atc_logit, _ = atc_model(recon_input)

        # block adversarial data
        recon_adv_clf_logit, _ = clf_model(recon_adversarial)
        recon_adv_atc_logit, _ = atc_model(recon_adversarial)
    # Get predictions
    clean_clf_pred = clean_clf_logit.cpu().numpy().argmax(axis=1)
    clean_atc_pred = clean_atc_logit.cpu().numpy().argmax(axis=1)
    adv_clf_pred = adv_clf_logit.cpu().numpy().argmax(axis=1)
    adv_atc_pred = adv_atc_logit.cpu().numpy().argmax(axis=1)
    recon_input = recon_input
    recon_adversarial = recon_adversarial
    recon_clean_clf_preds = recon_clean_clf_logit.cpu().numpy().argmax(axis=1)
    recon_clean_atc_preds = recon_clean_atc_logit.cpu().numpy().argmax(axis=1)
    recon_adv_clf_preds = recon_adv_clf_logit.cpu().numpy().argmax(axis=1)
    recon_adv_atc_preds = recon_adv_atc_logit.cpu().numpy().argmax(axis=1)

    return (
        clean_clf_pred,
        clean_atc_pred,
        adv_clf_pred,
        adv_atc_pred,
        recon_input,
        recon_adversarial,
        recon_clean_clf_preds,
        recon_clean_atc_preds,
        recon_adv_clf_preds,
        recon_adv_atc_preds,
    )


labels = [
    "REPLAY",
    "DNP3_INFO",
    "DNP3_ENUMERATE",
    "STOP_APP",
    "NORMAL",
    "INIT_DATA",
    "COLD_RESTART",
    "WARM_RESTART",
    "DISABLE_UNSOLICITED",
]

clf_adv_ae_models = [
    [
        Path("image_classification/resnet18_nosampling"),
        Path("image_classification/resnet18_nosampling_adv"),
        Path("autoencoder/rdunet_normal"),
    ],
    [
        Path("image_classification/resnet18_normalized_nosampling"),
        Path("image_classification/resnet18_normalized_nosampling_adv"),
        Path("autoencoder/rdunet_normalized"),
    ],
    [
        Path("image_classification/mobilenet_v3_large_nosampling"),
        Path("image_classification/mobilenet_v3_large_nosampling_adv"),
        Path("autoencoder/rdunet_normal"),
    ],
    [
        Path("image_classification/mobilenet_v3_large_normalized_nosampling"),
        Path("image_classification/mobilenet_v3_large_normalized_nosampling_adv"),
        Path("autoencoder/rdunet_normalized"),
    ],
    [
        Path("image_classification/resnet18_nosampling"),
        Path("image_classification/resnet18_nosampling_adv"),
        Path("autoencoder/unet_custom_normal"),
    ],
    [
        Path("image_classification/resnet18_normalized_nosampling"),
        Path("image_classification/resnet18_normalized_nosampling_adv"),
        Path("autoencoder/unet_custom_normalized"),
    ],
    [
        Path("image_classification/mobilenet_v3_large_nosampling"),
        Path("image_classification/mobilenet_v3_large_nosampling_adv"),
        Path("autoencoder/unet_custom_normal"),
    ],
    [
        Path("image_classification/mobilenet_v3_large_normalized_nosampling"),
        Path("image_classification/mobilenet_v3_large_normalized_nosampling_adv"),
        Path("autoencoder/unet_custom_normalized"),
    ],
]

skip_adversarial_names = ["Carlini", "DeepFool"]

results = []
for adversarial_name in adversarial_names:
    skip_this = False
    for skip_name in skip_adversarial_names:
        if skip_name.lower() in adversarial_name.lower():
            skip_this = True
            logger.info(f"Skipping {adversarial_name} due to skip list.")
            break
    if skip_this:
        continue
    if not (data_dir / adversarial_name).exists():
        logger.warning(f"Adversarial type {adversarial_name} does not exist.")

    for clf_path, adv_path, ae_path in clf_adv_ae_models:
        is_normalized = "normalized" in clf_path.name
        curr_dtype = "normalized" if is_normalized else "normal"
        if curr_dtype != data_type:
            logger.info(
                f"Skipping {adversarial_name} for {clf_path.name}, {adv_path.name}, {ae_path.name} due to data type mismatch."
            )
            continue
        sample_images_dir = output_dir / data_split / f"{adversarial_name}"
        if not sample_images_dir.exists():
            sample_images_dir.mkdir(parents=True, exist_ok=True)

        clf_model = torch.load(
            results_dir / clf_path / "best_model_full.pth",
            map_location=device,
            weights_only=False,
        )
        clf_model.eval()
        adv_model = torch.load(
            results_dir / adv_path / "best_model_full.pth",
            map_location=device,
            weights_only=False,
        )
        adv_model.eval()
        ae_model = torch.load(
            results_dir / ae_path / "best_model_full.pth",
            map_location=device,
            weights_only=False,
        )
        ae_model.eval()

        logger.info(
            f"Evaluating {adversarial_name} with models: {clf_path.name}, {adv_path.name}, {ae_path.name}"
        )

        dataset = DataSet(
            input_dir=data_dir,
            labels=labels,
            data_split=data_split,
            adversarial_type=adversarial_name,
            max_samples=max_samples,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        logger.info(f"Dataset length: {len(dataset)}")

        true_targets = []

        # clean data
        clean_clf_predictions = []
        clean_atc_predictions = []

        # adv data
        adv_clf_predictions = []
        adv_atc_predictions = []

        # blocked clean data
        recon_clean_clf_predictions = []
        recon_clean_atc_predictions = []

        # blocked adv data
        recon_adv_clf_predictions = []
        recon_adv_atc_predictions = []

        clean_recon_mae_values = []
        adv_recon_mae_values = []
        adv_recon_inp_mae_values = []


        # label, clean, adversarial, reconstructed
        sample_images = {}
        completed_non_adversarial = False

        for batch_idx, (input_batch, adversarial_batch, label_indices) in tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"{adversarial_name}",
            disable=not sys.stdout.isatty(),
        ):
            input_batch = input_batch.to(device)
            adversarial_batch = adversarial_batch.to(device)
            label_indices = label_indices.cpu().numpy()

            (
                clean_clf_pred,
                clean_atc_pred,
                adv_clf_pred,
                adv_atc_pred,
                recon_input,
                recon_adversarial,
                recon_clean_clf_preds,
                recon_clean_atc_preds,
                recon_adv_clf_preds,
                recon_adv_atc_preds,
            ) = get_predictions(
                clf_model,
                adv_model,
                ae_model,
                input_batch,
                adversarial_batch,
            )
            #  = get_predictions(
            #     clf_model,
            #     adv_model,
            #     ae_model,
            #     input_batch,
            #     adversarial_batch,
            # )
            if not completed_non_adversarial:
                for idx, label in enumerate(label_indices):
                    label_str = dataset.labels[label]
                    if label_str not in sample_images:
                        # as uint8
                        input_image = (
                            input_batch[idx].cpu().squeeze(0).numpy() * 255
                        ).astype(np.uint8)
                        adversarial_image = (
                            adversarial_batch[idx].cpu().squeeze(0).numpy() * 255
                        ).astype(np.uint8)
                        recon_input_image = (
                            recon_input[idx].cpu().squeeze(0).numpy() * 255
                        ).astype(np.uint8)
                        recon_adversarial_image = (
                            recon_adversarial[idx].cpu().squeeze(0).numpy() * 255
                        ).astype(np.uint8)
                        sample_images[label_str] = {
                            "input": input_image,
                            "adversarial": adversarial_image,
                            "recon_input": recon_input_image,
                            "recon_adversarial": recon_adversarial_image,
                        }

            sample_completed_lbls = set(sample_images.keys())
            if len(sample_completed_lbls) == len(dataset.labels):
                completed_non_adversarial = True

            true_targets.extend(label_indices.tolist())
            clean_clf_predictions.extend(clean_clf_pred.tolist())
            clean_atc_predictions.extend(clean_atc_pred.tolist())
            adv_clf_predictions.extend(adv_clf_pred.tolist())
            adv_atc_predictions.extend(adv_atc_pred.tolist())
            recon_clean_clf_predictions.extend(recon_clean_clf_preds.tolist())
            recon_clean_atc_predictions.extend(recon_clean_atc_preds.tolist())
            recon_adv_clf_predictions.extend(recon_adv_clf_preds.tolist())
            recon_adv_atc_predictions.extend(recon_adv_atc_preds.tolist())
            clean_recon_mae = (
                torch.mean(
                    torch.abs(input_batch - recon_input).view(input_batch.size(0), -1),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            adv_recon_mae = (
                torch.mean(
                    torch.abs(adversarial_batch - recon_adversarial).view(
                        adversarial_batch.size(0), -1
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            adv_recon_inp_mae = (
                torch.mean(
                    torch.abs(input_batch - recon_adversarial).view(
                        input_batch.size(0), -1
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            clean_recon_mae_values.extend(clean_recon_mae.tolist())
            adv_recon_mae_values.extend(adv_recon_mae.tolist())
            adv_recon_inp_mae_values.extend(adv_recon_inp_mae.tolist())


        true_targets = np.array(true_targets)

        # Calculate confusion matrix and F1 score
        # f1 from clean data passed to clf and atc clf
        clean_clf_f1, clean_clf_cm = get_confusion_matrix(
            clean_clf_predictions,
            true_targets,
            labels,
        )
        clean_atc_f1, clean_atc_cm = get_confusion_matrix(
            clean_atc_predictions,
            true_targets,
            labels,
        )
        # f1 from adv data passed to clf and atc clf
        adv_clf_f1, adv_clf_cm = get_confusion_matrix(
            adv_clf_predictions,
            true_targets,
            labels,
        )
        adv_atc_f1, adv_atc_cm = get_confusion_matrix(
            adv_atc_predictions,
            true_targets,
            labels,
        )
        # f1 from recon clean data passed to clf and atc clf
        recon_clean_clf_f1, recon_clean_clf_cm = get_confusion_matrix(
            recon_clean_clf_predictions,
            true_targets,
            labels,
        )
        recon_clean_atc_f1, recon_clean_atc_cm = get_confusion_matrix(
            recon_clean_atc_predictions,
            true_targets,
            labels,
        )

        # f1 from recon adv data passed to clf and adv clf
        recon_adv_clf_f1, recon_adv_clf_cm = get_confusion_matrix(
            recon_adv_clf_predictions,
            true_targets,
            labels,
        )
        recon_adv_atc_f1, recon_adv_atc_cm = get_confusion_matrix(
            recon_adv_atc_predictions,
            true_targets,
            labels,
        )

        # Calculate MAE values

        clean_recon_mae_values = np.mean(clean_recon_mae_values)
        adv_recon_mae_values = np.mean(adv_recon_mae_values)
        adv_recon_inp_mae_values = np.mean(adv_recon_inp_mae_values)

        logger.info(
            f"Results for {adversarial_name} "
            f"Clean CLF F1: {clean_clf_f1:.4f}, "
            f"Clean ATC F1: {clean_atc_f1:.4f}, "
            f"Adv CLF F1: {adv_clf_f1:.4f}, "
            f"Adv ATC F1: {adv_atc_f1:.4f}, "
            f"Blocked Clean CLF F1: {recon_clean_clf_f1:.4f}, "
            f"Blocked Adv CLF F1: {recon_clean_atc_f1:.4f}, "
            f"Adv Blocked CLF F1: {recon_adv_clf_f1:.4f}, "
            f"Adv Blocked ATC F1: {recon_adv_atc_f1:.4f}, "
            f"Clean Recon MAE: {clean_recon_mae_values:.4f}, "
            f"Adv Recon MAE: {adv_recon_mae_values:.4f}, "
            f"Adv Recon Input MAE: {adv_recon_inp_mae_values:.4f}",
        )
        results.append(
            {
                "clf_model": clf_path.name,
                "adv_model": adv_path.name,
                "ae_model": ae_path.name,
                "data_type": data_type,
                "data_split": data_split,
                "adversarial_name": adversarial_name,
                "clean_clf_f1": clean_clf_f1,
                "clean_atc_f1": clean_atc_f1,
                "adv_clf_f1": adv_clf_f1,
                "adv_atc_f1": adv_atc_f1,
                "recon_clean_clf_f1": recon_clean_clf_f1,
                "recon_clean_atc_f1": recon_clean_atc_f1,
                "recon_adv_clf_f1": recon_adv_clf_f1,
                "recon_adv_atc_f1": recon_adv_atc_f1,
                "clean_recon_mae": clean_recon_mae_values,
                "adv_recon_mae": adv_recon_mae_values,
                "adv_recon_inp_mae": adv_recon_inp_mae_values,
                
                "clean_clf_cm": clean_clf_cm.tolist(),
                "clean_atc_cm": clean_atc_cm.tolist(),
                "adv_clf_cm": adv_clf_cm.tolist(),
                "adv_atc_cm": adv_atc_cm.tolist(),
                "recon_clean_clf_cm": recon_clean_clf_cm.tolist(),
                "recon_clean_atc_cm": recon_clean_atc_cm.tolist(),
                "recon_adv_clf_cm": recon_adv_clf_cm.tolist(),
                "recon_adv_atc_cm": recon_adv_atc_cm.tolist(),
            }
        )
        # Save sample images
        for label, images in sample_images.items():
            image_prefix = f"{label}_{ae_path.name}"
            for img_name, image in images.items():

                input_path = sample_images_dir / f"{image_prefix}_{img_name}.png"
                cv2.imwrite(str(input_path), image)
                logger.info(f"Saved {img_name} image for {label} to {input_path}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(
    output_dir / f"evaluation_results_{data_type}_{data_split}.csv", index=False
)
logger.info(
    f"Evaluation results saved to {output_dir / f'evaluation_results_{data_type}_{data_split}.csv'}"
)
logger.info("Evaluation completed successfully.")
