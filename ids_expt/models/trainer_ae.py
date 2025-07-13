from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
import torch
from ids_expt.models.cnn_ae import DDSA_CNN
from loguru import logger


class AETrainer(NNTrainer):

    def __init__(
        self,
        config: NNTrainerConfig,
        model: DDSA_CNN,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        clf_model: torch.nn.Module = None,
        clf_loss_weight: float = 0.1,
        ae_loss_weight: float = 0.9,
        criterion=torch.nn.MSELoss(reduction="mean"),
        ae_type: str = "ddsa",
    ):
        super().__init__(config, model, train_dataset, val_dataset, criterion)
        self.ae_type = ae_type
        self.clf_model = clf_model
        self.clf_loss_weight = clf_loss_weight
        self.ae_loss_weight = ae_loss_weight
        self.model.to(self.device)

    def at_batch_end(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def at_epoch_end(self):
        # plot one image of all labels
        import numpy as np
        from ids_expt.utils.vis import subplot_images

        self.model.eval()
        with torch.no_grad():
            all_labels = [k for k in self.val_dataset.label_counts.keys()]
            all_images = []
            all_titles = []
            for curr_lbl in all_labels:
                while True:
                    random_idx = torch.randint(0, len(self.val_dataset), (1,)).item()
                    inputs, targets, label_tensor = self.val_dataset[random_idx]

                    label = self.val_dataset.current_label
                    adv_str = self.val_dataset.data_kind
                    if label == curr_lbl:
                        inputs = inputs.unsqueeze(0).to(self.device)
                        targets = targets.unsqueeze(0).to(self.device)
                        outputs = self.model(inputs)

                        if self.ae_type == "ddsa":
                            decoded, _ = outputs
                        elif self.ae_type == "vae":
                            decoded, _, _, _ = outputs
                            # VAE outputs are (decoded, z, mean, logvar)
                        else:
                            decoded = outputs
                        if self.clf_model is not None:
                            logits, _ = self.clf_model(decoded)
                            pred_lbl = logits.argmax(dim=1).item()
                            labels = list(self.val_dataset.label_encoding.keys())
                            pred_lbl = labels[pred_lbl]
                        else:
                            pred_lbl = ""

                        mse = self.criterion(decoded, targets)
                        # reverse normalize and uint8 numpy
                        inputs_np = (inputs.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )
                        decoded_np = (decoded.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )
                        targets_np = (targets.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )
                        abs_diff = np.abs(targets_np - decoded_np).astype(np.uint8)

                        all_images.append(inputs_np)
                        all_images.append(decoded_np)
                        all_images.append(targets_np)
                        all_images.append(abs_diff)
                        all_titles.append(f"{adv_str}: {label}")
                        all_titles.append(f"Out: ({pred_lbl})")
                        all_titles.append("Target")
                        all_titles.append(f"Out-Tar.: (MSE: {mse.item():.4f})")
                        break
            fig = subplot_images(
                all_images,
                titles=all_titles,
                fig_size=(10, 15),
                order=(len(all_labels), 4),
                axis=False,
                show=False,
            )
            out_dir = self.config.run_dir / "progress_images"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"epoch_{self.epoch}.png")
            fig.close()

            logger.info(
                f"Saved reconstruction images for epoch {self.epoch} to {out_dir / f'epoch_{self.epoch}.png'}"
            )

    def get_clf_loss(
        self,
        target_img: torch.Tensor,
        recon_img: torch.Tensor,
        label_tensor: torch.Tensor = None,
    ):
        from torchmetrics.functional.classification import multiclass_f1_score

        target_img = target_img.to(self.device)
        recon_img = recon_img.to(self.device)
        label_tensor = label_tensor.to(self.device)

        # Get the logits from the classifier
        target_logits, _ = self.clf_model(target_img)
        recon_logits, _ = self.clf_model(recon_img)

        targets = target_logits.argmax(dim=1)
        recons = recon_logits.argmax(dim=1)

        rec_f1 = multiclass_f1_score(
            preds=recons,
            target=label_tensor.argmax(dim=1).to(self.device),
            num_classes=label_tensor.shape[1],
            average="macro",
        )

        inp_f1 = multiclass_f1_score(
            preds=targets,
            target=label_tensor.argmax(dim=1),
            num_classes=label_tensor.shape[1],
            average="macro",
        )

        # logger.info(
        #     f"recon: {recon_logits.shape}, target: {target_logits.shape}, label: {label_tensor.shape}"
        # )

        # Calculate the classification loss as cross-entropy loss
        # clf_loss = self.criterion(recon_logits, label_tensor)
        clf_loss = torch.nn.CrossEntropyLoss(reduction="mean")(
            recon_logits, label_tensor.argmax(dim=1)
        )
        return clf_loss, rec_f1, inp_f1

    def forward_step(self, batch):
        inputs, targets, label_tensor = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)

        if self.ae_type == "ddsa":
            decoded, encoded = outputs
            loss = self.criterion(decoded, targets)
            sparse_loss = self.model.sparsity_penalty(encoded)
            total_loss = loss + sparse_loss

            clf_loss = 0
            f1 = 0
            if self.clf_model is not None:
                clf_loss, rec_f1, inp_f1 = self.get_clf_loss(
                    targets, decoded, label_tensor
                )
                total_loss = (
                    self.ae_loss_weight * total_loss + self.clf_loss_weight * clf_loss
                )

            return (
                outputs,
                total_loss,
                dict(
                    recon_loss=loss.item(),
                    sparsity_loss=sparse_loss.item(),
                    total_loss=total_loss.item(),
                    clf_loss=clf_loss.item() if self.clf_model is not None else 0.0,
                    rec_f1=rec_f1.item() if self.clf_model is not None else 0.0,
                    inp_f1=inp_f1.item() if self.clf_model is not None else 0.0,
                ),
            )
        elif self.ae_type == "vae":
            decoded, z, mean, logvar = outputs
            total_loss, recon_loss, kl_loss = self.model.loss_function(
                inputs, decoded, mean, logvar
            )
            clf_loss = 0
            if self.clf_model is not None:
                clf_loss, rec_f1, inp_f1 = self.get_clf_loss(
                    targets, decoded, label_tensor
                )
                total_loss = (
                    self.ae_loss_weight * total_loss + self.clf_loss_weight * clf_loss
                )
            return (
                outputs,
                total_loss,
                dict(
                    recon_loss=recon_loss.item(),
                    kl_loss=kl_loss.item(),
                    total_loss=total_loss.item(),
                    clf_loss=clf_loss.item() if self.clf_model is not None else 0.0,
                    rec_f1=rec_f1.item() if self.clf_model is not None else 0.0,
                    inp_f1=inp_f1.item() if self.clf_model is not None else 0.0,
                ),
            )
        elif self.ae_type == "unet":
            outputs = self.model(inputs)
            if self.clf_model is not None:
                clf_loss, rec_f1, inp_f1 = self.get_clf_loss(
                    targets, outputs, label_tensor
                )
                loss = self.criterion(outputs, targets) + clf_loss
                return (
                    outputs,
                    loss,
                    dict(
                        recon_loss=self.criterion(outputs, targets).item(),
                        clf_loss=clf_loss.item(),
                        rec_f1=rec_f1.item(),
                        inp_f1=inp_f1.item(),
                        total_loss=loss.item(),
                    ),
                )
        else:
            loss = self.criterion(outputs, targets)
            return outputs, loss, dict(recon_loss=loss.item())
