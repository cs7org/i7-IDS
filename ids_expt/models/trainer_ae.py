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
        criterion=torch.nn.MSELoss(reduction="mean"),
        is_sparse_ae: bool = True,
    ):
        super().__init__(config, model, train_dataset, val_dataset, criterion)
        self.is_sparse_ae = is_sparse_ae

    def at_batch_end(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def at_epoch_end(self):
        # read_ val_dataset and randomly sample 10 images from it
        # then visualize the original images and the reconstructed images side by side
        import matplotlib.pyplot as plt
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
                    inputs, targets = self.val_dataset[random_idx]
                    label = self.val_dataset.current_label
                    adv_str = self.val_dataset.data_kind
                    if label == curr_lbl:
                        inputs = inputs.unsqueeze(0).to(self.device)
                        targets = targets.unsqueeze(0).to(self.device)
                        outputs = self.model(inputs)

                        if self.is_sparse_ae:
                            decoded, _ = outputs
                        else:
                            decoded = outputs
                        mse = self.criterion(decoded, targets)
                        # reverse normalize and uint8 numpy
                        inputs_np = (inputs.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )
                        targets_np = (targets.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )
                        decoded_np = (decoded.cpu().numpy().squeeze() * 255).astype(
                            np.uint8
                        )

                        all_images.append(inputs_np)
                        all_images.append(decoded_np)
                        all_images.append(targets_np)
                        all_titles.append(f"{adv_str}: {label}")
                        all_titles.append(f"Recon. (MSE: {mse.item():.4f})")
                        all_titles.append("Target")
                        break
            fig = subplot_images(
                all_images,
                titles=all_titles,
                fig_size=(10, 15),
                order=(len(all_labels), 3),
                axis=False,
                show=False,
            )
            out_dir = self.config.run_dir / "progress_images"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"epoch_{self.epoch}.png")
            logger.info(
                f"Saved reconstruction images for epoch {self.epoch} to {out_dir / f'epoch_{self.epoch}.png'}"
            )

    def forward_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)
        if self.is_sparse_ae:
            decoded, encoded = outputs
            loss = self.criterion(decoded, targets)
            sparse_loss = self.model.sparsity_penalty(encoded)
            total_loss = loss + sparse_loss

            return (
                outputs,
                total_loss,
                dict(
                    recon_loss=loss.item(),
                    sparsity_loss=sparse_loss.item(),
                    total_loss=total_loss.item(),
                ),
            )
        else:
            loss = self.criterion(outputs, targets)
            return outputs, loss, dict(recon_loss=loss.item())
