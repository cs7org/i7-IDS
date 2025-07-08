from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
import torch
from ids_expt.models.cnn_ae import DDSA_CNN


class AETrainer(NNTrainer):

    def __init__(
        self,
        config: NNTrainerConfig,
        model: DDSA_CNN,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        criterion=torch.nn.MSELoss(reduction="mean"),
    ):
        super().__init__(config, model, train_dataset, val_dataset, criterion)

    def forward_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs, bottleneck = self.model(inputs)

        loss = self.criterion(outputs, targets)
        sparse_loss = self.model.sparsity_loss(bottleneck)
        total_loss = loss + sparse_loss * 0.1

        return (
            outputs,
            total_loss,
            dict(
                recon_loss=loss.item(),
                sparsity_loss=sparse_loss.item(),
                total_loss=total_loss.item(),
            ),
        )
