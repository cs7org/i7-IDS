from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
import torch


class AETrainer(NNTrainer):
    def __init__(
        self,
        config: NNTrainerConfig,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        criterion=torch.nn.MSELoss(reduction="mean"),
    ):
        super().__init__(config, model, train_dataset, val_dataset, criterion)

    def forward_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        return outputs, loss, dict()
