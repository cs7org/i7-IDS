from ids_expt.models.trainer import NNTrainer, NNTrainerConfig
from ids_expt.data.adversarial_data_pair import AdversarialDataPair
import torch
from loguru import logger


class AdvTrainer(NNTrainer):
    """Adversarial training as adversarial handling."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: NNTrainerConfig,
        train_dataset: AdversarialDataPair,
        val_dataset: AdversarialDataPair,
        **kwargs
    ):
        super().__init__(config, model, train_dataset, val_dataset, **kwargs)
        self.model.to(self.device)

    def forward_step(self, batch):
        inputs, targets, labels = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        labels = labels.to(self.device)
        logits, proba = self.model(inputs)

        loss = self.criterion(logits, labels.argmax(dim=1))
        self.metrics.update(proba.argmax(dim=1), labels.argmax(dim=1))
        metrics = self.metrics.compute()
        # itemize
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }
        return proba, loss, metrics
