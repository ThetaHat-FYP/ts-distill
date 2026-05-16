import copy
import torch

from ts_distill.trainer.trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer with callback support, explicit sequence/prediction length, and early stopping.
    Each batch is expected as shape (B, window_size, features);
    the window is split at seq_len into inputs and targets.
    """

    def __init__(self, model, optimizer, criterion, device, seq_len=None):
        super().__init__(model, optimizer, criterion, device)
        self.seq_len = seq_len

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            split = self.seq_len if self.seq_len is not None else batch.shape[1] // 2
            inputs  = batch[:, :split, :]
            targets = batch[:, split:, :]

            output = self.model(inputs)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def eval_epoch(self, dataloader) -> float:
        """Evaluates the model on a dataloader without updating weights. Returns mean batch loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch   = batch.to(self.device)
                split   = self.seq_len if self.seq_len is not None else batch.shape[1] // 2
                inputs  = batch[:, :split, :]
                targets = batch[:, split:, :]
                output  = self.model(inputs)
                total_loss += self.criterion(output, targets).item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def fit(self, dataloader, epochs: int, callbacks=None, val_loader=None, patience: int = 3):
        if callbacks is None:
            callbacks = []

        for callback in callbacks:
            callback.on_train_begin(self.model)

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)

            val_mse = None
            if val_loader is not None:
                val_mse = self.eval_epoch(val_loader)

            for callback in callbacks:
                callback.on_epoch_end(
                    self.model,
                    epoch,
                    avg_loss,
                    val_loss=val_mse
                )

        for callback in callbacks:
            callback.on_train_end(self.model)
