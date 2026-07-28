"""
Training loop — used for the expert, the probes, and every baseline.

Each batch arrives as (B, window_size, C) and is split at `seq_len` into input
and forecast target, so one loop serves every model and dataset.

Two modes:
  fit(...)                      train for a fixed number of epochs
  fit(..., val_loader=, patience=)  early-stop on validation loss and RESTORE
                                    the best weights before returning

Restoring the best weights matters: without it, a probe would be scored on
whatever overfitted state it happened to stop in, so utility numbers would
reflect the stopping point rather than the data being judged.

Callbacks fire at epoch boundaries and are how the expert trajectory gets
recorded (`SimpleRecorder`) and how the validation curve is captured for phase
detection (`ValLossRecorderCallback`).
"""

import copy
import torch

from ts_distill.trainer.trainer.base import BaseTrainer

from ts_distill._logging import get_logger

logger = get_logger(__name__)


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
        """Run one epoch. Returns the mean batch loss."""
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
        """
        Args:
            val_loader: If provided, evaluates after every epoch and applies early stopping.
            patience:   Number of epochs without val improvement before stopping.
                        Best weights are restored when training ends (by early stop or epoch limit).
        """
        if callbacks is None:
            callbacks = []

        for callback in callbacks:
            callback.on_train_begin(self.model)

        best_val_mse = float('inf')
        no_improve   = 0
        best_weights = None

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)

            for callback in callbacks:
                callback.on_epoch_end(self.model, epoch, avg_loss)

            if val_loader is not None:
                val_mse = self.eval_epoch(val_loader)
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    no_improve   = 0
                    best_weights = copy.deepcopy(self.model.state_dict())
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"   Early stopping at epoch {epoch + 1} - val MSE: {val_mse:.6f} (patience={patience})")
                        break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        for callback in callbacks:
            callback.on_train_end(self.model)
