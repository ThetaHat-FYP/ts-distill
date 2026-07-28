"""
Standard evaluator — trains nothing, just scores a trained model on real data.

Splits each window at seq_len into (input, target), runs the model in eval mode,
and returns MSE / RMSE. Training belongs to `Trainer`; this class only measures.

Produces every headline utility number in the project:

    real_mse       probe trained on real data      (the lower bound)
    transfer_mse   probe trained on synthetic data (what distillation costs)
    hybrid MSE     probe trained on a mixture

MSE is summed with reduction='sum' and divided by element count, so the value
does not depend on batch size or on the last batch being short.

RNG caution: iterating the internal DataLoader consumes one torch RNG draw.
Calling this between a seed and a later random step shifts everything after it —
which is why the demo scripts evaluate the real baseline AFTER distillation
rather than before.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ts_distill.evaluation.base import BaseEvaluator


class Evaluator(BaseEvaluator):
    """
    Evaluates a trained model on real held-out test data.

    Responsibilities:
      - Split each window into (input, target) using seq_len.
      - Run the model in eval mode and compute MSE / RMSE.

    Training is NOT this class's responsibility — use Trainer for that.
    """

    def __init__(self, seq_len, batch_size=64):
        """
        Args:
            seq_len (int):    Input length; the rest of each window is the target.
            batch_size (int): Inference batch size. Does not affect the result —
                              the loss is summed and divided by element count.
        """
        self.seq_len    = seq_len
        self.batch_size = batch_size

    def test_on_real(self, model, real_test_data):
        """
        Args:
            model          : Trained nn.Module.
            real_test_data : Tensor (N, window_size, features).

        Returns:
            dict: {'MSE': float, 'RMSE': float}
        """
        device = real_test_data.device
        model  = model.to(device)
        model.eval()

        inputs  = real_test_data[:, :self.seq_len, :]
        targets = real_test_data[:, self.seq_len:, :]

        dataset   = TensorDataset(inputs, targets)
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        criterion = nn.MSELoss(reduction='sum')

        total_loss = 0.0
        n_elements = 0

        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                output  = model(X_batch)
                total_loss += criterion(output, Y_batch).item()
                n_elements += Y_batch.numel()

        avg_mse = total_loss / n_elements
        return {'MSE': avg_mse, 'RMSE': avg_mse ** 0.5}
