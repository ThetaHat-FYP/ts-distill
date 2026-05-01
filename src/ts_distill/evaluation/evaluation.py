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
        seq_len   : input length; the rest of the window is the forecast target.
        batch_size: inference batch size (does not affect accuracy).
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
