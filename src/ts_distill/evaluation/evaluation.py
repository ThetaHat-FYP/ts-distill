import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ts_distill.evaluation.base import BaseEvaluator


class Evaluator(BaseEvaluator):
    """
    Tests synthetic data quality by:
    1. Training a fresh model on synthetic data
    2. Testing on real data
    """
    
    def __init__(self, model_factory, n_epochs=250, lr=1e-4, momentum=0.9, batch_size=64):
        """
        model_factory: Function that returns a new model instance
        """
        self.model_factory = model_factory
        self.n_epochs = n_epochs
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

    def train_on_synthetic(self, synthetic_data, model=None):
        """
        Train a fresh model on synthetic data using mini-batch SGD.
        synthetic_data: Tensor of shape (N, Seq, Features)
        """
        if model is None:
            model = self.model_factory()

        device = synthetic_data.device
        model = model.to(device)
        model.train()

        seq_len = synthetic_data.shape[1] // 2
        inputs  = synthetic_data[:, :seq_len, :].detach()
        targets = synthetic_data[:, seq_len:, :].detach()

        dataset = TensorDataset(inputs, targets)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.n_epochs):
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss   = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()

        return model

    def test_on_real(self, model, real_test_data):
        """
        Evaluate model on real data using mini-batches.
        real_test_data: Tensor of shape (N, Seq, Features)
        """
        device = real_test_data.device
        model  = model.to(device)
        model.eval()
        criterion = nn.MSELoss(reduction='sum')

        seq_len = real_test_data.shape[1] // 2
        inputs  = real_test_data[:, :seq_len, :]
        targets = real_test_data[:, seq_len:, :]

        dataset = TensorDataset(inputs, targets)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss, n = 0.0, 0
        with torch.no_grad():
            for X_batch, Y_batch in loader:
                output = model(X_batch)
                total_loss += criterion(output, Y_batch).item()
                n += Y_batch.numel()

        avg_loss = total_loss / n
        return {
            'MSE': avg_loss,
            'RMSE': avg_loss ** 0.5
        }