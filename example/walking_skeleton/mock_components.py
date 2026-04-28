"""
Mock Components for MTT Walking Skeleton
Implements minimal versions of base classes for testing pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple
from copy import deepcopy
import random

# Import base classes from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder
from src.ts_distill.evaluation.base import BaseEvaluator
from src.ts_distill.models.base import BaseForecaster
from src.ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from src.ts_distill.distillation_core.initializer.base import BaseInitializer
from src.ts_distill.trainer.trainer.base import BaseTrainer
from src.ts_distill.trainer.callback.base import BaseCallback


# ==================== SIMPLE MODEL ====================
class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series"""
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class StatelessDLinear(nn.Module):
    """Decomposition-Linear Model (Stateless for MTT)"""
    def __init__(self, seq_len, pred_len, channels=1):
        super(StatelessDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Decomp
        self.decomp = SeriesDecomp(kernel_size=25)
        
        # Linear layers for prediction
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Tin, Channels]
        seasonal_init, trend_init = self.decomp(x)
        
        # Apply linear layers across the sequence dimension
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1) # Output: [Batch, Tout, Channels]


class SimpleLSTM(BaseForecaster):
    """
    A custom LSTM wrapper that explicitly zeros out hidden states 
    during every forward pass to ensure strict functional compatibility.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Explicitly initialize hidden states to zero on the correct device
        # This breaks the graph connection to previous steps!
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1) # [batch, 1, 1] to match target slicing


# ==================== TRAJECTORY RECORDER ====================
class SimpleRecorder(BaseTrajectoryRecorder, BaseCallback):
    """
    In-memory trajectory storage with callback support.
    Saves model weights to a Python list instead of disk.
    """
    
    def __init__(self, record_every=1):
        self.trajectory = []
        self.record_every = record_every
        
    def on_train_begin(self, model, **kwargs):
        """Called when training starts."""
        self.trajectory = []
        # Record the exact initialization state
        self.record_checkpoint(model, 0)
        
    def on_epoch_end(self, model, epoch, loss, **kwargs):
        """Called after each epoch - record checkpoints."""
        if (epoch + 1) % self.record_every == 0:
            self.record_checkpoint(model, epoch + 1)
        
    def on_train_end(self, model, **kwargs):
        """Called when training ends."""
        pass
        
    def record_checkpoint(self, model, step: int):
        """Save current model weights to memory."""
        # Deep copy to prevent reference issues
        weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.trajectory.append({
            'step': step,
            'weights': weights
        })
        
    def save_buffer(self, file_path: str):
        """Save trajectory to disk (not needed for walking skeleton)."""
        torch.save(self.trajectory, file_path)
        print(f"Trajectory saved to {file_path}")
        
    def load_buffer(self, file_path: str):
        """Load trajectory from disk."""
        self.trajectory = torch.load(file_path)
        print(f"Trajectory loaded from {file_path}")
        
    def get_trajectory(self):
        """Return the current trajectory list."""
        return self.trajectory
    
    def sample_checkpoint_pair(self, step_gap: int = 10) -> Tuple[Dict, Dict]:
        """Sample a pair of checkpoints (θ_t, θ_{t+k}) for MTT.
        
        Args:
            step_gap: The number of recorded steps between the start and target checkpoint.
            
        Returns:
            Tuple of (start_checkpoint, end_checkpoint)
        """
        if len(self.trajectory) < 2:
            raise ValueError("Need at least 2 checkpoints for MTT trajectory matching!")
            
        # If the gap is larger than our trajectory, cap it safely
        actual_gap = min(step_gap, len(self.trajectory) - 1)
        
        # Sample start index, ensuring we have room for the gap
        start_idx = random.randint(0, len(self.trajectory) - actual_gap - 1)
        end_idx = start_idx + actual_gap
        
        return self.trajectory[start_idx], self.trajectory[end_idx]
    
    def sample_checkpoint(self):
        """Randomly sample one checkpoint from trajectory."""
        if not self.trajectory:
            raise ValueError("No trajectory recorded yet!")
        return random.choice(self.trajectory)


# ==================== EVALUATOR ====================
class SimpleEvaluator(BaseEvaluator):
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


# ==================== TRAJECTORY MATCHER ====================
class MSEMatcher(BaseTrajectoryMatcher):
    """
    Calculates MSE between student and expert parameters.
    This MUST be differentiable for meta-learning!
    """
    
    def calculate_loss(self, student_params, expert_params):
        """
        student_params: List of tensors
        expert_params: List of tensors
        Returns: Differentiable scalar loss
        """
        loss = 0.0
        for s_param, e_param in zip(student_params, expert_params):
            # Ensure expert params don't have gradients
            e_param = e_param.detach()
            loss += torch.sum((s_param - e_param) ** 2)
        
        return loss


# ==================== INITIALIZER ====================
class RealSampleInitializer(BaseInitializer):
    """
    Initialize synthetic data by sampling from real data.
    """
    
    def initialize(self, shape, real_data_reference=None):
        """
        shape: Tuple (n_samples, seq_len, n_features)
        real_data_reference: Tensor to sample from
        """
        if real_data_reference is None:
            # Fallback to random
            return torch.randn(shape) * 0.1
        
        # Randomly sample from real data
        n_samples = shape[0]
        indices = torch.randint(0, len(real_data_reference), (n_samples,))
        data = real_data_reference[indices].clone()
        data.requires_grad = True
        return data


# ==================== TRAINER ====================
class SimpleTrainer(BaseTrainer):
    """
    Simple trainer with callback support.
    """
    
    def train_epoch(self, dataloader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            seq_len = batch.shape[1] // 2
            inputs  = batch[:, :seq_len, :]
            targets = batch[:, seq_len:, :]
            
            # Predict using only past data
            output = self.model(inputs)
            
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def fit(self, dataloader, epochs: int, callbacks=None):
        """Main training loop with callbacks."""
        if callbacks is None:
            callbacks = []
        
        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin(self.model)
        
        # Training loop
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            
            # Call on_epoch_end
            for callback in callbacks:
                callback.on_epoch_end(self.model, epoch, avg_loss)
        
        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end(self.model)


# ==================== CALLBACK ====================
class SimpleCallback(BaseCallback):
    """
    Simple callback that prints training progress.
    """
    
    def on_train_begin(self, model, **kwargs):
        """Called when training starts."""
        print("Training started...")
    
    def on_epoch_end(self, model, epoch, loss, **kwargs):
        """Called after each epoch."""
        if (epoch + 1) % 4 == 0:
            print(f"  Epoch {epoch+1} | Loss: {loss:.6f}")
    
    def on_train_end(self, model, **kwargs):
        """Called when training ends."""
        print("Training completed.")
