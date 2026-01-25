"""
Mock Components for MTT Walking Skeleton
Implements minimal versions of base classes for testing pipeline.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from copy import deepcopy

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
class SimpleLSTM(BaseForecaster):
    """
    Minimal LSTM for time series forecasting.
    Input: (Batch, Seq, Features) where Seq can be variable length.
    Output: (Batch, 1, 1) - prediction for next timestep.
    """
    
    def __init__(self, input_size=3, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, Seq, Features) - past observations
        Returns:
            (Batch, 1, 1) - prediction for next timestep
        """
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.unsqueeze(1)  # (Batch, 1) -> (Batch, 1, 1)


# ==================== TRAJECTORY RECORDER ====================
class SimpleRecorder(BaseTrajectoryRecorder, BaseCallback):
    """
    In-memory trajectory storage with callback support.
    Saves model weights to a Python list instead of disk.
    """
    
    def __init__(self, record_every=2):
        self.trajectory = []
        self.record_every = record_every
        
    def on_train_begin(self, model, **kwargs):
        """Called when training starts."""
        self.trajectory = []
        
    def on_epoch_end(self, model, epoch, loss, **kwargs):
        """Called after each epoch - record checkpoints."""
        if (epoch + 1) % self.record_every == 0:
            self.record_checkpoint(model, epoch)
        
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
    
    def sample_checkpoint(self):
        """Randomly sample one checkpoint from trajectory."""
        if not self.trajectory:
            raise ValueError("No trajectory recorded yet!")
        import random
        return random.choice(self.trajectory)


# ==================== EVALUATOR ====================
class SimpleEvaluator(BaseEvaluator):
    """
    Tests synthetic data quality by:
    1. Training a fresh model on synthetic data
    2. Testing on real data
    """
    
    def __init__(self, model_factory, n_epochs=5, lr=0.001):
        """
        model_factory: Function that returns a new model instance
        """
        self.model_factory = model_factory
        self.n_epochs = n_epochs
        self.lr = lr
        
    def train_on_synthetic(self, synthetic_data, model=None):
        """
        Train a fresh model on synthetic data.
        synthetic_data: Tensor of shape (N, Seq, Features)
        """
        if model is None:
            model = self.model_factory()
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Simple training loop
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            
            # Split into inputs (past) and targets (future)
            # Input: all timesteps except the last one
            # Target: only the last timestep
            inputs = synthetic_data[:, :-1, :]
            targets = synthetic_data[:, -1:, :]
            
            # Predict using only past data
            output = model(inputs)
            
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
        return model
    
    def test_on_real(self, model, real_test_loader):
        """
        Evaluate model on real data.
        real_test_loader: MockDataLoader with test data
        """
        model.eval()
        criterion = nn.MSELoss()
        
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in real_test_loader.get_batches():
                # Split into inputs (past) and targets (future)
                # Input: all timesteps except the last one
                # Target: only the last timestep
                inputs = batch[:, :-1, :]
                targets = batch[:, -1:, :]
                
                # Predict using only past data
                output = model(inputs)
                
                loss = criterion(output, targets)
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
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
            
            # Split into inputs (past) and targets (future)
            # Input: all timesteps except the last one
            # Target: only the last timestep
            inputs = batch[:, :-1, :]
            targets = batch[:, -1:, :]
            
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
