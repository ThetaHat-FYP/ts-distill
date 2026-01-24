"""
Mock Components for MTT Walking Skeleton
Implements minimal versions of base classes for testing pipeline.
"""

import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict
from copy import deepcopy

# Import base classes from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ts_distill.data_pipeline.data_loader.base import BaseDataLoader
from src.ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder
from src.ts_distill.evaluation.base import BaseEvaluator
from src.ts_distill.models.base import BaseForecaster
from src.ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from src.ts_distill.distillation_core.initializer.base import BaseInitializer


# ==================== DATA LOADER ====================
class MockDataLoader(BaseDataLoader):
    """
    Generates random time-series data in memory.
    No CSV needed - perfect for testing.
    """
    
    def __init__(self, n_samples=100, seq_len=10, n_features=3, batch_size=32):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size
        self.data = None
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Generate random data (ignores file_path)."""
        # Shape: (n_samples, seq_len, n_features)
        self.data = torch.randn(self.n_samples, self.seq_len, self.n_features)
        
        # Return a dummy DataFrame for compatibility
        return pd.DataFrame({"shape": [self.data.shape]})
    
    def view_data(self, n_rows: int = 5) -> None:
        """Print data summary."""
        print(f"Mock Data Shape: {self.data.shape}")
        print(f"Data Range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        print(f"First sample:\n{self.data[0, :3, :]}")
    
    def get_batches(self):
        """Return data as list of batches."""
        batches = []
        for i in range(0, self.n_samples, self.batch_size):
            batch = self.data[i:i+self.batch_size]
            batches.append(batch)
        return batches


# ==================== SIMPLE MODEL ====================
class SimpleLSTM(BaseForecaster):
    """
    Minimal LSTM for testing.
    Input: (Batch, Seq, Features) -> Output: (Batch, 1)
    """
    
    def __init__(self, input_size=3, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        x: (Batch, Seq, Features)
        Returns: (Batch, 1, 1) to match target shape
        """
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.unsqueeze(1)  # (Batch, 1) -> (Batch, 1, 1)


# ==================== TRAJECTORY RECORDER ====================
class SimpleRecorder(BaseTrajectoryRecorder):
    """
    In-memory trajectory storage.
    Saves model weights to a Python list instead of disk.
    """
    
    def __init__(self):
        self.trajectory = []
        
    def on_train_begin(self, model, **kwargs):
        """Called when training starts."""
        self.trajectory = []
        
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
            
            # Forward pass (predict next value)
            output = model(synthetic_data)
            
            # Create targets for windowed data
            if synthetic_data.shape[1] == 1:
                targets = synthetic_data[:, -1:, :]
            else:
                targets = synthetic_data[:, -1, :].unsqueeze(1)
            
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
                output = model(batch)
                # Match target shape for windowed data
                if batch.shape[1] == 1:
                    targets = batch[:, -1:, :]
                else:
                    targets = batch[:, -1, :].unsqueeze(1)
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
class RandomInitializer(BaseInitializer):
    """
    Initialize synthetic data with random noise.
    """
    
    def initialize(self, shape, real_data_reference=None):
        """
        shape: Tuple (n_samples, seq_len, n_features)
        Returns: Tensor with requires_grad=True
        """
        # Small random values
        data = torch.randn(shape) * 0.1
        data.requires_grad = True
        return data


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
