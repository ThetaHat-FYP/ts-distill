from abc import ABC, abstractmethod
import torch.nn as nn

class BaseForecaster(nn.Module, ABC):
    """Interface for any model (LSTM, Transformer, TCN)."""
    
    @abstractmethod
    def forward(self, x):
        """Standard forward pass."""
        pass