from abc import ABC, abstractmethod
import torch.nn as nn

class BaseForecaster(nn.Module, ABC):
    """Interface for any model (LSTM, Transformer, TCN)."""

    # @abstractmethod
    # def get_features(self, x):
    #     """
    #     Returns intermediate layer outputs. 
    #     Crucial for 'Distribution Matching' which matches features, not just outputs.
    #     """
    #     pass

    @abstractmethod
    def forward(self, x):
        """Standard forward pass."""
        pass