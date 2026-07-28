"""
Forecaster interface — the shape contract every model in this library obeys.

One signature, enforced across all architectures:

    (batch, seq_len, channels)  ->  (batch, pred_len, channels)

That uniformity is what makes cross-architecture evaluation possible: any model
can be the expert whose trajectory is distilled, and any other can be the probe
trained on the result, with no adapter code between them.
"""

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseForecaster(nn.Module, ABC):
    """Interface for any model (LSTM, Transformer, TCN)."""
    
    @abstractmethod
    def forward(self, x):
        """Standard forward pass."""
        pass