"""
MLP forecaster — the simplest baseline, and the reference model for experiments.

Two dense layers applied CHANNEL-INDEPENDENTLY: the input is transposed so the
same small network maps seq_len -> pred_len for every channel separately. That
keeps the parameter count tiny and independent of channel count, which is why
this model distils quickly and is used for most reference runs.

Channel independence also means the model cannot learn relationships BETWEEN
channels — the property `CrossCorrelationMetric` measures. Useful to remember
when interpreting why cross-channel damage in the synthetic data barely affects
this model's accuracy.
"""

import torch
import torch.nn as nn
from ts_distill.models.base import BaseForecaster


class MLP(BaseForecaster):
    """
    Channel-independent two-layer MLP: seq_len -> 64 -> pred_len.

    Args:
        seq_len (int):  Look-back window length.
        pred_len (int): Forecast horizon length.
    """

    def __init__(self, seq_len=96, pred_len=96):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len),
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Map (B, seq_len, C) history to a (B, pred_len, C) forecast."""
        # history_data: [B, seq_len, C]
        # Channel-independent: apply the same linear to every channel
        x = history_data.permute(0, 2, 1)  # [B, C, seq_len]
        x = self.linear(x)                  # [B, C, pred_len]
        return x.permute(0, 2, 1)           # [B, pred_len, C]
