"""
CNN forecaster — local pattern extractor, and the least stable model to distil.

Two 1-D convolutions followed by a linear projection, applied channel-wise:
every channel is reshaped into its own sample so one shared filter bank learns
local shapes without mixing channels.

Stability warning. This is the architecture that diverges during distillation.
Convolutions amplify the large synthetic learning rate that MTT needs, so the
probe MSE occasionally comes back NaN — callers should filter non-finite
results rather than assume every run produces a number. Its phase-boundary
detector settings are also the most heavily damped of the three (see
`PHASE_BOUNDARY_CONFIGS` in `ts_distill.config`) for the same reason: a noisy
validation curve.

Note this class does NOT subclass BaseForecaster, unlike the other models, but
it honours the same input/output shape contract.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Channel-independent 1-D CNN: two conv layers, then a linear projection.

    Args:
        channel (int):  Number of input channels (kept for shape bookkeeping).
        seq_len (int):  Look-back window length.
        pred_len (int): Forecast horizon length.
    """

    def __init__(self, channel, seq_len=96, pred_len=96):
        super(CNN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel = channel

        # Channel-independent 1-D convolution: [B*C, 1, seq_len] → [B*C, 1, seq_len]
        self.convs = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )

        # Maps seq_len features → pred_len for each channel
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        """Map (B, seq_len, C) history to a (B, pred_len, C) forecast."""
        # x: [B, seq_len, C]
        B, L, C = x.shape

        # Reshape to apply conv per-channel
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)  # [B*C, 1, L]
        x = self.convs(x).squeeze(1)                   # [B*C, L]
        x = self.linear(x)                              # [B*C, pred_len]

        return x.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]
