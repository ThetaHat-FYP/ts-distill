"""
Mock Components — Backward-Compatibility Shim
=============================================
The production implementations of SimpleRecorder, MSEMatcher,
RealSampleInitializer, and SimpleCallback now live in the ts_distill
source package.  This file re-exports them from there so that any
existing code that imports from mock_components continues to work
without modification.

Example-only models (StatelessDLinear, SimpleLSTM) are kept here
because they are demonstration code, not part of the core framework.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make sure the project root is on sys.path so that ts_distill is importable.
sys.path.append(str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Re-export the four core components from their canonical src locations.
# Import these from ts_distill directly in new code — not from here.
# ---------------------------------------------------------------------------
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder          # noqa: F401
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher                  # noqa: F401
from ts_distill.distillation_core.initializer.random_sample_initializer import (    # noqa: F401
    RealSampleInitializer,
)
from ts_distill.trainer.callback.simple_callback import SimpleCallback             # noqa: F401

# Also re-export base classes that example-specific code may need.
from ts_distill.models.base import BaseForecaster                                  # noqa: F401
from ts_distill.trainer.callback.base import BaseCallback                          # noqa: F401


# ---------------------------------------------------------------------------
# Example-only models — kept here as demonstration code only.
# These are NOT used by run_cycle.py; they exist for experimentation.
# ---------------------------------------------------------------------------

class MovingAvg(nn.Module):
    """Moving-average block used by StatelessDLinear below."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Trend/seasonal decomposition used by StatelessDLinear below."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class StatelessDLinear(nn.Module):
    """
    A simplified DLinear kept here for quick experimentation.
    The canonical implementation is in ts_distill/models/dlinear.py.
    """

    def __init__(self, seq_len, pred_len, channels=1):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.decomp   = SeriesDecomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend    = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init    = trend_init.permute(0, 2, 1)
        out = self.Linear_Seasonal(seasonal_init) + self.Linear_Trend(trend_init)
        return out.permute(0, 2, 1)


class SimpleLSTM(BaseForecaster):
    """
    LSTM wrapper that explicitly zeros hidden states each forward pass
    to ensure compatibility with torch.func.functional_call in MTT.
    Kept here as a demonstration; use ts_distill/models/lstm.py for production.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0  = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0  = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).unsqueeze(1)
