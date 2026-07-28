"""
DLinear forecaster — decomposition + linear (Zeng et al., 2022).

Splits the input into TREND (moving average) and SEASONAL (the residual),
forecasts each with its own linear layer, and adds the results. Despite having
no non-linearity it is a strong benchmark forecaster, which is the point of the
original paper.

Two properties make it the best-behaved model in this library:

* The explicit trend/seasonal split means it depends directly on the temporal
  structure that distillation damages — so it is the most informative
  architecture for the fidelity question.
* Its validation curve is smooth and flattens hard, so phase-boundary detection
  is reliable (its detector config is the strictest of the three).

Weights initialise to 1/seq_len, matching the reference implementation, so the
model starts as a moving average rather than at random.
"""

import torch
import torch.nn as nn
from ts_distill.models.base import BaseForecaster


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """Smooth (B, L, C) with a moving average, preserving length L."""
        # Asymmetric padding matches the reference implementation:
        # front pads (kernel_size-1)//2, end pads kernel_size//2.
        # For odd kernels both equal (kernel_size-1)//2; for even kernels the
        # extra sample goes to the end so the output length equals the input length.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Decompose a time series into seasonal residual and trend components."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """Split x into (seasonal residual, trend), both shaped like x."""
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(BaseForecaster):
    """
    DLinear — Decomposition-Linear forecaster (Zeng et al., 2022).

    individual=False : one shared linear layer across all channels.
    individual=True  : separate linear layers per channel.

    Both modes initialise weights to 1/seq_len, matching the reference
    implementation (LTSF-Linear / the attached Colab notebook).
    """

    def __init__(self, seq_len, pred_len, channels, individual=False):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.individual = individual
        self.channels   = channels

        self.decomposition = series_decomp(kernel_size=25)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend    = nn.ModuleList()
            for _ in range(self.channels):
                ls = nn.Linear(self.seq_len, self.pred_len)
                lt = nn.Linear(self.seq_len, self.pred_len)
                ls.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
                lt.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
                self.Linear_Seasonal.append(ls)
                self.Linear_Trend.append(lt)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend    = nn.Linear(self.seq_len, self.pred_len)
            # Initialise weights to 1/seq_len, same as the individual mode
            # and the reference Colab notebook.
            self.Linear_Seasonal.weight.data.fill_(1 / self.seq_len)
            self.Linear_Trend.weight.data.fill_(1 / self.seq_len)

    def forward(self, x):
        """Forecast each component separately, then sum: (B, pred_len, C)."""
        # x: (Batch, seq_len, Channels)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)   # (B, C, seq_len)
        trend_init    = trend_init.permute(0, 2, 1)

        if self.individual:
            B, C, _ = seasonal_init.shape
            seasonal_output = torch.zeros(B, C, self.pred_len,
                                          dtype=seasonal_init.dtype,
                                          device=seasonal_init.device)
            trend_output    = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :]    = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output    = self.Linear_Trend(trend_init)

        return (seasonal_output + trend_output).permute(0, 2, 1)  # (B, pred_len, C)

    def extract_features(self, x):
        """Concatenated seasonal+trend features for FRePO kernel regression."""
        seasonal_init, trend_init = self.decomposition(x)
        return torch.cat(
            [seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)], dim=-1
        )
