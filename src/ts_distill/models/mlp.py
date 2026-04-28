import torch
import torch.nn as nn
from ts_distill.models.base import BaseForecaster


class MLP(BaseForecaster):
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
        # history_data: [B, seq_len, C]
        # Channel-independent: apply the same linear to every channel
        x = history_data.permute(0, 2, 1)  # [B, C, seq_len]
        x = self.linear(x)                  # [B, C, pred_len]
        return x.permute(0, 2, 1)           # [B, pred_len, C]
