import torch
import torch.nn as nn
from ts_distill.models.base import BaseForecaster


class MLP(BaseForecaster):
    def __init__(self, seq_len=24, pred_len=24):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Sequential(nn.Linear(self.seq_len, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, self.pred_len))

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        assert history_data.shape[-1] == 1
        # Flatten channel dimension and predict next sequence
        x = history_data[..., 0]
        x = self.Linear(x)
        return x.unsqueeze(-1)