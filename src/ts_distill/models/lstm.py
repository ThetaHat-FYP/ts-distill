import torch
import torch.nn as nn
import torch.nn.functional as F
from ts_distill.models.base import BaseForecaster

class LSTM(BaseForecaster):
    def __init__(self, input_dim=1, embed_dim=24, hidden_dim=24, end_dim=128, num_layer=1, dropout=0.2, horizon=24):
        super(LSTM, self).__init__()
        self.horizon = horizon
        self.start_conv = nn.Conv2d(in_channels=input_dim, 
                                    out_channels=embed_dim, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True, dropout=dropout)
        
        self.end_linear1 = nn.Linear(hidden_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, horizon)


    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        if history_data.dim() == 4:
            # [B, L, N, C] -> [B*N, L, C]
            b, l, n, c = history_data.shape
            x = history_data.permute(0, 2, 1, 3).reshape(b * n, l, c)
        elif history_data.dim() == 3:
            # [B, L, C]
            b, l, c = history_data.shape
            n = 1
            x = history_data
        else:
            raise ValueError("Expected 3D or 4D history_data tensor")

        # Project channel dimension to embed_dim using a 1x1 conv
        x_proj = x.permute(0, 2, 1).unsqueeze(2)
        x_proj = self.start_conv(x_proj).squeeze(2)
        x_in = x_proj.permute(0, 2, 1)

        out, _ = self.lstm(x_in)

        x = F.relu(self.end_linear1(out[:, -1, :]))
        x = self.end_linear2(x)
        x = x.unsqueeze(-1)

        if history_data.dim() == 4:
            x = x.reshape(b, n, self.horizon, 1).permute(0, 2, 1, 3)

        return x