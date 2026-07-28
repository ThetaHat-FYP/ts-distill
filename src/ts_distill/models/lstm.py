"""
LSTM forecaster — recurrent baseline. Available, but excluded from experiments.

Encodes the history with an LSTM, takes the final hidden state, and projects it
to all channels x all future steps in one shot.

Why it is not in the reference matrix: MTT backpropagates through an unrolled
inner training loop, and stacking that on top of the LSTM's own recurrence
makes the meta-gradient vanish. The synthetic data barely moves, so distillation
produces almost nothing. That is a property of second-order trajectory matching
meeting a recurrent model, not a bug here — hence MLP, CNN, and DLinear only.
"""

import torch
import torch.nn as nn
from ts_distill.models.base import BaseForecaster


class LSTM(BaseForecaster):
    """
    LSTM encoder + linear decoder; forecasts all horizons in one shot.

    Args:
        input_dim (int):  Number of channels.
        hidden_dim (int): LSTM hidden size.
        num_layer (int):  Stacked LSTM layers.
        seq_len (int):    Look-back window length.
        pred_len (int):   Forecast horizon length.
    """

    def __init__(self, input_dim=7, hidden_dim=64, num_layer=1, seq_len=96, pred_len=96):
        super(LSTM, self).__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            batch_first=True,
        )
        # Projects last hidden state → all channels × all future steps
        self.linear = nn.Linear(hidden_dim, input_dim * pred_len)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Map (B, seq_len, C) history to a (B, pred_len, C) forecast."""
        # history_data: [B, seq_len, C]
        B = history_data.shape[0]
        out, _ = self.lstm(history_data)   # [B, seq_len, hidden_dim]
        out = out[:, -1, :]                # [B, hidden_dim]  (last step)
        out = self.linear(out)             # [B, C * pred_len]
        return out.reshape(B, self.pred_len, self.input_dim)  # [B, pred_len, C]
