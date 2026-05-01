import torch
import torch.nn as nn


class CNN(nn.Module):
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
        # x: [B, seq_len, C]
        B, L, C = x.shape

        # Reshape to apply conv per-channel
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)  # [B*C, 1, L]
        x = self.convs(x).squeeze(1)                   # [B*C, L]
        x = self.linear(x)                              # [B*C, pred_len]

        return x.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]
