import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, channel):
        super(CNN, self).__init__()
        
        self.start_conv = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=8*channel, kernel_size=16, stride=4),
                                        nn.Dropout(p=0.1),
                                        nn.ReLU())
        
        self.fc = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 24)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            # [B, L, C] -> [B, C, L]
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            # [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)
        else:
            raise ValueError("Expected 2D or 3D input tensor")

        B, C, L = x.shape
        x_tmp = self.start_conv(x)
        x = x_tmp.reshape(B, C, -1) + x
        
        x = self.fc(x)
        return x.permute(0, 2, 1)