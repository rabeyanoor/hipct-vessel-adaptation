# models/collapse_head.py
import torch
import torch.nn as nn

class CollapseDetectionHead(nn.Module):
    """
    Sub-head for vascular collapse prediction in 3D HiP-CT volumes.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
