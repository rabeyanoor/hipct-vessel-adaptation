# models/refine_head.py
"""
Full-resolution refinement head for the SAM-Med3D based models.

SAM-Med3D uses 16^3 patches and its mask decoder only upsamples 4x, so a 64^3
input gives a 16^3 mask (a 128^3 input gives 32^3). Kidney vessels are often
1-3 voxels wide, which that grid cannot represent. This head takes the
trilinearly upsampled coarse logits together with the input image and predicts
a residual correction at full resolution, similar in spirit to the extra
up-sampling layers in MA-SAM and the detail-recovery stream in DINO-Med3D.

The last conv is zero-initialised, so at the start of training the output is
exactly the coarse logits.
"""

import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class RefineHead(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.enc1 = _block(2, channels)
        self.down = nn.Conv3d(channels, channels * 2, kernel_size=2, stride=2)
        self.enc2 = _block(channels * 2, channels * 2)
        self.up = nn.ConvTranspose3d(channels * 2, channels, kernel_size=2, stride=2)
        self.dec = _block(channels * 2, channels)
        self.out = nn.Conv3d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image, coarse_logits):
        x1 = self.enc1(torch.cat([image, coarse_logits], dim=1))
        x2 = self.enc2(self.down(x1))
        x = self.dec(torch.cat([self.up(x2), x1], dim=1))
        return coarse_logits + self.out(x)
