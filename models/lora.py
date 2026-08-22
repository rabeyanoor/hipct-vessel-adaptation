# models/lora.py
"""
Plain LoRA (Low-Rank Adaptation) for SAM-Med3D's frozen image_encoder attention
layers -- the "PEFT without spacing awareness" baseline requested to isolate
whether spacing conditioning (PhysicsAwareAdapter) contributes anything beyond
what generic low-rank adaptation already provides.

Targets `qkv` and `proj` (the two nn.Linear layers in each 3D attention block,
see external/SAM-Med3D/segment_anything/modeling/image_encoder3D.py's
Attention class) -- the standard LoRA injection points for ViT-style attention.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank residual: y = Wx + b + (alpha/r) * B(A(x))."""

    def __init__(self, base_linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # LoRA starts as a no-op (B=0)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


def inject_lora(module, target_names=("qkv", "proj"), r=4, alpha=1.0):
    """
    Recursively replaces named nn.Linear children matching target_names with
    LoRALinear wrappers, in place. Returns the number of layers wrapped.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, target_names=target_names, r=r, alpha=alpha)
    return count
