# models/spacing_lora.py
"""
Spacing-Conditioned LoRA: LoRALinear (models/lora.py) plus FiLM modulation of
the LoRA branch from the voxel spacing:
`h' = base(x) + scaling * (gamma(spacing) * B(A(x)) + beta(spacing))`.

Same injection points as Plain LoRA (attention qkv/proj), but the update depends
on the spacing, so a 50um and a 5.2um scan can be treated differently.

SAM-Med3D's Attention.forward() calls `self.qkv(x)` with one argument, so there
is no way to pass spacing in. Instead of editing the external repo, the model
sets `current_spacing` on every injected layer right before the encoder call
(see SpacingConditionedLoRAModel.forward).
"""

import math

import torch
import torch.nn as nn


REF_SPACING_UM = 50.0


def encode_spacing(spacing, mode):
    """FiLM input. "raw": spacing in um, as in the first experiments; the FiLM
    output then scales almost linearly with the spacing value (at 5.2um the
    modulation was ~0.10x of that at 50um). "log": log2(spacing / 50um), so the
    training spacing maps to 0 and each halving/doubling is one unit."""
    if mode == "raw":
        return spacing
    if mode == "log":
        return torch.log2(spacing.clamp_min(1e-3) / REF_SPACING_UM)
    raise ValueError(mode)


class SpacingConditionedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=4, alpha=1.0, spacing_hidden=16, spacing_encoding="raw"):
        super().__init__()
        self.spacing_encoding = spacing_encoding
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
        nn.init.zeros_(self.lora_B.weight)  # LoRA branch starts as a no-op

        # Small fixed hidden size. The input is just 3 numbers; scaling the
        # hidden size with out_features added ~31M params for qkv (2304-dim).
        self.spacing_film = nn.Sequential(
            nn.Linear(3, spacing_hidden), nn.GELU(), nn.Linear(spacing_hidden, out_features * 2)
        )
        # zero init -> gamma=1, beta=0 at the start (identity)
        nn.init.zeros_(self.spacing_film[-1].weight)
        nn.init.zeros_(self.spacing_film[-1].bias)

        self.current_spacing = None  # set by the model before each forward

    def forward(self, x):
        if self.current_spacing is None:
            raise RuntimeError(
                "SpacingConditionedLoRALinear.current_spacing was not set before forward(). "
                "Use SpacingConditionedLoRAModel, which broadcasts spacing to every injected "
                "layer before calling the encoder."
            )
        lora_out = self.lora_B(self.lora_A(x))

        film = self.spacing_film(encode_spacing(self.current_spacing, self.spacing_encoding))  # (B, 2*out)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = 1.0 + gamma
        while gamma.dim() < lora_out.dim():  # broadcast over token/spatial dims
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        modulated = gamma * lora_out + beta
        return self.base(x) + self.scaling * modulated


def inject_spacing_lora(module, target_names=("qkv", "proj"), r=4, alpha=1.0, spacing_encoding="raw"):
    """
    Recursively replaces named nn.Linear children matching target_names with
    SpacingConditionedLoRALinear wrappers, in place. Returns the list of
    injected layers (needed to broadcast spacing to them before each forward).
    """
    injected = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            new_layer = SpacingConditionedLoRALinear(child, r=r, alpha=alpha, spacing_encoding=spacing_encoding)
            setattr(module, name, new_layer)
            injected.append(new_layer)
        else:
            injected.extend(inject_spacing_lora(child, target_names=target_names, r=r, alpha=alpha,
                                                spacing_encoding=spacing_encoding))
    return injected
