# models/spacing_lora.py
"""
Spacing-Conditioned LoRA: extends models/lora.py's LoRALinear with a FiLM-style
modulation of the LoRA branch, driven by real physical voxel spacing --
`h' = base(x) + scaling * (gamma(spacing) * B(A(x)) + beta(spacing))`.

This is the project's main proposed method: the same low-rank injection
points as Plain LoRA (attention qkv/proj), but the low-rank update itself is
now conditioned on the scan's real spacing, so the model can behave
differently for a 50um scan vs. a 5.2um scan of the same tissue.

Implementation note: nn.Linear has no way to receive a second forward
argument when called as `self.qkv(x)` deep inside SAM-Med3D's unmodified
Attention.forward() (see external/SAM-Med3D/segment_anything/modeling/
image_encoder3D.py) -- rather than editing that external repo, spacing is
passed via a plain (non-parameter, non-buffer) attribute set on every
injected layer immediately before each forward pass, by the owning model
(see SpacingConditionedLoRAModel.forward in models/spacing_lora_model.py).
"""

import math

import torch
import torch.nn as nn


class SpacingConditionedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=4, alpha=1.0, spacing_hidden=16):
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
        nn.init.zeros_(self.lora_B.weight)  # LoRA branch starts as a no-op

        # Fixed hidden size independent of out_features: the input is only 3
        # numbers (spacing), so parameter count should not scale with the
        # attention layer's width. A width-scaled hidden layer would make this
        # network's params grow as out_features^2, which for qkv (2304-dim)
        # would dominate the total trainable budget and defeat the point of a
        # parameter-efficient comparison against plain LoRA.
        self.spacing_film = nn.Sequential(
            nn.Linear(3, spacing_hidden), nn.GELU(), nn.Linear(spacing_hidden, out_features * 2)
        )
        # Zero-init the last layer so gamma starts at 1 and beta at 0 -- i.e.
        # spacing conditioning starts as an identity modulation (matching
        # LoRALinear's own zero-init no-op start), not a random perturbation.
        nn.init.zeros_(self.spacing_film[-1].weight)
        nn.init.zeros_(self.spacing_film[-1].bias)

        self.current_spacing = None  # set externally each forward pass, see module docstring

    def forward(self, x):
        if self.current_spacing is None:
            raise RuntimeError(
                "SpacingConditionedLoRALinear.current_spacing was not set before forward(). "
                "Use SpacingConditionedLoRAModel, which broadcasts spacing to every injected "
                "layer before calling the encoder."
            )
        lora_out = self.lora_B(self.lora_A(x))

        film = self.spacing_film(self.current_spacing)  # (B, 2*out_features)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = 1.0 + gamma
        while gamma.dim() < lora_out.dim():  # broadcast over token/spatial dims
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        modulated = gamma * lora_out + beta
        return self.base(x) + self.scaling * modulated


def inject_spacing_lora(module, target_names=("qkv", "proj"), r=4, alpha=1.0):
    """
    Recursively replaces named nn.Linear children matching target_names with
    SpacingConditionedLoRALinear wrappers, in place. Returns the list of
    injected layers (needed to broadcast spacing to them before each forward).
    """
    injected = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            new_layer = SpacingConditionedLoRALinear(child, r=r, alpha=alpha)
            setattr(module, name, new_layer)
            injected.append(new_layer)
        else:
            injected.extend(inject_spacing_lora(child, target_names=target_names, r=r, alpha=alpha))
    return injected
