# losses/combined_loss.py
import torch
import torch.nn as nn

from utils.load_cldice import load_soft_cldice

try:
    soft_cldice = load_soft_cldice()
    HAS_CLDICE = True
except Exception:
    HAS_CLDICE = False

class HiPVesselLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.05, gamma=1.0, max_pos_weight=50.0, use_cldice=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_pos_weight = max_pos_weight
        # use_cldice=False turns the term off for the ablation
        self.use_cldice = use_cldice and HAS_CLDICE
        if self.use_cldice:
            self.cldice_fn = soft_cldice()

    def forward(self, pred, target, spacing, logits=None):
        # 1. Soft Dice Loss
        intersection = torch.sum(pred * target)
        dice_loss = 1.0 - (2. * intersection + 1e-5) / (torch.sum(pred) + torch.sum(target) + 1e-5)

        # 2. Topological Continuity Loss (clDice)
        cldice_l = self.cldice_fn(pred, target) if self.use_cldice else torch.tensor(0.0, device=pred.device)

        # 3. Anisotropic Penalty based on Voxel Spacing
        if isinstance(spacing, torch.Tensor) and spacing.dim() == 2:
            sp = spacing.mean(dim=0)
        else:
            sp = spacing

        dz, dy, dx = torch.gradient(pred, dim=(-3, -2, -1))
        aniso_penalty = torch.mean((dz / sp[0])**2 + (dy / sp[1])**2 + (dx / sp[2])**2)

        total_loss = dice_loss + (self.alpha * cldice_l if self.use_cldice else 0.0) + (self.beta * aniso_penalty)

        # 4. Weighted BCE on logits. Still gives gradients when the sigmoid is
        # saturated (Dice/clDice don't). Foreground is only ~1-5% of voxels, so
        # pos_weight comes from the foreground ratio of the batch.
        if logits is not None:
            with torch.no_grad():
                num_pos = target.sum()
                num_total = target.numel()
                pos_weight = ((num_total - num_pos) / num_pos.clamp(min=1.0)).clamp(max=self.max_pos_weight)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight
            )
            total_loss = total_loss + self.gamma * bce_loss

        return total_loss
