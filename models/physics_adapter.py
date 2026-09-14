# models/physics_adapter.py
import torch
import torch.nn as nn

class PhysicsAwareAdapter(nn.Module):
    def __init__(self, embed_dim=384, reduction=4):
        super().__init__()
        # Voxel Spacing Conditioning Vector
        self.spacing_fc = nn.Sequential(
            nn.Linear(3, embed_dim // reduction),
            nn.GELU(),
            nn.Linear(embed_dim // reduction, embed_dim)
        )

        # Low-Rank Adapter Path
        self.down_proj = nn.Linear(embed_dim, embed_dim // reduction)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(embed_dim // reduction, embed_dim)

    def forward(self, feature_map, spacing_tensor):
        """
        feature_map: (Batch, Channels, D, H, W) or (Batch, Tokens, Embed_dim)
        spacing_tensor: Voxel Spacing (Batch, 3) e.g., [26.04, 26.04, 26.04]
        """
        if feature_map.dim() == 5:
            B, C, D, H, W = feature_map.shape
            spacing_emb = self.spacing_fc(spacing_tensor).view(B, C, 1, 1, 1)
            conditioned_features = feature_map * spacing_emb

            # Low-rank adaptation
            permuted = conditioned_features.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
            adapter_out = self.up_proj(self.act(self.down_proj(permuted))).permute(0, 4, 1, 2, 3)
            return feature_map + adapter_out
        else:
            B, N, C = feature_map.shape
            spacing_emb = self.spacing_fc(spacing_tensor).unsqueeze(1)  # (B, 1, C)
            conditioned_features = feature_map * spacing_emb
            adapter_out = self.up_proj(self.act(self.down_proj(conditioned_features)))
            return feature_map + adapter_out
