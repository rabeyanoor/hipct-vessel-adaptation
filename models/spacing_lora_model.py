# models/spacing_lora_model.py
"""
Spacing-Conditioned LoRA model -- this project's main proposed method: same
architecture as PlainLoRAModel (frozen SAM-Med3D backbone, LoRA in attention
qkv/proj, trainable mask_decoder), except the LoRA branches are additionally
modulated by real physical voxel spacing (see models/spacing_lora.py).

Direct comparison point for PlainLoRAModel (no spacing conditioning) and
HiPVesselAdaptModel (spacing-conditioned, but via a single post-encoder
adapter rather than in-encoder LoRA) -- same frozen backbone, same
trainable-parameter order of magnitude, same real-GT data and training
protocol (see train_plain_lora.py / train_real_gt.py for the others).
"""

import os
import sys

import torch
import torch.nn as nn

sam_med3d_path = os.path.abspath("./external/SAM-Med3D")
if os.path.exists(sam_med3d_path) and sam_med3d_path not in sys.path:
    sys.path.append(sam_med3d_path)

try:
    from segment_anything.build_sam3D import sam_model_registry3D
    HAS_SAM_MED3D = True
except ImportError:
    HAS_SAM_MED3D = False

from models.spacing_lora import inject_spacing_lora

DEFAULT_SAM_MED3D_CHECKPOINT = os.path.abspath("./external/SAM-Med3D/ckpt/sam_med3d.pth")


class SpacingConditionedLoRAModel(nn.Module):
    def __init__(self, checkpoint_path=DEFAULT_SAM_MED3D_CHECKPOINT, model_type="vit_b_ori",
                 lora_r=4, lora_alpha=1.0):
        super().__init__()
        if not HAS_SAM_MED3D:
            raise RuntimeError("SAM-Med3D not importable -- check external/SAM-Med3D setup.")

        self.sam_model = sam_model_registry3D[model_type](checkpoint=None)

        if checkpoint_path and os.path.exists(checkpoint_path):
            raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
            missing, unexpected = self.sam_model.load_state_dict(state_dict, strict=False)
            print(f"[*] Loaded pretrained SAM-Med3D weights from {checkpoint_path} "
                  f"({len(missing)} missing, {len(unexpected)} unexpected keys)")

        for param in self.sam_model.image_encoder.parameters():
            param.requires_grad = False
        self.lora_layers = inject_spacing_lora(self.sam_model.image_encoder, r=lora_r, alpha=lora_alpha)
        print(f"[*] Injected Spacing-Conditioned LoRA (r={lora_r}, alpha={lora_alpha}) "
              f"into {len(self.lora_layers)} attention layers")

        for param in self.sam_model.mask_decoder.parameters():
            param.requires_grad = True

    def forward(self, x, spacing):
        """spacing: (B, 3) real physical voxel spacing -- REQUIRED (unlike
        PlainLoRAModel, where it's accepted but unused)."""
        orig_spatial = x.shape[2:]

        for layer in self.lora_layers:
            layer.current_spacing = spacing
        image_embeddings = self.sam_model.image_encoder(x)
        for layer in self.lora_layers:
            layer.current_spacing = None  # avoid holding a stale reference between calls

        batch_size = x.shape[0]
        ref_size = self.sam_model.prompt_encoder.input_image_size
        corner_points = torch.tensor(
            [[0, 0, 0], [ref_size[0] - 1, ref_size[1] - 1, ref_size[2] - 1]],
            dtype=torch.float32, device=x.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        corner_labels = torch.ones((batch_size, 2), dtype=torch.float32, device=x.device)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=(corner_points, corner_labels), boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        if low_res_masks.shape[2:] != orig_spatial:
            low_res_masks = torch.nn.functional.interpolate(
                low_res_masks, size=orig_spatial, mode="trilinear", align_corners=False
            )
        logits = low_res_masks
        mask_pred = torch.sigmoid(logits)
        return mask_pred, logits
