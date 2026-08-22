# models/hip_vessel_model.py
import os
import torch
import torch.nn as nn

# SAM-Med3D path integration
from utils.load_sam_med3d import import_sam_model_registry3D

sam_model_registry3D = import_sam_model_registry3D()
HAS_SAM_MED3D = sam_model_registry3D is not None

from models.physics_adapter import PhysicsAwareAdapter
from models.collapse_head import CollapseDetectionHead

DEFAULT_SAM_MED3D_CHECKPOINT = os.path.abspath("./external/SAM-Med3D/ckpt/sam_med3d.pth")


class HiPVesselAdaptModel(nn.Module):
    def __init__(self, checkpoint_path=DEFAULT_SAM_MED3D_CHECKPOINT, model_type="vit_b_ori"):
        super().__init__()
        # 1. Base Backbone Model (SAM-Med3D)
        # The released checkpoint is "vit_b_ori" (768-dim, 128^3), not the
        # registry's "vit_b" (384-dim, 256^3) -- shapes won't match with vit_b.
        if HAS_SAM_MED3D:
            # SAM-Med3D's checkpoint= expects a plain state_dict, but the released
            # file is a full training checkpoint, so load it ourselves below.
            if model_type in sam_model_registry3D:
                self.sam_model = sam_model_registry3D[model_type](checkpoint=None)
            else:
                self.sam_model = sam_model_registry3D["vit_b_ori"](checkpoint=None)

            if checkpoint_path and os.path.exists(checkpoint_path):
                raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
                missing, unexpected = self.sam_model.load_state_dict(state_dict, strict=False)
                print(f"[*] Loaded pretrained SAM-Med3D weights from {checkpoint_path} "
                      f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
            elif checkpoint_path:
                print(f"[!] WARNING: SAM-Med3D checkpoint not found at {checkpoint_path} -- "
                      f"backbone is randomly initialized (frozen noise, will not learn).")

            # Freeze heavy ViT image encoder backbone, keep mask decoder & adapter trainable
            for param in self.sam_model.image_encoder.parameters():
                param.requires_grad = False
            for param in self.sam_model.mask_decoder.parameters():
                param.requires_grad = True

            embed_dim = getattr(self.sam_model.image_encoder, "out_chans", 384)
        else:
            self.sam_model = None
            embed_dim = 384

        # 2. Trainable modules
        self.physics_adapter = PhysicsAwareAdapter(embed_dim=embed_dim, reduction=4)
        self.collapse_head = CollapseDetectionHead(in_channels=1)
        
        # Ensure custom modules require gradients
        for param in self.physics_adapter.parameters():
            param.requires_grad = True
        for param in self.collapse_head.parameters():
            param.requires_grad = True

    def forward(self, x, spacing):
        """
        x: Input image volume tensor of shape (B, 1, D, H, W)
        spacing: Voxel spacing tensor of shape (B, 3)
        """
        orig_spatial = x.shape[2:]
        if self.sam_model is not None:
            # 1. Encoder is frozen, no need to keep its graph (saves memory)
            with torch.no_grad():
                image_embeddings = self.sam_model.image_encoder(x)

            # 2. Inject Physical Spacing Conditioning
            adapted_embeddings = self.physics_adapter(image_embeddings, spacing)

            # 3. Prompt embeddings
            # The decoder needs a prompt; without one it ends up predicting
            # foreground everywhere. Use two corner points covering the whole
            # volume (like a full box). _embed_boxes still has a 2D reshape
            # that breaks in 3D, so points are used instead. Coordinates are in
            # the prompt encoder's input_image_size, not the patch size.
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
            
            # 4. Mask Decoder Prediction
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=adapted_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Upsample mask prediction to match original input patch dimensions
            if low_res_masks.shape[2:] != orig_spatial:
                low_res_masks = torch.nn.functional.interpolate(
                    low_res_masks, size=orig_spatial, mode="trilinear", align_corners=False
                )
            logits = low_res_masks
            mask_pred = torch.sigmoid(logits)
        else:
            # Fallback tensor operation if SAM backbone unavailable
            logits = x
            mask_pred = torch.sigmoid(logits)

        # 5. Collapse Prediction Sub-head
        collapse_pred = self.collapse_head(mask_pred)

        return mask_pred, collapse_pred, logits
