# evaluate_fov_confound.py
"""
Tests whether the OOD collapse on kidney_1_voi (section 5.3/3.3 of README.md
and test.sh) is actually a spacing-conditioning failure, or just an ordinary
domain shift caused by field of view: a 64^3 patch at kidney_1_voi's native
5.2um spacing covers ~333um per side, versus ~3200um for the same patch
shape at the 50um training spacing this project's models never see anything
smaller than.

Approach: extract a much larger voxel cube from kidney_1_voi at its native
5.2um resolution -- sized so its PHYSICAL extent matches a 64^3 patch at
~50um (i.e. side length in voxels roughly 50/5.2 = ~9.6x larger, unless
overridden) -- then downsample that cube to 64^3 with trilinear
interpolation for the image and nearest-neighbor for the label. The
downsampled patch has the same physical field of view as an in-distribution
training patch, but its content still comes entirely from real 5.2um data
(no synthetic upsampling of a smaller true patch).

If Dice recovers on these FOV-matched patches relative to the native-scale
OOD eval, the collapse is a field-of-view/never-seen-visual-scale effect,
not something specific to conditioning on the numeric spacing value. If it
stays collapsed, FOV is not the (or not the whole) explanation.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import TRAIN_CONFIG
from evaluate_real_gt import evaluate_model_on_loader
from models.plain_lora_model import PlainLoRAModel
from models.spacing_lora_model import SpacingConditionedLoRAModel
from utils.real_gt_dataset import RealGTKidneyDataset, _SAMPLE_INFO


class FOVMatchedKidneyDataset(RealGTKidneyDataset):
    """Same sample bookkeeping as RealGTKidneyDataset, but __getitem__ reads
    a large native-resolution cube and downsamples it to `patch_size` so its
    physical field of view matches `target_fov_um` instead of matching
    `patch_size` voxel-for-voxel at native spacing."""

    def __init__(self, sample_ids, patch_size=(64, 64, 64), samples_per_epoch=10,
                 raw_tiff_root=None, target_fov_um=None):
        super().__init__(sample_ids=sample_ids, patch_size=patch_size,
                          samples_per_epoch=samples_per_epoch, raw_tiff_root=raw_tiff_root)
        # Default target FOV: this project's training patch size at the
        # in-distribution 50um spacing (see TRAIN_CONFIG / config.py).
        self.target_fov_um = target_fov_um or (patch_size[0] * 50.0)

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        native_spacing_um = sample["spacing"][0].item()
        full_d, full_h, full_w = sample["shape"]
        pd = ph = pw = self.patch_size[0]

        large_side = max(pd, min(
            int(round(self.target_fov_um / native_spacing_um)),
            full_d, full_h, full_w,
        ))

        grid_d = max(1, full_d // large_side)
        grid_h = max(1, full_h // large_side)
        grid_w = max(1, full_w // large_side)
        import random
        gz = random.randint(0, grid_d - 1)
        gy = random.randint(0, grid_h - 1)
        gx = random.randint(0, grid_w - 1)
        start_d, start_h, start_w = gz * large_side, gy * large_side, gx * large_side

        if sample["backend"] == "memmap":
            vol, mask = self._read_patch_memmap(sample, start_d, start_h, start_w,
                                                 large_side, large_side, large_side)
        else:
            vol, mask = self._read_patch_raw_tiff(sample, start_d, start_h, start_w,
                                                    large_side, large_side, large_side)

        vol_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(vol_t, size=self.patch_size, mode="trilinear", align_corners=False)
        mask_t = F.interpolate(mask_t, size=self.patch_size, mode="nearest")
        vol_t, mask_t = vol_t.squeeze(0), mask_t.squeeze(0)

        std_val = vol_t.std()
        vol_t = (vol_t - vol_t.mean()) / std_val if std_val > 1e-8 else vol_t - vol_t.mean()

        effective_spacing_um = native_spacing_um * large_side / self.patch_size[0]
        return {
            "image": vol_t,
            "mask": mask_t,
            "spacing": torch.tensor([effective_spacing_um] * 3, dtype=torch.float32),
            "organ": "kidney",
            "sample_id": sample["sample_id"],
        }


def evaluate_fov_confound(checkpoint_path, model_class, label,
                           sample_id="kidney_1_voi", samples_per_sample=40,
                           raw_tiff_root=None, target_fov_um=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = TRAIN_CONFIG["patch_size"]

    model = model_class().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[✓] Loaded {checkpoint_path}")

    native_spacing_um = _SAMPLE_INFO[sample_id]["spacing"]
    effective_fov = target_fov_um or (patch_size[0] * 50.0)
    print(f"[*] {sample_id} native spacing: {native_spacing_um}um -- "
          f"extracting large cubes and downsampling to {patch_size} so "
          f"physical FOV matches a {patch_size[0]}^3 patch at ~{effective_fov / patch_size[0]:.1f}um")

    dataset = FOVMatchedKidneyDataset(sample_ids=[sample_id], patch_size=patch_size,
                                       samples_per_epoch=samples_per_sample,
                                       raw_tiff_root=raw_tiff_root, target_fov_um=target_fov_um)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return evaluate_model_on_loader(model, device, loader, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["plain_lora", "spacing_lora"], required=True)
    parser.add_argument("--sample-id", default="kidney_1_voi")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--raw-tiff-root", default=None)
    args = parser.parse_args()

    model_cls = PlainLoRAModel if args.model == "plain_lora" else SpacingConditionedLoRAModel
    evaluate_fov_confound(
        checkpoint_path=args.checkpoint, model_class=model_cls,
        label=f"FOV-matched OOD eval ({args.model}) on {args.sample_id}",
        sample_id=args.sample_id, samples_per_sample=args.samples,
        raw_tiff_root=args.raw_tiff_root,
    )
