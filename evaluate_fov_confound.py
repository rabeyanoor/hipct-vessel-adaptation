# evaluate_fov_confound.py
"""
Checks whether the OOD collapse on kidney_1_voi (README 5.3 / test.sh 3.3) is
about the spacing conditioning or just the field of view: a 64^3 patch at
5.2um is ~333um wide, while at the 50um training spacing it is ~3200um.

Take a bigger cube from kidney_1_voi at 5.2um, sized to the same physical
extent as a 64^3 patch at ~50um (about 50/5.2 = ~9.6x larger per side), and
downsample it to 64^3 (trilinear for the image, nearest for the label). Same
field of view as training, but still real 5.2um data.

If Dice comes back on these patches, the collapse is mostly a field-of-view
effect. If not, FOV is not (or not the whole) explanation.
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
    """Like RealGTKidneyDataset, but reads a large cube and downsamples it to
    `patch_size` so the physical field of view is `target_fov_um`."""

    def __init__(self, sample_ids, patch_size=(64, 64, 64), samples_per_epoch=10,
                 raw_tiff_root=None, target_fov_um=None):
        super().__init__(sample_ids=sample_ids, patch_size=patch_size,
                          samples_per_epoch=samples_per_epoch, raw_tiff_root=raw_tiff_root)
        # default: training patch size at 50um
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
