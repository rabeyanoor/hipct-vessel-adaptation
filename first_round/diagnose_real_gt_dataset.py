# diagnose_real_gt_dataset.py
"""
Sanity check for the SenNet+HOA kidney dataset before training: image/label/
spacing load, shapes match, masks are binary, foreground looks reasonable.
Prints text only.
"""

import numpy as np
from torch.utils.data import DataLoader

from utils.real_gt_dataset import RealGTKidneyDataset


def diagnose():
    print("=" * 65)
    print("[*] Real-GT Kidney Dataset Diagnostics (SenNet+HOA)")
    print("=" * 65)

    dataset = RealGTKidneyDataset(patch_size=(64, 64, 64), samples_per_epoch=3)
    print(f"[✓] Dataset initialized with {len(dataset)} samples "
          f"across: {[s['sample_id'] for s in dataset.samples]}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(loader):
        image = batch["image"][0, 0].numpy()
        mask = batch["mask"][0, 0].numpy()
        spacing = batch["spacing"][0].numpy()
        sample_id = batch["sample_id"][0]

        assert image.shape == mask.shape, f"Shape mismatch: image {image.shape} vs mask {mask.shape}"
        assert set(np.unique(mask).tolist()) <= {0.0, 1.0}, f"Mask not binary: {np.unique(mask)}"

        fg_pixels = int(np.sum(mask > 0))
        total_pixels = mask.size
        fg_percent = (fg_pixels / total_pixels) * 100.0

        print(f"\n--- Sample [{idx+1}/{len(loader)}] - {sample_id} ---")
        print(f"  Voxel spacing (μm)  : {spacing.tolist()}")
        print(f"  Image shape/stats   : {image.shape}, min={image.min():.4f} max={image.max():.4f} "
              f"mean={image.mean():.4f} std={image.std():.4f}")
        print(f"  Mask shape          : {mask.shape}")
        print(f"  Foreground voxels   : {fg_pixels}/{total_pixels} ({fg_percent:.3f}%)")
        print(f"  Mask unique values  : {np.unique(mask).tolist()}")

    print("\n" + "=" * 65)
    print("[✓] Real-GT dataset diagnostics complete!")
    print("=" * 65)


if __name__ == "__main__":
    diagnose()
