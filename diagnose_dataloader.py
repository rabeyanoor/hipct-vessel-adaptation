# diagnose_dataloader.py
"""
Quick check of the multi-organ HiP-CT data loading: shapes, spacing, intensity
stats and pseudo-label foreground. Prints text only.
"""

import numpy as np
from torch.utils.data import DataLoader
from config import ORGAN_CONFIG, TRAIN_CONFIG
from utils.dataset import MultiOrganHiPCTDataset

def diagnose():
    print("=" * 65)
    print("[*] HiP-CT Data Pipeline & Dataloader Diagnostics")
    print("=" * 65)

    dataset = MultiOrganHiPCTDataset(config=ORGAN_CONFIG, patch_size=TRAIN_CONFIG["patch_size"], samples_per_epoch=6)
    print(f"[✓] Dataset initialized successfully with {len(dataset)} samples across organs: {list(ORGAN_CONFIG.keys())}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(loader):
        image = batch["image"][0, 0].numpy()      # Shape: (D, H, W)
        target_mask = batch["mask"][0, 0].numpy() # Shape: (D, H, W), Frangi pseudo-label
        spacing = batch["spacing"][0].numpy()     # Shape: (3,)
        organ = batch["organ"][0]

        fg_pixels = np.sum(target_mask > 0)
        total_pixels = target_mask.size
        fg_percent = (fg_pixels / total_pixels) * 100.0

        print(f"\n--- Sample [{idx+1}/{len(loader)}] - Organ: {organ.upper()} ---")
        print(f"  Voxel Spacing (μm) : {spacing.tolist()}")
        print(f"  Image Shape        : {image.shape}")
        print(f"  Image Intensity    : Min={image.min():.4f}, Max={image.max():.4f}, Mean={image.mean():.4f}, Std={image.std():.4f}")
        print(f"  Target Mask Shape  : {target_mask.shape}")
        print(f"  Foreground Voxels  : {fg_pixels}/{total_pixels} ({fg_percent:.2f}%)")
        print(f"  Target Unique Values: {np.unique(target_mask).tolist()}")

    print("\n" + "=" * 65)
    print("[✓] Diagnostics completed!")
    print("=" * 65)

if __name__ == "__main__":
    diagnose()
