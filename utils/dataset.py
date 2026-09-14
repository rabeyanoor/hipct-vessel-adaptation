# utils/dataset.py
import sys
import os
import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np

from utils.pseudo_labels import generate_vessel_pseudolabel
from utils.convert_to_memmap import MEMMAP_DIR

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))

class MultiOrganHiPCTDataset(Dataset):
    def __init__(self, config, patch_size=(128, 128, 128), samples_per_epoch=30,
                 pseudo_cache_dir="data/pseudo_cache"):
        self.patch_size = list(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.pseudo_cache_dir = pseudo_cache_dir
        os.makedirs(self.pseudo_cache_dir, exist_ok=True)

        self.organ_data = []
        for organ, meta in config.items():
            data_path = os.path.join(MEMMAP_DIR, f"{organ}.dat")
            meta_path = os.path.join(MEMMAP_DIR, f"{organ}_meta.json")
            if os.path.exists(data_path) and os.path.exists(meta_path):
                with open(meta_path) as f:
                    mm_meta = json.load(f)
                full_size = tuple(mm_meta["shape"])  # (D, H, W), numpy order
                mmap = np.memmap(data_path, dtype=mm_meta["dtype"], mode="r", shape=full_size)

                self.organ_data.append({
                    "organ": organ,
                    "mmap": mmap,
                    "full_size": full_size,
                    "spacing": torch.tensor(meta["spacing"], dtype=torch.float32),
                    "black_ridges": meta.get("black_ridges", False),
                })
            else:
                print(f"[WARNING] Memmap volume not found for organ '{organ}' "
                      f"(expected {data_path}); run utils/convert_to_memmap.py first.")

    def __len__(self):
        if not self.organ_data:
            return 0
        return len(self.organ_data) * self.samples_per_epoch

    def __getitem__(self, idx):
        sample = self.organ_data[idx % len(self.organ_data)]
        full_d, full_h, full_w = sample["full_size"]
        pw, ph, pd = self.patch_size

        # Snap patches to a fixed non-overlapping grid (instead of arbitrary pixel
        # offsets) so that repeated visits to the same block -- across epochs, or
        # across train/eval -- hit the pseudo-label disk cache below instead of
        # re-running the ~10-30s/patch Frangi filter every time.
        grid_w = max(1, full_w // pw)
        grid_h = max(1, full_h // ph)
        grid_d = max(1, full_d // pd)
        gx = random.randint(0, grid_w - 1)
        gy = random.randint(0, grid_h - 1)
        gz = random.randint(0, grid_d - 1)
        start_w, start_h, start_d = gx * pw, gy * ph, gz * pd

        mmap = sample["mmap"]
        vol = np.array(
            mmap[start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw],
            dtype=np.float32,
        )  # numpy shape: (D, H, W)

        # Pad if volume shape is smaller than patch_size
        if vol.shape != tuple(self.patch_size):
            pad_d = max(0, pd - vol.shape[0])
            pad_h = max(0, ph - vol.shape[1])
            pad_w = max(0, pw - vol.shape[2])
            vol = np.pad(vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        # Z-score Normalization
        std_val = np.std(vol)
        if std_val > 1e-8:
            vol = (vol - np.mean(vol)) / std_val
        else:
            vol = vol - np.mean(vol)

        vol_tensor = torch.from_numpy(vol).unsqueeze(0)  # Shape: (1, 128, 128, 128)

        # Frangi-vesselness pseudo-label: HiP-CT raw downloads carry no expert
        # vessel annotation, so this heuristic tubular-structure mask stands in
        # as weak supervision (see utils/pseudo_labels.py). Cached per grid block
        # to disk since each computation costs ~10-30s.
        cache_path = os.path.join(
            self.pseudo_cache_dir,
            f"{sample['organ']}_{gx}-{gy}-{gz}_{pw}x{ph}x{pd}.npy",
        )
        if os.path.exists(cache_path):
            pseudo_mask = np.load(cache_path)
        else:
            pseudo_mask = generate_vessel_pseudolabel(vol, black_ridges=sample["black_ridges"])
            np.save(cache_path, pseudo_mask)
        mask_tensor = torch.from_numpy(pseudo_mask).unsqueeze(0)  # Shape: (1, 128, 128, 128)

        return {
            "image": vol_tensor,          # Shape: (1, 128, 128, 128)
            "mask": mask_tensor,          # Shape: (1, 128, 128, 128), pseudo-label
            "spacing": sample["spacing"], # Voxel Spacing (3,)
            "organ": sample["organ"]
        }

# Alias for compatibility
HiPCTDataset = MultiOrganHiPCTDataset
