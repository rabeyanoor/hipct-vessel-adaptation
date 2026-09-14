# utils/real_gt_dataset.py
"""
Dataset for the real expert-annotated SenNet+HOA kidney samples (see
README.md section 2.1 and data/kidney_metadata_verified.csv). This is a
SEPARATE, configurable dataset mode from MultiOrganHiPCTDataset -- it does not
touch or replace the existing Frangi-pseudo-labeled Heart/Lung/Kidney
pipeline.

Two backends:
  - memmap (default): requires utils/convert_kidney_gt_to_memmap.py to have
    been run first. Fast random access, but needs disk space to materialize
    every sample's full image+label arrays (~23GB for kidney_1_dense +
    kidney_1_voi alone) -- fine on a local machine, but exceeds constrained
    environments (e.g. Kaggle kernels' ~20GB /kaggle/working quota, which is
    exactly what failed when this was first run there).
  - raw_tiff_root: reads individual TIFF slices directly from a folder tree
    shaped like the competition's train/<sample>/{images,labels}/*.tif (this
    is exactly how Kaggle mounts the competition data, read-only, and doesn't
    count against the working-directory quota since nothing is copied).
    Slower per-patch (real disk reads every __getitem__ call, no caching),
    but disk-safe on quota-constrained environments.
"""

import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.kidney_metadata_verification import VERIFIED_SAMPLES

MEMMAP_DIR = "data/memmap_kidney_gt"

# sample_id -> (spacing_um, split, n_label_slices), from the verified metadata table
_SAMPLE_INFO = {s[0]: {"spacing": s[2], "split": s[5], "n_label_slices": s[4]} for s in VERIFIED_SAMPLES}

IMAGE_SOURCE_OVERRIDE = {"kidney_3_dense": "kidney_3_sparse"}


class RealGTKidneyDataset(Dataset):
    def __init__(self, sample_ids=None, patch_size=(64, 64, 64), samples_per_epoch=6,
                 raw_tiff_root=None):
        self.patch_size = list(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.raw_tiff_root = raw_tiff_root

        available = [sid for sid, info in _SAMPLE_INFO.items()
                     if info["split"] == "train" and info["n_label_slices"] > 0]
        if sample_ids is None:
            sample_ids = available
        else:
            unknown = [sid for sid in sample_ids if sid not in available]
            if unknown:
                raise ValueError(f"No real GT available for: {unknown} (check "
                                  f"data/kidney_metadata_verified.csv)")

        self.samples = []
        for sid in sample_ids:
            if raw_tiff_root is not None:
                sample = self._init_raw_tiff_sample(sid, raw_tiff_root)
            else:
                sample = self._init_memmap_sample(sid)
            if sample is not None:
                self.samples.append(sample)

    def _init_memmap_sample(self, sid):
        meta_path = os.path.join(MEMMAP_DIR, f"{sid}_meta.json")
        img_path = os.path.join(MEMMAP_DIR, f"{sid}_image.dat")
        lbl_path = os.path.join(MEMMAP_DIR, f"{sid}_label.dat")
        if not (os.path.exists(meta_path) and os.path.exists(img_path) and os.path.exists(lbl_path)):
            print(f"[WARNING] Real-GT kidney memmap not found for '{sid}'; "
                  f"run utils/convert_kidney_gt_to_memmap.py first.")
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        img_mmap = np.memmap(img_path, dtype=meta["image_dtype"], mode="r", shape=shape)
        lbl_mmap = np.memmap(lbl_path, dtype=meta["label_dtype"], mode="r", shape=shape)
        return {
            "sample_id": sid, "backend": "memmap",
            "img_mmap": img_mmap, "lbl_mmap": lbl_mmap, "shape": shape,
            "spacing": torch.tensor([_SAMPLE_INFO[sid]["spacing"]] * 3, dtype=torch.float32),
        }

    def _init_raw_tiff_sample(self, sid, raw_tiff_root):
        import tifffile
        image_source = IMAGE_SOURCE_OVERRIDE.get(sid, sid)
        label_dir = os.path.join(raw_tiff_root, sid, "labels")
        image_dir = os.path.join(raw_tiff_root, image_source, "images")
        if not os.path.isdir(label_dir):
            print(f"[WARNING] Raw TIFF label dir not found for '{sid}': {label_dir}")
            return None
        label_names = sorted(os.listdir(label_dir))
        first = tifffile.imread(os.path.join(label_dir, label_names[0]))
        h, w = first.shape
        d = len(label_names)
        return {
            "sample_id": sid, "backend": "raw_tiff",
            "label_dir": label_dir, "image_dir": image_dir, "label_names": label_names,
            "shape": (d, h, w),
            "spacing": torch.tensor([_SAMPLE_INFO[sid]["spacing"]] * 3, dtype=torch.float32),
        }

    def __len__(self):
        if not self.samples:
            return 0
        return len(self.samples) * self.samples_per_epoch

    def _read_patch_memmap(self, sample, start_d, start_h, start_w, pd, ph, pw):
        vol = np.array(
            sample["img_mmap"][start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw],
            dtype=np.float32,
        )
        mask = np.array(
            sample["lbl_mmap"][start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw],
            dtype=np.float32,
        )
        return vol, mask

    def _read_patch_raw_tiff(self, sample, start_d, start_h, start_w, pd, ph, pw):
        import tifffile
        d_total = len(sample["label_names"])
        vol_slices, mask_slices = [], []
        for z in range(start_d, min(start_d + pd, d_total)):
            name = sample["label_names"][z]
            img_slice = tifffile.imread(os.path.join(sample["image_dir"], name))
            lbl_slice = tifffile.imread(os.path.join(sample["label_dir"], name))
            vol_slices.append(img_slice[start_h:start_h + ph, start_w:start_w + pw])
            mask_slices.append((lbl_slice[start_h:start_h + ph, start_w:start_w + pw] > 0).astype(np.float32))
        vol = np.stack(vol_slices, axis=0).astype(np.float32)
        mask = np.stack(mask_slices, axis=0).astype(np.float32)
        return vol, mask

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        full_d, full_h, full_w = sample["shape"]
        pw, ph, pd = self.patch_size

        grid_w = max(1, full_w // pw)
        grid_h = max(1, full_h // ph)
        grid_d = max(1, full_d // pd)
        gx = random.randint(0, grid_w - 1)
        gy = random.randint(0, grid_h - 1)
        gz = random.randint(0, grid_d - 1)
        start_w, start_h, start_d = gx * pw, gy * ph, gz * pd

        if sample["backend"] == "memmap":
            vol, mask = self._read_patch_memmap(sample, start_d, start_h, start_w, pd, ph, pw)
        else:
            vol, mask = self._read_patch_raw_tiff(sample, start_d, start_h, start_w, pd, ph, pw)

        if vol.shape != tuple(self.patch_size):
            pad_d = max(0, pd - vol.shape[0])
            pad_h = max(0, ph - vol.shape[1])
            pad_w = max(0, pw - vol.shape[2])
            pad_spec = ((0, pad_d), (0, pad_h), (0, pad_w))
            vol = np.pad(vol, pad_spec, mode="constant")
            mask = np.pad(mask, pad_spec, mode="constant")

        std_val = np.std(vol)
        if std_val > 1e-8:
            vol = (vol - np.mean(vol)) / std_val
        else:
            vol = vol - np.mean(vol)

        return {
            "image": torch.from_numpy(vol).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),   # REAL expert ground truth, not pseudo-label
            "spacing": sample["spacing"],
            "organ": "kidney",
            "sample_id": sample["sample_id"],
        }
