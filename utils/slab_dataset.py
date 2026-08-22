# utils/slab_dataset.py
"""
Faster patch sampler for training on the raw TIFF stacks.

RealGTKidneyDataset reads every slice of a patch from disk for each patch. For
128^3 patches that is ~460 MB of TIFF decoding per patch on kidney_1_dense,
far too slow for thousands of steps. Here each worker reads one slab of
consecutive slices into memory and cuts many patches from it before moving on
to a new slab, which cuts the I/O by about patches_per_slab times.

The patch options are the same as RealGTKidneyDataset's sampled mode
(fg_prob, fov_scale_range) and the returned dict has the same keys.

z_frac limits which slices of a volume are used, as fractions of its depth,
e.g. {"kidney_1_dense": (0.0, 0.8)} keeps the last 20 % free for validation
(see VAL_HOLDOUT in train_lora_v2.py).
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from utils.real_gt_dataset import RealGTKidneyDataset


class SlabPatchDataset(IterableDataset):
    def __init__(self, sample_ids, raw_tiff_root=None, patch_size=128, fg_prob=0.5,
                 fov_scale_range=None, patches_per_slab=32, slab_depth=None, z_frac=None):
        self.base = RealGTKidneyDataset(sample_ids=sample_ids, raw_tiff_root=raw_tiff_root,
                                        patch_size=(patch_size,) * 3, samples_per_epoch=1)
        if not self.base.samples:
            raise RuntimeError("no training volumes found")
        self.patch = patch_size
        self.fg_prob = fg_prob
        self.fov_scale_range = fov_scale_range
        self.patches_per_slab = patches_per_slab
        self.z_frac = z_frac or {}
        hi = fov_scale_range[1] if fov_scale_range else 1.0
        self.slab_depth = slab_depth or int(math.ceil(patch_size * hi)) + 32
        # pick volumes in proportion to their size
        sizes = np.array([np.prod(s["shape"]) for s in self.base.samples], dtype=float)
        self.weights = sizes / sizes.sum()

    def _read_slab(self, sample, z0, depth):
        # keep the native dtypes (uint16 image, uint8 label); float32 slabs of
        # kidney_3_sparse would need ~6 GB per worker
        if sample["backend"] == "memmap":
            return (np.asarray(sample["img_mmap"][z0:z0 + depth]),
                    np.asarray(sample["lbl_mmap"][z0:z0 + depth]) > 0)
        import tifffile
        names = sample["label_names"][z0:z0 + depth]
        img = np.stack([tifffile.imread(os.path.join(sample["image_dir"], n)) for n in names])
        lbl = np.stack([tifffile.imread(os.path.join(sample["label_dir"], n)) > 0 for n in names])
        return img, lbl

    def _cut(self, img, lbl, native_um, fg_coords):
        depth, h, w = img.shape
        if self.fov_scale_range:
            lo, hi = self.fov_scale_range
            s = math.exp(random.uniform(math.log(lo), math.log(hi)))
        else:
            s = 1.0
        side = int(round(self.patch * s))
        side = max(8, min(side, depth, h, w))
        if len(fg_coords) and random.random() < self.fg_prob:
            c = fg_coords[random.randrange(len(fg_coords))]
            start = [int(c[a]) - side // 2 + random.randint(-side // 4, side // 4) for a in range(3)]
        else:
            start = [random.randint(0, img.shape[a] - side) for a in range(3)]
        start = [min(max(0, start[a]), img.shape[a] - side) for a in range(3)]
        sl = tuple(slice(start[a], start[a] + side) for a in range(3))
        v = torch.from_numpy(np.ascontiguousarray(img[sl], dtype=np.float32))[None, None]
        m = torch.from_numpy(np.ascontiguousarray(lbl[sl], dtype=np.float32))[None, None]
        if side != self.patch:
            v = F.interpolate(v, size=(self.patch,) * 3, mode="trilinear", align_corners=False)
            m = (F.interpolate(m, size=(self.patch,) * 3, mode="trilinear", align_corners=False) > 0.5).float()
        v, m = v[0], m[0]
        std = v.std()
        v = (v - v.mean()) / std if std > 1e-8 else v - v.mean()
        eff = native_um * side / self.patch
        return v, m, eff

    def __iter__(self):
        info = get_worker_info()
        seed = (torch.initial_seed() + (info.id if info else 0)) % 2 ** 32
        random.seed(seed)
        rng = np.random.default_rng(seed)
        while True:
            sample = self.base.samples[rng.choice(len(self.base.samples), p=self.weights)]
            d = sample["shape"][0]
            lo, hi = self.z_frac.get(sample["sample_id"], (0.0, 1.0))
            z_lo, z_hi = int(lo * d), int(hi * d)
            depth = min(self.slab_depth, z_hi - z_lo)
            z0 = random.randint(z_lo, z_hi - depth)
            img, lbl = self._read_slab(sample, z0, depth)
            fg_coords = np.argwhere(lbl[::2, ::2, ::2] > 0) * 2
            native_um = float(sample["spacing"][0])
            for _ in range(self.patches_per_slab):
                v, m, eff = self._cut(img, lbl, native_um, fg_coords)
                yield {"image": v, "mask": m,
                       "spacing": torch.tensor([eff] * 3, dtype=torch.float32),
                       "organ": "kidney", "sample_id": sample["sample_id"]}
