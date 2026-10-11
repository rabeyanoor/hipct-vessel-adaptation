# utils/real_gt_dataset.py
"""
Dataset for the SenNet+HOA kidney samples with expert labels (README section
2.1, data/kidney_metadata_verified.csv). Separate from MultiOrganHiPCTDataset,
which is for the pseudo-labeled organs.

Two backends:
  - memmap (default): needs utils/convert_kidney_gt_to_memmap.py first. Fast,
    but uses a lot of disk (~23GB just for kidney_1_dense + kidney_1_voi), which
    is more than the ~20GB /kaggle/working quota on Kaggle.
  - raw_tiff_root: reads TIFF slices directly from a
    train/<sample>/{images,labels}/*.tif folder (how Kaggle mounts the
    competition data, read-only, not counted in the quota). Slower, since it
    reads from disk on every __getitem__, but needs no extra space.
    Can also be a list of roots (e.g. the competition folder plus
    data/zenodo for the test kidneys); each sample uses the first root that
    has it.

Sampling. With the default arguments patches are drawn on a fixed grid, exactly
as in the first experiments. Three optional settings change that:
  - fg_prob: probability of centering the patch on a labelled vessel voxel
    (like nnU-Net's foreground oversampling). Most random 64^3 patches have no
    vessel at all, so without this most training steps are skipped.
  - fov_scale_range=(lo, hi): read a cube of patch*s voxels with s drawn
    log-uniformly from [lo, hi] and resize it to the patch size. The spacing
    returned is native_spacing * s, so the model sees a range of effective
    spacings (spacing augmentation, as in HyperSpace).
  - target_fov_um: fixed version of the above; the cube is sized so the patch
    covers target_fov_um per side (the FOV-matched setting of
    evaluate_fov_confound.py).
"""

import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.kidney_metadata_verification import VERIFIED_SAMPLES

MEMMAP_DIR = "data/memmap_kidney_gt"
FG_INDEX_DIR = "data/fg_index"

# sample_id -> (spacing_um, split, n_label_slices)
_SAMPLE_INFO = {s[0]: {"spacing": s[2], "split": s[5], "n_label_slices": s[4]} for s in VERIFIED_SAMPLES}

IMAGE_SOURCE_OVERRIDE = {"kidney_3_dense": "kidney_3_sparse"}


class RealGTKidneyDataset(Dataset):
    def __init__(self, sample_ids=None, patch_size=(64, 64, 64), samples_per_epoch=6,
                 raw_tiff_root=None, fg_prob=0.0, fov_scale_range=None, target_fov_um=None,
                 fg_index_stride=4):
        self.patch_size = list(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.raw_tiff_root = raw_tiff_root
        self.fg_prob = fg_prob
        self.fov_scale_range = fov_scale_range
        self.target_fov_um = target_fov_um
        self.fg_index_stride = fg_index_stride

        labelled = [sid for sid, info in _SAMPLE_INFO.items() if info["n_label_slices"] > 0]
        if sample_ids is None:
            sample_ids = [sid for sid in labelled if _SAMPLE_INFO[sid]["split"] == "train"]
        else:
            unknown = [sid for sid in sample_ids if sid not in labelled]
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
        roots = [raw_tiff_root] if isinstance(raw_tiff_root, str) else list(raw_tiff_root)
        image_source = IMAGE_SOURCE_OVERRIDE.get(sid, sid)
        label_dir = image_dir = None
        for root in roots:
            if os.path.isdir(os.path.join(root, sid, "labels")):
                label_dir = os.path.join(root, sid, "labels")
                image_dir = os.path.join(root, image_source, "images")
                break
        if label_dir is None:
            print(f"[WARNING] Raw TIFF label dir not found for '{sid}' under {roots}")
            return None
        # some released volumes have empty image files for part of the stack
        # (kidney_additional_voi_13um), so keep only slices with image data
        label_names = [n for n in sorted(os.listdir(label_dir))
                       if os.path.exists(os.path.join(image_dir, n))
                       and os.path.getsize(os.path.join(image_dir, n)) > 0]
        # a missing slice in the middle (e.g. from an interrupted download)
        # would silently shift everything below it by one slice
        idx = [int(os.path.splitext(n)[0]) for n in label_names if os.path.splitext(n)[0].isdigit()]
        if idx and idx[-1] - idx[0] + 1 != len(idx):
            raise RuntimeError(f"'{sid}': slice numbering has gaps ({len(idx)} slices between "
                               f"{idx[0]} and {idx[-1]}); re-download the missing files")
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

    def _read_cube(self, sample, start_d, start_h, start_w, pd, ph, pw):
        if sample["backend"] == "memmap":
            return self._read_patch_memmap(sample, start_d, start_h, start_w, pd, ph, pw)
        return self._read_patch_raw_tiff(sample, start_d, start_h, start_w, pd, ph, pw)

    # ---------------------------------------------------------------- foreground index
    def _fg_coords(self, sample):
        """Coarse list of labelled voxel coordinates (every fg_index_stride-th
        voxel), cached in memory and on disk."""
        if "fg_coords" in sample:
            return sample["fg_coords"]
        s = self.fg_index_stride
        cache = os.path.join(FG_INDEX_DIR, f"{sample['sample_id']}_s{s}.npy")
        if os.path.exists(cache):
            coords = np.load(cache)
        else:
            if sample["backend"] == "memmap":
                sub = np.asarray(sample["lbl_mmap"][::s, ::s, ::s])
            else:
                import tifffile
                names = sample["label_names"][::s]
                sub = np.stack([tifffile.imread(os.path.join(sample["label_dir"], n))[::s, ::s]
                                for n in names], axis=0)
            coords = (np.argwhere(sub > 0) * s).astype(np.int32)
            os.makedirs(FG_INDEX_DIR, exist_ok=True)
            np.save(cache, coords)
        sample["fg_coords"] = coords
        return coords

    # ---------------------------------------------------------------- sampling
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        if self.fg_prob == 0 and self.fov_scale_range is None and self.target_fov_um is None:
            return self._grid_item(sample)
        return self._sampled_item(sample)

    def _grid_item(self, sample):
        full_d, full_h, full_w = sample["shape"]
        pw, ph, pd = self.patch_size

        grid_w = max(1, full_w // pw)
        grid_h = max(1, full_h // ph)
        grid_d = max(1, full_d // pd)
        gx = random.randint(0, grid_w - 1)
        gy = random.randint(0, grid_h - 1)
        gz = random.randint(0, grid_d - 1)
        start_w, start_h, start_d = gx * pw, gy * ph, gz * pd

        vol, mask = self._read_cube(sample, start_d, start_h, start_w, pd, ph, pw)

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
            "mask": torch.from_numpy(mask).unsqueeze(0),   # expert label
            "spacing": sample["spacing"],
            "organ": "kidney",
            "sample_id": sample["sample_id"],
        }

    def _scale_factor(self, native_um):
        if self.target_fov_um is not None:
            return self.target_fov_um / (self.patch_size[0] * native_um)
        if self.fov_scale_range is not None:
            lo, hi = self.fov_scale_range
            return math.exp(random.uniform(math.log(lo), math.log(hi)))
        return 1.0

    def _sampled_item(self, sample):
        full = sample["shape"]
        native_um = float(sample["spacing"][0])
        scale = self._scale_factor(native_um)
        side = int(round(self.patch_size[0] * scale))
        side = max(8, min(side, *full))

        if self.fg_prob > 0 and random.random() < self.fg_prob and len(self._fg_coords(sample)):
            coords = self._fg_coords(sample)
            c = coords[random.randrange(len(coords))]
            # jitter so the vessel is not always in the exact center
            start = [int(c[a]) - side // 2 + random.randint(-side // 4, side // 4) for a in range(3)]
        else:
            start = [random.randint(0, full[a] - side) for a in range(3)]
        start = [min(max(0, start[a]), full[a] - side) for a in range(3)]

        vol, mask = self._read_cube(sample, start[0], start[1], start[2], side, side, side)
        vol_t = torch.from_numpy(vol).float()[None, None]
        mask_t = torch.from_numpy(mask).float()[None, None]
        if side != self.patch_size[0]:
            vol_t = F.interpolate(vol_t, size=self.patch_size, mode="trilinear", align_corners=False)
            mask_t = (F.interpolate(mask_t, size=self.patch_size, mode="trilinear",
                                    align_corners=False) > 0.5).float()
        vol_t, mask_t = vol_t[0], mask_t[0]

        std_val = vol_t.std()
        vol_t = (vol_t - vol_t.mean()) / std_val if std_val > 1e-8 else vol_t - vol_t.mean()

        effective_um = native_um * side / self.patch_size[0]
        return {
            "image": vol_t,
            "mask": mask_t,
            "spacing": torch.tensor([effective_um] * 3, dtype=torch.float32),
            "organ": "kidney",
            "sample_id": sample["sample_id"],
        }
