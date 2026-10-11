# benchmark/rois.py
"""
Fixed evaluation regions (ROIs) for the held-out test volumes.

Earlier results were computed on random 64^3 patches drawn from the training
kidneys, so they were neither held-out nor comparable to other papers. Here
every method is scored on the same set of ROIs cut from volumes that are never
used for training:

  kidney_5                    50.28um  same donor as kidney_1, other kidney
  kidney_6                    63.08um  other donor, other beamline (BM18)
  kidney_additional_voi_13um  13um     donor 1, 6.5um VOI binned 2x
  kidney_1_voi                5.2um    donor 1, held out from training

ROI size is set in physical units (ROI_PHYS_UM per side) and clipped to
[MIN_SIDE, MAX_SIDE] voxels, so the coarse volumes get ~9.6mm cubes and the
fine ones are still large enough (>= 615 voxels at 5.2um) for the
FOV-matched setting, where the ROI is resampled to 50um before inference.

ROI centres are drawn with a fixed seed from labelled voxels, without overlap.

Usage:
    python -m benchmark.rois --raw-tiff-root /kaggle/input/.../train data/zenodo
"""

import argparse
import json
import os
import random

import numpy as np

from utils.real_gt_dataset import RealGTKidneyDataset, _SAMPLE_INFO

ROI_DIR = "data/benchmark/rois"
VAL_ROI_DIR = "data/benchmark/val_rois"
TEST_SETS = ["kidney_5", "kidney_6", "kidney_additional_voi_13um", "kidney_1_voi"]
ROI_PHYS_UM = 9600.0
MIN_SIDE, MAX_SIDE = 160, 640
DEFAULT_N_ROIS = {"kidney_5": 8, "kidney_6": 8, "kidney_additional_voi_13um": 6, "kidney_1_voi": 6}
MIN_FG_FRACTION = 1e-4


def roi_side(spacing_um):
    return int(np.clip(round(ROI_PHYS_UM / spacing_um), MIN_SIDE, MAX_SIDE))


def choose_rois(sample, fg_coords, n, side, seed, z_limits=None):
    """z_limits=(z0, z1) keeps every ROI inside slices [z0, z1) (validation ROIs)."""
    rng = random.Random(seed)
    full = sample["shape"]
    z0, z1 = z_limits or (0, full[0])
    side_zyx = [min(side, z1 - z0), min(side, full[1]), min(side, full[2])]
    fg_coords = fg_coords[(fg_coords[:, 0] >= z0) & (fg_coords[:, 0] < z1)]
    order = list(range(len(fg_coords)))
    rng.shuffle(order)
    starts = []
    for i in order:
        c = fg_coords[i]
        lo = [z0, 0, 0]
        hi = [z1, full[1], full[2]]
        s = [int(min(max(lo[a], c[a] - side_zyx[a] // 2), hi[a] - side_zyx[a])) for a in range(3)]
        # no overlap with ROIs already chosen
        if any(all(abs(s[a] - t[a]) < side_zyx[a] for a in range(3)) for t in starts):
            continue
        starts.append(s)
        if len(starts) == n:
            break
    return starts, side_zyx


def make_rois(raw_tiff_root, test_sets=TEST_SETS, seed=0, out_dir=ROI_DIR, n_rois=None, z_frac=None):
    """z_frac={sample_id: (lo, hi)} restricts ROIs to that fraction of the slices
    (used for the validation ROIs in the held-out part of a training volume)."""
    # add to an existing manifest, so sets can be built one at a time
    manifest = {}
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
        if old.get("seed") == seed:
            manifest = old["sets"]
    for sid in test_sets:
        ds = RealGTKidneyDataset(sample_ids=[sid], raw_tiff_root=raw_tiff_root, samples_per_epoch=1)
        if not ds.samples:
            print(f"[!] {sid}: not found, skipped")
            continue
        sample = ds.samples[0]
        spacing = _SAMPLE_INFO[sid]["spacing"]
        side = roi_side(spacing)
        n = (n_rois or DEFAULT_N_ROIS).get(sid, 6)
        d = sample["shape"][0]
        zl = None
        if z_frac and sid in z_frac:
            zl = (int(z_frac[sid][0] * d), int(z_frac[sid][1] * d))
        starts, side_zyx = choose_rois(sample, ds._fg_coords(sample), n * 3, side, seed, zl)
        os.makedirs(os.path.join(out_dir, sid), exist_ok=True)
        entries = []
        for s in starts:
            if len(entries) == n:
                break
            img, lbl = ds._read_cube(sample, s[0], s[1], s[2], *side_zyx)
            fg = float(lbl.mean())
            if fg < MIN_FG_FRACTION:
                continue
            k = len(entries)
            names = sample["label_names"][s[0]:s[0] + side_zyx[0]] if sample["backend"] == "raw_tiff" else []
            path = os.path.join(out_dir, sid, f"roi_{k:02d}.npz")
            np.savez_compressed(path, image=img.astype(np.uint16), label=(lbl > 0).astype(np.uint8))
            entries.append({"roi": k, "path": path, "start_zyx": s, "size_zyx": side_zyx,
                            "slice_names": names, "volume_shape": list(sample["shape"]),
                            "spacing_um": spacing, "fg_fraction": fg})
            print(f"[*] {sid} roi_{k:02d}: start={s} size={side_zyx} fg={100 * fg:.3f}%")
        manifest[sid] = entries
    os.makedirs(out_dir, exist_ok=True)
    with open(mpath, "w") as f:
        json.dump({"seed": seed, "roi_phys_um": ROI_PHYS_UM, "sets": manifest}, f, indent=1)
    print(f"[✓] manifest written to {out_dir}/manifest.json")
    return manifest


def load_manifest(roi_dir=ROI_DIR):
    """Manifest with each entry's "path" resolved against roi_dir, so the ROI
    folder can be moved (e.g. uploaded as a Kaggle dataset)."""
    with open(os.path.join(roi_dir, "manifest.json")) as f:
        sets = json.load(f)["sets"]
    for sid, entries in sets.items():
        for e in entries:
            e["path"] = os.path.join(roi_dir, sid, f"roi_{e['roi']:02d}.npz")
    return sets


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-tiff-root", nargs="+", required=True)
    p.add_argument("--sets", nargs="+", default=TEST_SETS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=ROI_DIR)
    p.add_argument("--validation", action="store_true",
                   help="build validation ROIs in the held-out slices of the training volumes")
    a = p.parse_args()
    if a.validation:
        from train_lora_v2 import VAL_HOLDOUT
        make_rois(a.raw_tiff_root, list(VAL_HOLDOUT), a.seed, VAL_ROI_DIR,
                  n_rois={s: 6 for s in VAL_HOLDOUT}, z_frac=VAL_HOLDOUT)
    else:
        make_rois(a.raw_tiff_root, a.sets, a.seed, a.out_dir)
