# benchmark/build_rois_zenodo.py
"""
Builds the benchmark ROIs straight from the Zenodo release without
downloading whole volumes (the 13um VOI alone is ~24 GB).

For each test set:
  1. list the slices from the zip index and keep those with image data;
  2. download every 4th label slice and build the same coarse foreground index
     as RealGTKidneyDataset._fg_coords (stride 4);
  3. pick ROI positions with benchmark.rois.choose_rois (same seed, same rule);
  4. download only the slices inside each ROI, cut the cube, save it, and
     delete the slices again.
Steps 2-3 give the same ROIs as benchmark.rois.make_rois on the full data.

Usage (e.g. in a Kaggle CPU kernel with internet on):
    python -m benchmark.build_rois_zenodo --sets kidney_5 kidney_6 kidney_additional_voi_13um kidney_1_voi
"""

import argparse
import json
import os
import shutil

import numpy as np
import tifffile

from benchmark.rois import DEFAULT_N_ROIS, MIN_FG_FRACTION, ROI_DIR, ROI_PHYS_UM, TEST_SETS, choose_rois, roi_side
from utils.fetch_zenodo_subset import fetch_files, list_slices
from utils.real_gt_dataset import _SAMPLE_INFO

STRIDE = 4


def _read(tmp, sid, kind, name):
    return tifffile.imread(os.path.join(tmp, sid, kind, name))


def build(sid, seed=0, out_dir=ROI_DIR, tmp=None, n=None, workers=4):
    tmp = tmp or os.path.normpath(out_dir) + "_tmp"
    slices = [s for s in list_slices(sid, out_dir=tmp) if s[1] > 0 and s[2] > 0]
    names = [s[0] for s in slices]
    index_names = set(names[::STRIDE])
    fetch_files(sid, names[::STRIDE], kinds=("labels",), out_dir=tmp, workers=workers)
    sub = np.stack([_read(tmp, sid, "labels", nm)[::STRIDE, ::STRIDE] for nm in names[::STRIDE]])
    h, w = _read(tmp, sid, "labels", names[0]).shape
    coords = (np.argwhere(sub > 0) * STRIDE).astype(np.int32)
    shape = (len(names), h, w)
    spacing = _SAMPLE_INFO[sid]["spacing"]
    n = n or DEFAULT_N_ROIS.get(sid, 6)
    starts, size = choose_rois({"shape": shape}, coords, n * 3, roi_side(spacing), seed)
    os.makedirs(os.path.join(out_dir, sid), exist_ok=True)

    entries = []
    for s in starts:
        if len(entries) == n:
            break
        roi_names = names[s[0]:s[0] + size[0]]
        fetch_files(sid, roi_names, out_dir=tmp, workers=workers)
        img = np.stack([_read(tmp, sid, "images", nm)[s[1]:s[1] + size[1], s[2]:s[2] + size[2]] for nm in roi_names])
        lbl = np.stack([_read(tmp, sid, "labels", nm)[s[1]:s[1] + size[1], s[2]:s[2] + size[2]] > 0
                        for nm in roi_names])
        for kind in ("images", "labels"):
            for nm in roi_names:
                p = os.path.join(tmp, sid, kind, nm)
                if kind == "images" or nm not in index_names:
                    if os.path.exists(p):
                        os.remove(p)
        fg = float(lbl.mean())
        if fg < MIN_FG_FRACTION:
            continue
        k = len(entries)
        np.savez_compressed(os.path.join(out_dir, sid, f"roi_{k:02d}.npz"),
                            image=img.astype(np.uint16), label=lbl.astype(np.uint8))
        entries.append({"roi": k, "path": os.path.join(out_dir, sid, f"roi_{k:02d}.npz"), "start_zyx": s,
                        "size_zyx": size, "slice_names": roi_names, "volume_shape": list(shape),
                        "spacing_um": spacing, "fg_fraction": fg})
        print(f"[*] {sid} roi_{k:02d}: start={s} size={size} fg={100 * fg:.3f}%", flush=True)
    shutil.rmtree(os.path.join(tmp, sid), ignore_errors=True)
    return entries


def main(sets, seed=0, out_dir=ROI_DIR, workers=4, tmp=None):
    mpath = os.path.join(out_dir, "manifest.json")
    manifest = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
        if old.get("seed") == seed:
            manifest = old["sets"]
    for sid in sets:
        manifest[sid] = build(sid, seed, out_dir, tmp=tmp, workers=workers)
        with open(mpath, "w") as f:
            json.dump({"seed": seed, "roi_phys_um": ROI_PHYS_UM, "sets": manifest}, f, indent=1)
    print(f"[✓] manifest written to {mpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=TEST_SETS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=ROI_DIR)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tmp", default=None, help="where slices are downloaded temporarily")
    a = p.parse_args()
    main(a.sets, a.seed, a.out_dir, a.workers, a.tmp)
