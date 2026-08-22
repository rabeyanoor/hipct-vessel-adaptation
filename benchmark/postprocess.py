# benchmark/postprocess.py
"""
Choose a probability threshold and a minimum connected-component size on the
VALIDATION ROIs, then apply them to test predictions. The test ROIs are never
used to choose anything.

Both runs must have been made with `benchmark.infer ... --save-prob`.

Removing small pieces follows the observation of Jain et al. (2026) that the
one challenge team that did this (team 2) was the only one with a realistic
number of connected components.

Usage:
    python -m benchmark.postprocess tune  --val-run data/benchmark/preds/X_val --val-roi-dir data/benchmark/val_rois
    python -m benchmark.postprocess apply --run data/benchmark/preds/X --params data/benchmark/preds/X_val/postprocess.json
"""

import argparse
import itertools
import json
import os

import numpy as np
from skimage.morphology import remove_small_objects

from benchmark.rois import ROI_DIR, VAL_ROI_DIR, load_manifest

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SIZES = [0, 20, 100, 500]


def binarize(prob_u8, threshold, min_size):
    mask = prob_u8 >= round(threshold * 255)
    if min_size:
        mask = remove_small_objects(mask, max_size=min_size, connectivity=3)
    return mask


def _dice(a, b):
    s = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / s if s else 1.0


def tune(val_run, val_roi_dir=VAL_ROI_DIR):
    manifest = load_manifest(val_roi_dir)
    pairs = []
    for sid, entries in manifest.items():
        for e in entries:
            pred = np.load(os.path.join(val_run, sid, f"roi_{e['roi']:02d}.npz"))
            if "prob" not in pred:
                raise ValueError("run benchmark.infer with --save-prob first")
            pairs.append((pred["prob"], np.load(e["path"])["label"] > 0))
    results = []
    for t, m in itertools.product(THRESHOLDS, MIN_SIZES):
        d = float(np.mean([_dice(binarize(p, t, m), g) for p, g in pairs]))
        results.append({"threshold": t, "min_size": m, "val_dice": d})
        print(f"  threshold={t:.1f} min_size={m:4d}: val Dice {d:.4f}")
    best = max(results, key=lambda r: r["val_dice"])
    out = {"best": best, "grid": results, "val_run": val_run}
    with open(os.path.join(val_run, "postprocess.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[✓] best: {best}")
    return best


def apply(run, params, roi_dir=ROI_DIR, suffix="_pp"):
    best = json.load(open(params))["best"]
    manifest = load_manifest(roi_dir)
    out_run = run.rstrip("/") + suffix
    for sid, entries in manifest.items():
        os.makedirs(os.path.join(out_run, sid), exist_ok=True)
        for e in entries:
            src = os.path.join(run, sid, f"roi_{e['roi']:02d}.npz")
            if not os.path.exists(src):
                continue
            prob = np.load(src)["prob"]
            pred = binarize(prob, best["threshold"], best["min_size"]).astype(np.uint8)
            np.savez_compressed(os.path.join(out_run, sid, f"roi_{e['roi']:02d}.npz"), pred=pred)
    with open(os.path.join(out_run, "postprocess.json"), "w") as f:
        json.dump(best, f, indent=1)
    print(f"[✓] {out_run} (threshold {best['threshold']}, min_size {best['min_size']})")
    return out_run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tune", "apply"])
    p.add_argument("--val-run")
    p.add_argument("--val-roi-dir", default=VAL_ROI_DIR)
    p.add_argument("--run")
    p.add_argument("--params")
    p.add_argument("--roi-dir", default=ROI_DIR)
    a = p.parse_args()
    if a.mode == "tune":
        tune(a.val_run, a.val_roi_dir)
    else:
        apply(a.run, a.params, a.roi_dir)
