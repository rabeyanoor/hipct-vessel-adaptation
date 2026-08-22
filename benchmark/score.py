# benchmark/score.py
"""
Scores a folder of ROI predictions against the benchmark ROIs.

Predictions are read from <pred_dir>/<sample_id>/roi_XX.npz (key "pred",
binary uint8, same shape as the ROI). Any method can be scored this way:
our models (benchmark/infer.py), nnU-Net and vesselFM (benchmark/nifti_io.py)
and the Kaggle winners (benchmark/import_team_rle.py).

Metrics per ROI, all in voxel units:
  Dice, clDice (hard, on skeletons, Shit et al. 2021),
  NSD at tolerance 0 and 1 voxel (the competition metric is NSD at 0),
  HD95, ASSD, and skeleton-graph stats (components, branch points).
Per test set we report the mean and a bootstrap 95% CI over ROIs.
compare() gives paired differences between two runs (bootstrap CI of the mean
difference and a Wilcoxon signed-rank p-value).

Usage:
    python -m benchmark.score data/benchmark/preds/<run> --out results/scores/<run>.json
    python -m benchmark.score --compare results/scores/a.json results/scores/b.json
"""

import argparse
import json
import os

import numpy as np
import torch
from monai.metrics import (compute_average_surface_distance, compute_hausdorff_distance,
                           compute_surface_dice)
from skimage.morphology import skeletonize

from benchmark.rois import ROI_DIR, load_manifest
from utils.vascular_graph_metrics import compute_vascular_graph_stats

METRICS = ["dice", "cldice", "nsd_t0", "nsd_t1", "hd95", "assd", "pred_components",
           "gt_components", "pred_branch_points", "gt_branch_points"]


def _crop_to_union(pred, gt, margin=4):
    union = pred | gt
    if not union.any():
        return pred, gt
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, union.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return pred[sl], gt[sl]


def _onehot(x):
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    return torch.cat([1 - t, t], dim=1)


def score_roi(pred, gt, graph_stats=True):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    # everything below only depends on the region around pred | gt; cropping
    # to it makes skeletons and graphs much cheaper on 640^3 ROIs
    pred, gt = _crop_to_union(pred, gt, margin=2)
    tp = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    out = {"dice": float(2 * tp / denom) if denom else 1.0}

    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (sp & gt).sum() / sp.sum() if sp.sum() else 0.0
    tsens = (sg & pred).sum() / sg.sum() if sg.sum() else 0.0
    out["cldice"] = float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0

    if pred.any() and gt.any():
        yp, yg = _onehot(pred), _onehot(gt)
        out["nsd_t0"] = float(compute_surface_dice(yp, yg, class_thresholds=[0.0])[0, 0])
        out["nsd_t1"] = float(compute_surface_dice(yp, yg, class_thresholds=[1.0])[0, 0])
        out["hd95"] = float(compute_hausdorff_distance(yp, yg, percentile=95)[0, 0])
        out["assd"] = float(compute_average_surface_distance(yp, yg, symmetric=True)[0, 0])
    else:
        out.update({"nsd_t0": 0.0, "nsd_t1": 0.0, "hd95": float("nan"), "assd": float("nan")})

    if graph_stats:
        ps = compute_vascular_graph_stats(pred)
        gs = compute_vascular_graph_stats(gt)
        out.update({"pred_components": ps["n_components"], "gt_components": gs["n_components"],
                    "pred_branch_points": ps["n_branch_points"], "gt_branch_points": gs["n_branch_points"]})
    return out


def bootstrap_ci(values, n_boot=10000, seed=0):
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_run(pred_dir, roi_dir=ROI_DIR, graph_stats=True):
    manifest = load_manifest(roi_dir)
    result = {"pred_dir": pred_dir, "sets": {}}
    for sid, entries in manifest.items():
        rows = []
        for e in entries:
            ppath = os.path.join(pred_dir, sid, f"roi_{e['roi']:02d}.npz")
            if not os.path.exists(ppath):
                continue
            gt = np.load(e["path"])["label"]
            pred = np.load(ppath)["pred"]
            if pred.shape != gt.shape:
                raise ValueError(f"{ppath}: shape {pred.shape} != ROI {gt.shape}")
            r = score_roi(pred, gt, graph_stats)
            r["roi"] = e["roi"]
            rows.append(r)
            print(f"  {sid} roi_{e['roi']:02d}: dice={r['dice']:.4f} cldice={r['cldice']:.4f} "
                  f"nsd0={r['nsd_t0']:.4f} nsd1={r['nsd_t1']:.4f}")
        if not rows:
            continue
        summary = {m: bootstrap_ci([r[m] for r in rows if m in r]) for m in METRICS if m in rows[0]}
        result["sets"][sid] = {"n_rois": len(rows), "per_roi": rows, "summary": summary}
    return result


def compare(path_a, path_b, metrics=("dice", "cldice", "nsd_t0", "nsd_t1"), n_boot=10000):
    from scipy.stats import wilcoxon
    a, b = json.load(open(path_a)), json.load(open(path_b))
    out = {}
    for sid in sorted(set(a["sets"]) & set(b["sets"])):
        ra = {r["roi"]: r for r in a["sets"][sid]["per_roi"]}
        rb = {r["roi"]: r for r in b["sets"][sid]["per_roi"]}
        common = sorted(set(ra) & set(rb))
        out[sid] = {}
        for m in metrics:
            d = np.array([ra[k][m] - rb[k][m] for k in common])
            mean, lo, hi = bootstrap_ci(d, n_boot)
            p = float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else float("nan")
            out[sid][m] = {"mean_diff": mean, "ci95": [lo, hi], "wilcoxon_p": p, "n": len(d)}
    return out


def print_table(results):
    for res in results:
        print(f"\n{res['pred_dir']}")
        for sid, s in res["sets"].items():
            cells = " | ".join(f"{m}={s['summary'][m][0]:.3f} [{s['summary'][m][1]:.3f},{s['summary'][m][2]:.3f}]"
                               for m in ("dice", "cldice", "nsd_t0", "nsd_t1") if m in s["summary"])
            print(f"  {sid:28s} n={s['n_rois']:2d} | {cells}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pred_dir", nargs="?")
    p.add_argument("--roi-dir", default=ROI_DIR)
    p.add_argument("--out", default=None)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--compare", nargs=2, default=None)
    a = p.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=1))
    else:
        res = score_run(a.pred_dir, a.roi_dir, graph_stats=not a.no_graph)
        print_table([res])
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
