# benchmark/import_team_rle.py
"""
Turns the Kaggle winners' RLE predictions (released by Jain et al. on Zenodo,
fetched with `python -m utils.fetch_zenodo_subset team_predictions`) into ROI
predictions that benchmark/score.py can read.

Covered sets: kidney_5, kidney_6 (competition-test-set/team_N.csv) and
kidney_additional_voi_13um (team-N-kidney_additional_voi_13um-predictions.csv).
The teams trained on kidney_1_voi, so there is no OOD prediction for it.

RLE format (Kaggle): per slice, space separated 1-based start/length pairs over
the row-major flattened H x W slice; "1 0" means empty.

Usage:
    python -m benchmark.import_team_rle --teams 1 2 3 4 5
"""

import argparse
import csv
import os
import sys

import numpy as np

from benchmark.rois import ROI_DIR, load_manifest

PRED_ROOT = "data/benchmark/preds"
TEAM_DIR = "data/zenodo/team_predictions"


def rle_decode(rle, shape):
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    nums = rle.split()
    for s, n in zip(nums[0::2], nums[1::2]):
        s, n = int(s) - 1, int(n)
        flat[s:s + n] = 1
    return flat.reshape(shape)


def _team_files(team):
    return [os.path.join(TEAM_DIR, f"team_{team}.csv"),
            os.path.join(TEAM_DIR, f"team-{team}-kidney_additional_voi_13um-predictions.csv")]


def _load_rles(paths, wanted_ids):
    csv.field_size_limit(sys.maxsize)
    rles = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["id"] in wanted_ids:
                    rles[row["id"]] = row["rle"]
    return rles


def import_team(team, roi_dir=ROI_DIR, pred_root=PRED_ROOT):
    manifest = load_manifest(roi_dir)
    sets = [s for s in manifest if s != "kidney_1_voi"]
    wanted = {}
    for sid in sets:
        for e in manifest[sid]:
            for name in e["slice_names"]:
                wanted[f"{sid}_{os.path.splitext(name)[0]}"] = None
    rles = _load_rles(_team_files(team), set(wanted))
    out_dir = os.path.join(pred_root, f"kaggle_team{team}")
    for sid in sets:
        os.makedirs(os.path.join(out_dir, sid), exist_ok=True)
        for e in manifest[sid]:
            z0, y0, x0 = e["start_zyx"]
            d, h, w = e["size_zyx"]
            hw = tuple(e["volume_shape"][1:])
            pred = np.zeros((d, h, w), dtype=np.uint8)
            missing = 0
            for k, name in enumerate(e["slice_names"]):
                key = f"{sid}_{os.path.splitext(name)[0]}"
                if key not in rles:
                    missing += 1
                    continue
                pred[k] = rle_decode(rles[key], hw)[y0:y0 + h, x0:x0 + w]
            if missing:
                print(f"[!] team {team} {sid} roi_{e['roi']:02d}: {missing} slices missing in RLE file")
            np.savez_compressed(os.path.join(out_dir, sid, f"roi_{e['roi']:02d}.npz"), pred=pred)
    print(f"[✓] team {team} -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teams", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--roi-dir", default=ROI_DIR)
    a = p.parse_args()
    for t in a.teams:
        import_team(t, a.roi_dir)
