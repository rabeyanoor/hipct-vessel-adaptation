# benchmark/aggregate.py
"""
Builds the result tables from scored runs (benchmark/score.py JSON files).

Runs of the same configuration trained with different seeds are named
<config>_s<seed>_<setting>.json (e.g. spacing_log_aug_s1_native.json). For each
configuration and test set, every ROI's metric is first averaged over the
seeds, and the mean and 95 % bootstrap CI are then taken over ROIs, so the ROI
is the unit of analysis (the seeds only reduce training noise). Per-seed means
are reported as well.

paired(): difference between two configurations on the same ROIs (seed-averaged),
with a bootstrap CI of the mean difference and a two-sided Wilcoxon signed-rank
p-value (exact for these small n).

Usage:
    python -m benchmark.aggregate --results benchmark_results --out results/tables.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from benchmark.score import bootstrap_ci

SETS = ["kidney_5", "kidney_6", "kidney_additional_voi_13um", "kidney_1_voi"]
METRICS = ["dice", "cldice", "nsd_t0", "nsd_t1", "hd95", "assd", "pred_components", "gt_components",
           "pred_branch_points", "gt_branch_points"]
_SEED = re.compile(r"^(?P<cfg>.+?)_s(?P<seed>\d+)_(?P<setting>[a-z0-9]+(?:_pp)?)$")


def load(results_dir):
    """{(config, setting): {seed: result}}; runs without a seed get seed None."""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        # <run>__<part>.json: the same run scored in several kernels (e.g. coarse
        # sets on CPU, fine sets on GPU); their test sets are merged
        name = os.path.splitext(os.path.basename(path))[0].split("__")[0]
        res = json.load(open(path))
        if "sets" not in res:
            continue
        m = _SEED.match(name)
        key, seed = ((m["cfg"], m["setting"]), int(m["seed"])) if m else ((name, ""), None)
        if seed in runs[key]:
            overlap = set(runs[key][seed]["sets"]) & set(res["sets"])
            if overlap:
                raise ValueError(f"{path}: sets {overlap} already loaded for {name}")
            runs[key][seed]["sets"].update(res["sets"])
        else:
            runs[key][seed] = res
    return runs


def per_roi(seed_results, sid, metric):
    """{roi: mean over seeds} for one test set."""
    vals = defaultdict(list)
    for res in seed_results.values():
        if sid not in res["sets"]:
            continue
        for r in res["sets"][sid]["per_roi"]:
            v = r.get(metric)
            if v is not None and v == v:
                vals[r["roi"]].append(v)
    return {k: float(np.mean(v)) for k, v in vals.items()}


def summarize(runs):
    table = {}
    for (cfg, setting), seeds in sorted(runs.items()):
        key = f"{cfg}|{setting}" if setting else cfg
        table[key] = {"n_seeds": len(seeds), "seeds": sorted(s for s in seeds if s is not None), "sets": {}}
        for sid in SETS:
            if not any(sid in r["sets"] for r in seeds.values()):
                continue
            entry = {}
            for m in METRICS:
                roi_vals = per_roi(seeds, sid, m)
                if not roi_vals:
                    continue
                mean, lo, hi = bootstrap_ci(list(roi_vals.values()))
                per_seed = {}
                for s, r in seeds.items():
                    if sid in r["sets"]:
                        v = [x.get(m) for x in r["sets"][sid]["per_roi"]]
                        v = [x for x in v if x is not None and x == x]
                        if v:
                            per_seed[str(s)] = float(np.mean(v))
                entry[m] = {"mean": mean, "ci95": [lo, hi], "n_rois": len(roi_vals), "per_seed_mean": per_seed}
            table[key]["sets"][sid] = entry
    return table


def paired(runs, a, b, metric="dice", sets=SETS, n_boot=10000):
    """a, b: (config, setting) keys of `runs`."""
    from scipy.stats import wilcoxon
    out = {}
    for sid in sets:
        ra, rb = per_roi(runs[a], sid, metric), per_roi(runs[b], sid, metric)
        common = sorted(set(ra) & set(rb))
        if len(common) < 2:
            continue
        d = np.array([ra[k] - rb[k] for k in common])
        mean, lo, hi = bootstrap_ci(d, n_boot)
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        out[sid] = {"mean_diff": mean, "ci95": [lo, hi], "wilcoxon_p": p, "n_rois": len(common),
                    "n_positive": int((d > 0).sum())}
    return out


def fmt(entry, m, digits=3):
    if m not in entry:
        return "n/a"
    e = entry[m]
    return f"{e['mean']:.{digits}f} [{e['ci95'][0]:.{digits}f}, {e['ci95'][1]:.{digits}f}]"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="benchmark_results")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    runs = load(a.results)
    table = summarize(runs)
    for key, t in table.items():
        print(f"\n{key}  (seeds: {t['seeds'] or '-'})")
        for sid, e in t["sets"].items():
            print(f"  {sid:28s} dice {fmt(e, 'dice')} | cldice {fmt(e, 'cldice')} | "
                  f"nsd0 {fmt(e, 'nsd_t0')} | nsd1 {fmt(e, 'nsd_t1')}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(table, open(a.out, "w"), indent=1)
