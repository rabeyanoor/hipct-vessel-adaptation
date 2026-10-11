# benchmark/paper_tables.py
"""
Builds every results table of the paper (PAPER_MEDIA.md, Section 5) from the
scored runs in one folder (benchmark/score.py JSON files), so each number in
the paper can be regenerated with one command.

Unit of analysis: the ROI. For a configuration trained with several seeds,
each ROI's metric is first averaged over the seeds (benchmark/aggregate.py).
Paired comparisons use the same ROIs for both sides: mean difference with a
95 % bootstrap CI and a two-sided Wilcoxon signed-rank test, per test set and
pooled over all test sets of a group.

Usage:
    python -m benchmark.paper_tables --results results/scores --out results/paper_tables.md \
        --json results/paper_tables.json
"""

import argparse
import json

import numpy as np

from benchmark.aggregate import SETS, load, paired, per_roi, summarize
from benchmark.score import bootstrap_ci

SET_LABEL = {"kidney_5": "kidney_5 (50.28 µm)", "kidney_6": "kidney_6 (63.08 µm)",
             "kidney_additional_voi_13um": "13 µm", "kidney_1_voi": "kidney_1_voi (5.2 µm)"}

# (key in the loaded runs, label in the paper)
MAIN_ROWS = [
    (("kaggle_team1", ""), "Kaggle team 1"),
    (("kaggle_team2", ""), "Kaggle team 2"),
    (("kaggle_team3", ""), "Kaggle team 3"),
    (("kaggle_team4", ""), "Kaggle team 4"),
    (("kaggle_team5", ""), "Kaggle team 5"),
    (("nnunet_resampled", ""), "nnU-Net (resampled to 50 µm)"),
    (("nnunet_native", ""), "nnU-Net (native grid)"),
    (("vesselfm_noinvert", ""), "vesselFM"),
    (("vesselfm_invert", ""), "vesselFM (inverted)"),
    (("sam_zeroshot_native", ""), "SAM-Med3D, no fine-tuning"),
    (("r1_plain", "native"), "Plain LoRA, first round (64³)"),
    (("r1_spacing", "native"), "Spacing LoRA, first round (64³)"),
    (("p_base", "native"), "Plain LoRA"),
    (("s_base", "native"), "Spacing LoRA"),
    (("p_aug", "native"), "Plain LoRA + spacing aug."),
    (("s_aug", "native"), "Spacing LoRA + spacing aug."),
    (("p_base", "fov50"), "Plain LoRA, resampled to 50 µm"),
    (("s_base", "fov50"), "Spacing LoRA, resampled to 50 µm"),
    (("p_aug", "fov50"), "Plain LoRA + aug., resampled to 50 µm"),
    (("s_aug", "fov50"), "Spacing LoRA + aug., resampled to 50 µm"),
]

# configurations of the spacing / field-of-view test
FOV_CONFIGS = [("r1_plain", "Plain LoRA, first round"), ("r1_spacing", "Spacing LoRA, first round"),
               ("p_base", "Plain LoRA"), ("s_base", "Spacing LoRA"),
               ("p_aug", "Plain LoRA + aug."), ("s_aug", "Spacing LoRA + aug.")]

# paired comparisons: (a, b, label); difference is a - b
COMPARISONS = [
    (("s_base", "native"), ("p_base", "native"), "spacing input (no aug.): Spacing LoRA - Plain LoRA"),
    (("s_aug", "native"), ("p_aug", "native"), "spacing input (with aug.): Spacing LoRA - Plain LoRA"),
    (("p_aug", "native"), ("p_base", "native"), "augmentation, Plain LoRA: with - without"),
    (("s_aug", "native"), ("s_base", "native"), "augmentation, Spacing LoRA: with - without"),
    (("p_base", "fov50"), ("p_base", "native"), "resampling to 50 µm, Plain LoRA: fov50 - native"),
    (("s_base", "fov50"), ("s_base", "native"), "resampling to 50 µm, Spacing LoRA: fov50 - native"),
    (("p_aug", "fov50"), ("p_aug", "native"), "resampling to 50 µm, Plain LoRA + aug.: fov50 - native"),
    (("s_aug", "fov50"), ("s_aug", "native"), "resampling to 50 µm, Spacing LoRA + aug.: fov50 - native"),
    (("s_base", "tell50"), ("s_base", "native"), "spacing value only, Spacing LoRA: tell50 - native"),
    (("s_aug", "tell50"), ("s_aug", "native"), "spacing value only, Spacing LoRA + aug.: tell50 - native"),
    (("r1_spacing", "tell50"), ("r1_spacing", "native"), "spacing value only, first round: tell50 - native"),
    (("r1_spacing", "fov50"), ("r1_spacing", "native"), "resampling to 50 µm, first round spacing: fov50 - native"),
    (("r1_plain", "fov50"), ("r1_plain", "native"), "resampling to 50 µm, first round plain: fov50 - native"),
]
GROUPS = {"coarse (kidney_5, kidney_6)": ["kidney_5", "kidney_6"],
          "fine (13 µm, 5.2 µm)": ["kidney_additional_voi_13um", "kidney_1_voi"],
          "all four": SETS}


def pooled(runs, a, b, metric, sets, n_boot=10000):
    """Paired difference over the ROIs of several test sets together."""
    from scipy.stats import wilcoxon
    d = []
    for sid in sets:
        ra, rb = per_roi(runs[a], sid, metric), per_roi(runs[b], sid, metric)
        d += [ra[k] - rb[k] for k in sorted(set(ra) & set(rb))]
    if len(d) < 2:
        return None
    d = np.array(d)
    mean, lo, hi = bootstrap_ci(d, n_boot)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean_diff": mean, "ci95": [lo, hi], "wilcoxon_p": p, "n_rois": len(d), "n_positive": int((d > 0).sum())}


def cell(table, key, sid, metrics=("dice", "cldice", "nsd_t0", "nsd_t1")):
    if key not in table or sid not in table[key]["sets"]:
        return "–"
    e = table[key]["sets"][sid]
    if "dice" not in e:
        return "–"
    d = e["dice"]
    rest = " / ".join(f"{e[m]['mean']:.3f}" if m in e else "–" for m in metrics[1:])
    return f"{d['mean']:.3f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] / {rest}"


def table_key(cfg, setting):
    return f"{cfg}|{setting}" if setting else cfg


def build(results_dir):
    runs = load(results_dir)
    table = summarize(runs)
    lines, out = [], {"summary": table, "comparisons": {}}

    lines.append("## Table 1. Held-out benchmark\n")
    lines.append("Dice [95 % CI] / clDice / NSD (t = 0) / NSD (t = 1); mean over ROIs (seed-averaged per ROI).\n")
    lines.append("| Method | seeds | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
    lines.append("|---|---|" + "---|" * len(SETS))
    for (cfg, setting), label in MAIN_ROWS:
        k = table_key(cfg, setting)
        if k not in table:
            continue
        seeds = table[k]["seeds"]
        lines.append(f"| {label} | {len(seeds) if seeds else '–'} | "
                     + " | ".join(cell(table, k, s) for s in SETS) + " |")

    lines.append("\n## Table 2. Spacing input vs. field of view (Dice)\n")
    lines.append("native: native grid, true spacing; tell50: native grid, model told 50 µm; "
                 "fov50: ROI resampled to 50 µm.\n")
    lines.append("| Model | setting | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
    lines.append("|---|---|" + "---|" * len(SETS))
    for cfg, label in FOV_CONFIGS:
        for setting in ("native", "tell50", "fov50"):
            k = table_key(cfg, setting)
            if k not in table:
                continue
            cells = []
            for s in SETS:
                e = table[k]["sets"].get(s, {})
                cells.append(f"{e['dice']['mean']:.3f} [{e['dice']['ci95'][0]:.2f}, {e['dice']['ci95'][1]:.2f}]"
                             if "dice" in e else "–")
            lines.append(f"| {label} | {setting} | " + " | ".join(cells) + " |")

    lines.append("\n## Table 3. Paired comparisons (Dice difference, a - b)\n")
    lines.append("Mean difference [95 % bootstrap CI], Wilcoxon signed-rank p, ROIs where a > b.\n")
    lines.append("| Comparison | " + " | ".join(SET_LABEL[s] for s in SETS) + " | " + " | ".join(GROUPS) + " |")
    lines.append("|---|" + "---|" * (len(SETS) + len(GROUPS)))
    for a, b, label in COMPARISONS:
        if a not in runs or b not in runs:
            continue
        per_set = paired(runs, a, b, "dice")
        pools = {g: pooled(runs, a, b, "dice", sets) for g, sets in GROUPS.items()}
        out["comparisons"][label] = {"per_set": per_set, "pooled": pools}

        def fmt(r):
            if not r:
                return "–"
            return (f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], "
                    f"p={r['wilcoxon_p']:.3g}, {r['n_positive']}/{r['n_rois']}")
        lines.append(f"| {label} | " + " | ".join(fmt(per_set.get(s)) for s in SETS) + " | "
                     + " | ".join(fmt(pools[g]) for g in GROUPS) + " |")

    lines.append("\n## Table 4. Structure of the errors (median over ROIs, seed 0 or single run)\n")
    lines.append("Skeleton connected components and branch voxels (degree >= 3), prediction vs. labels.\n")
    lines.append("| Method | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
    lines.append("|---|" + "---|" * len(SETS))
    for (cfg, setting), label in MAIN_ROWS:
        if (cfg, setting) not in runs:
            continue
        seeds = runs[(cfg, setting)]
        res = seeds[min(seeds, key=lambda s: -1 if s is None else s)]
        cells = []
        for s in SETS:
            rows = res["sets"].get(s, {}).get("per_roi", [])
            if not rows:
                cells.append("–")
                continue
            med = {m: float(np.median([r[m] for r in rows])) for m in
                   ("pred_components", "gt_components", "pred_branch_points", "gt_branch_points")}
            cells.append(f"{med['pred_components']:.0f} vs {med['gt_components']:.0f} comp.; "
                         f"{med['pred_branch_points']:.0f} vs {med['gt_branch_points']:.0f} br.")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/scores")
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    md, data = build(a.results)
    print(md)
    if a.out:
        open(a.out, "w").write(md)
    if a.json:
        json.dump(data, open(a.json, "w"), indent=1)
