# generate_summary_tables.py
"""
Assembles every real result this project has produced into one consolidated,
text-only report -- source material for the paper's results section, not a
new experiment. No plotting anywhere (project rule) -- every table below is
plain aligned text.

Two kinds of numbers, clearly labeled apart:
  1. Read live from this repo's own *_results.json files (sweep_results.json,
     full_sweep_results.json, vascular_graph_results.json, ablation_results.json
     and multiseed_seed1_results.json if/when they exist) -- always current,
     computed fresh from whatever's on disk when this script runs.
  2. TRANSCRIBED from test.sh (the 100-epoch main comparison, FOV-confound
     test, FiLM breakdown, clDice ablation) -- these were never saved as JSON
     by the scripts that produced them (evaluate_fov_confound.py,
     analyze_film_layers.py, run_ablation.py's manual invocation), so they are
     copied here as literal constants with their test.sh section cited. If
     test.sh's numbers for these sections ever change, this file's constants
     need updating by hand to match -- they do not auto-sync.

Missing JSON files print as "not yet available" rather than crashing, since
several (multiseed in particular) are still in progress.
"""

import glob
import json
import os
import statistics

# ================================================================
# Section 2: transcribed constants (see module docstring, part 2)
# ================================================================

MAIN_100EP_COMPARISON = {
    # test.sh section 3.4
    "plain_lora": {"in_dist_dice": 0.1838, "in_dist_n": "26/100", "ood_dice": 0.1672, "ood_n": "5/40"},
    "spacing_lora": {"in_dist_dice": 0.3145, "in_dist_n": "18/100", "ood_dice": 0.0058, "ood_n": "5/40"},
}

FOV_CONFOUND = {
    # test.sh section 3.5
    "plain_lora": {"in_dist_dice": 0.1838, "ood_native_dice": 0.1672, "ood_native_n": "5/40",
                   "ood_fov_matched_dice": 0.2895, "ood_fov_matched_n": "40/40"},
    "spacing_lora": {"in_dist_dice": 0.3145, "ood_native_dice": 0.0058, "ood_native_n": "5/40",
                      "ood_fov_matched_dice": 0.2215, "ood_fov_matched_n": "40/40"},
}
FOV_SIGNIFICANCE = {
    # test.sh section 3.5 -- fresh 40-patch draw per model, independent of the table above
    "plain_lora_dice": 0.3681, "spacing_lora_dice": 0.2618,
    "mann_whitney_p": 0.0216, "alpha": 0.05,
}

FILM_BREAKDOWN = {
    # test.sh section 3.6 -- Spacing-Conditioned LoRA, 100ep checkpoint, 2 of 24 layers cited as examples
    "layer_19": {"gamma_mean_in_dist_50um": 6.52, "gamma_max_in_dist_50um": 21.6,
                 "gamma_mean_ood_5_2um": 1.59, "gamma_max_ood_5_2um": 3.23},
    "layer_2": {"gamma_max_in_dist_50um": 19.7, "gamma_max_ood_5_2um": 3.06},
    "pattern": "holds broadly across all 24 layers, not one outlier",
}

CLDICE_ABLATION = {
    # test.sh section 3.7
    "spacing_lora_with_cldice": {"in_dist_dice": 0.3145, "in_dist_n": "18/100", "ood_dice": 0.0058, "ood_n": "5/40"},
    "spacing_lora_no_cldice": {"in_dist_dice": 0.1898, "in_dist_n": "19/100", "ood_dice": 0.0015, "ood_n": "3/40"},
}

VASCULAR_GRAPH_MEDIANS = {
    # test.sh section 3.10 -- median across non-empty patches, computed from vascular_graph_results.json
    # (also re-derived live below if the JSON is present; kept here as a fallback/cross-check reference)
    "plain_lora_in_dist": {"n": "23/80", "pred_components": 3.0, "gt_components": 3.0,
                            "pred_branch": 0.0, "gt_branch": 1.0, "pred_endpoints": 6.0, "gt_endpoints": 8.0},
    "plain_lora_ood": {"n": "5/40", "pred_components": 6.0, "gt_components": 1.0,
                        "pred_branch": 1.0, "gt_branch": 1.0, "pred_endpoints": 10.0, "gt_endpoints": 3.0},
    "spacing_lora_in_dist": {"n": "22/80", "pred_components": 3.0, "gt_components": 1.0,
                              "pred_branch": 0.5, "gt_branch": 0.5, "pred_endpoints": 7.0, "gt_endpoints": 3.0},
    "spacing_lora_ood": {"n": "8/40", "pred_components": 11.0, "gt_components": 1.0,
                          "pred_branch": 15.5, "gt_branch": 0.5, "pred_endpoints": 20.5, "gt_endpoints": 2.5},
}


def _hr(char="=", width=100):
    print(char * width)


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def section_main_comparison():
    print("\n1. MAIN 100-EPOCH COMPARISON (transcribed from test.sh section 3.4)")
    _hr()
    print(f"{'Model':<28} | {'In-dist Dice':<18} | {'OOD Dice':<18}")
    _hr("-")
    for name, d in MAIN_100EP_COMPARISON.items():
        print(f"{name:<28} | {d['in_dist_dice']:.4f} (n={d['in_dist_n']:<7}) | "
              f"{d['ood_dice']:.4f} (n={d['ood_n']:<7})")
    _hr()


def section_hp_sweep():
    print("\n2. HYPERPARAMETER SWEEPS (live from sweep_results.json / full_sweep_results.json)")
    for label, path in [("Reduced sweep (5 epochs, 12 configs)", "sweep_results.json"),
                         ("Full sweep (15 epochs, 18 configs)", "full_sweep_results.json")]:
        _hr()
        print(label)
        _hr("-")
        data = _load_json(path)
        if data is None:
            print(f"  not yet available ({path} not found)")
            continue
        for arch in ("plain_lora", "spacing_lora"):
            arch_runs = {k: v for k, v in data.items() if k.startswith(arch)}
            if not arch_runs:
                continue
            best_id = max(arch_runs, key=lambda k: arch_runs[k]["dice"] if arch_runs[k]["dice"] is not None else -1)
            best = arch_runs[best_id]
            print(f"  {arch:<14} best: lr={best['lr']}, alpha={best['alpha']}, Dice={best['dice']:.4f} "
                  f"({len(arch_runs)} configs tried)")
    _hr()


def section_fov_confound():
    print("\n3. FIELD-OF-VIEW CONFOUND TEST (transcribed from test.sh section 3.5)")
    _hr()
    print(f"{'Model':<16} | {'In-dist':<10} | {'OOD native':<18} | {'OOD FOV-matched':<18}")
    _hr("-")
    for name, d in FOV_CONFOUND.items():
        print(f"{name:<16} | {d['in_dist_dice']:<10.4f} | {d['ood_native_dice']:.4f} (n={d['ood_native_n']:<6}) | "
              f"{d['ood_fov_matched_dice']:.4f} (n={d['ood_fov_matched_n']:<6})")
    print(f"\nSignificance (independent 40-patch redraw): Plain LoRA={FOV_SIGNIFICANCE['plain_lora_dice']}, "
          f"Spacing LoRA={FOV_SIGNIFICANCE['spacing_lora_dice']}, "
          f"Mann-Whitney U p={FOV_SIGNIFICANCE['mann_whitney_p']} "
          f"(significant at alpha={FOV_SIGNIFICANCE['alpha']})")
    _hr()


def section_film_breakdown():
    print("\n4. PER-LAYER FiLM BREAKDOWN (transcribed from test.sh section 3.6)")
    _hr()
    l19, l2 = FILM_BREAKDOWN["layer_19"], FILM_BREAKDOWN["layer_2"]
    print(f"  Layer 19: gamma_mean {l19['gamma_mean_in_dist_50um']} (50um) -> {l19['gamma_mean_ood_5_2um']} (5.2um); "
          f"gamma_max {l19['gamma_max_in_dist_50um']} -> {l19['gamma_max_ood_5_2um']}")
    print(f"  Layer 2:  gamma_max {l2['gamma_max_in_dist_50um']} (50um) -> {l2['gamma_max_ood_5_2um']} (5.2um)")
    print(f"  Pattern: {FILM_BREAKDOWN['pattern']}")
    _hr()


def section_cldice_ablation():
    print("\n5. clDICE LOSS TERM ABLATION (transcribed from test.sh section 3.7)")
    _hr()
    for name, d in CLDICE_ABLATION.items():
        print(f"  {name:<32} in-dist={d['in_dist_dice']:.4f} (n={d['in_dist_n']}), "
              f"OOD={d['ood_dice']:.4f} (n={d['ood_n']})")
    _hr()


def section_vascular_graph():
    print("\n6. VASCULAR GRAPH VALIDATION (live from vascular_graph_results.json if present, else fallback constants)")
    _hr()
    data = _load_json("vascular_graph_results.json")
    print(f"{'Condition':<24} | {'n':<8} | {'Pred comp':<10} | {'GT comp':<8} | {'Pred branch':<12} | "
          f"{'GT branch':<10} | {'Pred endp':<10} | {'GT endp':<8}")
    _hr("-")
    if data is None:
        print("  (vascular_graph_results.json not found -- showing fallback constants transcribed from test.sh 3.10)")
        for cond, d in VASCULAR_GRAPH_MEDIANS.items():
            print(f"{cond:<24} | {d['n']:<8} | {d['pred_components']:<10} | {d['gt_components']:<8} | "
                  f"{d['pred_branch']:<12} | {d['gt_branch']:<10} | {d['pred_endpoints']:<10} | {d['gt_endpoints']:<8}")
    else:
        for cond, rows in data.items():
            n = len(rows)
            if n == 0:
                print(f"{cond:<24} | {'0/?':<8} | (no non-empty patches)")
                continue

            def med(side, field):
                return statistics.median(row[side][field] for row in rows)

            print(f"{cond:<24} | {n:<8} | {med('pred', 'n_components'):<10} | {med('gt', 'n_components'):<8} | "
                  f"{med('pred', 'n_branch_points'):<12} | {med('gt', 'n_branch_points'):<10} | "
                  f"{med('pred', 'n_endpoints'):<10} | {med('gt', 'n_endpoints'):<8}")
    _hr()


def section_multiseed():
    print("\n7. MULTI-SEED STATISTICAL REPORTING (live, in progress)")
    _hr()
    full = _load_json("multiseed_results.json")
    if full is not None:
        print("  Full multiseed_results.json found:")
        for arch, d in full.items():
            in_vals = [v for v in d.get("in_dist", []) if v is not None]
            ood_vals = [v for v in d.get("ood", []) if v is not None]
            in_mean = statistics.mean(in_vals) if in_vals else float("nan")
            in_std = statistics.stdev(in_vals) if len(in_vals) > 1 else 0.0
            ood_mean = statistics.mean(ood_vals) if ood_vals else float("nan")
            ood_std = statistics.stdev(ood_vals) if len(ood_vals) > 1 else 0.0
            print(f"  {arch:<16} n={len(in_vals)} seeds: in-dist Dice = {in_mean:.4f} +/- {in_std:.4f}, "
                  f"OOD Dice = {ood_mean:.4f} +/- {ood_std:.4f}")
        _hr()
        return

    # Combine the existing 100-epoch checkpoint (unspecified seed, section 1 above) with
    # every multiseed_seed<N>_results.json found on disk -- n grows as more seeds are added,
    # no need to update this script each time.
    seed_files = sorted(glob.glob("multiseed_seed*_results.json"))
    if not seed_files:
        print("  not yet available -- kaggle_kernels/kaggle_multiseed_kernel*.py has not completed yet "
              "(see baki.sh section 2 for status)")
        _hr()
        return

    print(f"  Existing checkpoint (unspecified seed) + {len(seed_files)} additional seed run(s) "
          f"({', '.join(seed_files)}):")
    for arch in ("plain_lora", "spacing_lora"):
        existing = MAIN_100EP_COMPARISON.get(arch, {})
        in_vals = [existing.get("in_dist_dice")]
        ood_vals = [existing.get("ood_dice")]
        for sf in seed_files:
            d = _load_json(sf).get(arch, {})
            in_vals.append(d.get("in_dist_dice"))
            ood_vals.append(d.get("ood_dice"))
        in_vals = [v for v in in_vals if v is not None]
        ood_vals = [v for v in ood_vals if v is not None]
        in_mean = statistics.mean(in_vals) if in_vals else float("nan")
        in_std = statistics.stdev(in_vals) if len(in_vals) > 1 else 0.0
        ood_mean = statistics.mean(ood_vals) if ood_vals else float("nan")
        ood_std = statistics.stdev(ood_vals) if len(ood_vals) > 1 else 0.0
        print(f"  {arch:<16} n={len(in_vals)}: in-dist Dice = {in_mean:.4f} +/- {in_std:.4f}, "
              f"OOD Dice = {ood_mean:.4f} +/- {ood_std:.4f}")
    n_total = 1 + len(seed_files)
    if n_total < 3:
        print(f"  CAVEAT: n={n_total} is not a properly powered estimate -- more seeds are still "
              f"pending (see baki.sh section 2).")
    else:
        print(f"  n={n_total} reached (originally-planned multiseed size) -- see test.sh section 3.11 "
              f"for the full interpretation, still a small-n result by normal statistical standards.")
    _hr()


def main():
    print("HiP-VesselAdapt -- Consolidated Results Summary (text-only, no plotting)")
    print("Generated by generate_summary_tables.py from live *_results.json files where available,")
    print("transcribed test.sh constants otherwise (see this file's module docstring).")
    section_main_comparison()
    section_hp_sweep()
    section_fov_confound()
    section_film_breakdown()
    section_cldice_ablation()
    section_vascular_graph()
    section_multiseed()


if __name__ == "__main__":
    main()
