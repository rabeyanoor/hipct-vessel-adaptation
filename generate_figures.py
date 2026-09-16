# generate_figures.py
"""
Generates the three real-data figures for the paper (PAPER_DRAFT.md) as PNG
files, at publication resolution (300 dpi). Reads only this project's own
result JSON files -- multiseed_seed*_results.json, film_layer_analysis.json,
vascular_graph_results.json -- plus MAIN_100EP_COMPARISON (imported from
generate_summary_tables.py, the same constants that script's own text tables
use, so the two never drift apart).

Scope, per feedback_no_plotting_in_repo.md (memory, superseded 2026-09-16):
this script is a deliberate, narrow exception to keeping the rest of the
codebase numeric/text-only -- ordinary evaluation/diagnostic scripts
(evaluate_real_gt.py, analyze_film_layers.py's own stdout, etc.) still don't
plot anything themselves; this is the one place chart-generation code lives,
and only for these three specific paper figures.

Two more figures already exist as mechanism diagrams (shared backbone/FiLM
branch/resolution setup) -- those are conceptual, not data, and were built as
a separate Claude Artifact rather than matplotlib; see PAPER_DRAFT.md's
figures section for that link. Two figures from the six-figure plan are not
yet built at all (qualitative prediction renders, FOV-confound bar chart).

Run: python3 generate_figures.py
Output: figures/figure4_dice_comparison.png, figures/figure5_film_layers.png,
figures/figure6_vascular_graph.png
"""

import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_summary_tables import MAIN_100EP_COMPARISON

OUTPUT_DIR = "figures"

# Same validated (colorblind-safe, checked with the dataviz skill's
# validate_palette.js, not eyeballed) pair used in the companion Artifact
# figures, so the whole figure set reads as one consistent system.
PLAIN_COLOR = "#2a78d6"
SPACING_COLOR = "#eb6834"
INK = "#1D2220"
MUTED = "#6A655C"
GRID = "#DEDACD"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def figure4_dice_comparison():
    """n-seed Dice comparison, in-distribution vs OOD, separate panels
    (different scales -- OOD is roughly 15x smaller, a shared axis would
    flatten it to nothing) with individual seed points over the mean+-std bar."""
    seed_files = sorted(glob.glob("multiseed_seed*_results.json"))
    series = {}
    for arch in ("plain_lora", "spacing_lora"):
        existing = MAIN_100EP_COMPARISON.get(arch, {})
        in_vals = [existing.get("in_dist_dice")]
        ood_vals = [existing.get("ood_dice")]
        for sf in seed_files:
            d = _load_json(sf).get(arch, {})
            in_vals.append(d.get("in_dist_dice"))
            ood_vals.append(d.get("ood_dice"))
        series[arch] = {
            "in_dist": [v for v in in_vals if v is not None],
            "ood": [v for v in ood_vals if v is not None],
        }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    labels = ["Plain LoRA", "Spacing-Conditioned\nLoRA"]
    colors = [PLAIN_COLOR, SPACING_COLOR]

    for ax, condition, title in zip(axes, ("in_dist", "ood"), ("In-distribution (50 μm)", "Out-of-distribution (5.2 μm)")):
        means, stds = [], []
        for arch in ("plain_lora", "spacing_lora"):
            vals = series[arch][condition]
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)

        x = [0, 1]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
               width=0.55, edgecolor=INK, linewidth=0.8, error_kw={"ecolor": INK, "elinewidth": 1.3})
        for xi, arch in zip(x, ("plain_lora", "spacing_lora")):
            vals = series[arch][condition]
            jitter = [-0.08, 0, 0.08][:len(vals)]
            ax.scatter([xi + j for j in jitter], vals, color=INK, s=22, zorder=5,
                       edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Dice")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(means[i] + stds[i] for i in range(2)) * 1.35 + 0.02)

    fig.suptitle("Figure 4 — In-distribution gain, out-of-distribution collapse (n=%d seeds)" % len(series["plain_lora"]["in_dist"]),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure4_dice_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] {out}")


def figure5_film_layers():
    """Dumbbell chart: gamma_max per layer, in-distribution vs OOD spacing,
    all 24 injected layers -- shows the identity-reversion is uniform, not
    one outlier layer hiding inside a fine-looking average."""
    data = _load_json("film_layer_analysis.json")
    if data is None:
        print("[!] film_layer_analysis.json not found -- run "
              "analyze_film_layers.py --checkpoint <path> --output-json film_layer_analysis.json first")
        return
    rows = sorted(data["per_layer"], key=lambda r: r["layer"])

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    layers = [r["layer"] for r in rows]
    in_vals = [r["in_dist"]["gamma_max"] for r in rows]
    ood_vals = [r["ood"]["gamma_max"] for r in rows]

    for y, gi, go in zip(layers, in_vals, ood_vals):
        ax.plot([go, gi], [y, y], color=MUTED, linewidth=1.2, zorder=1)
    ax.scatter(in_vals, layers, color=PLAIN_COLOR, s=45, zorder=3, label="γ_max at 50 μm (in-dist)", edgecolor="white", linewidth=0.5)
    ax.scatter(ood_vals, layers, color=SPACING_COLOR, s=45, zorder=3, label="γ_max at 5.2 μm (OOD)", edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1, label="γ = 1 (identity)")

    ax.set_yticks(layers)
    ax.set_yticklabels(layers, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("γ_max")
    ax.set_ylabel("injected layer")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("Figure 5 — FiLM γ reverts toward identity at the OOD spacing,\nuniformly across all 24 layers",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure5_film_layers.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] {out}")


def figure6_vascular_graph():
    """Predicted vs real-GT skeleton-graph medians, 4 conditions x 2 metrics --
    filled bar = predicted, outlined bar = real GT, same color = same model."""
    data = _load_json("vascular_graph_results.json")
    if data is None:
        print("[!] vascular_graph_results.json not found")
        return

    conditions = ["plain_lora_in_dist", "plain_lora_ood", "spacing_lora_in_dist", "spacing_lora_ood"]
    short_labels = ["Plain\nin-dist", "Plain\nOOD", "Spacing\nin-dist", "Spacing\nOOD"]
    bar_colors = [PLAIN_COLOR, PLAIN_COLOR, SPACING_COLOR, SPACING_COLOR]

    def median(cond, side, field):
        rows = data[cond]
        return statistics.median(row[side][field] for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    for ax, field, title in zip(axes, ("n_components", "n_branch_points"), ("Connected components (median)", "Branch points (median)")):
        x = range(len(conditions))
        pred = [median(c, "pred", field) for c in conditions]
        gt = [median(c, "gt", field) for c in conditions]
        w = 0.32
        ax.bar([xi - w / 2 - 0.02 for xi in x], pred, width=w, color=bar_colors, alpha=0.9,
               edgecolor=INK, linewidth=0.8)
        ax.bar([xi + w / 2 + 0.02 for xi in x], gt, width=w, facecolor="white",
               edgecolor=bar_colors, linewidth=1.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("median per patch")

    fig.suptitle("Figure 6 — OOD failure mode: fragmentation and spurious branching, not graceful shrinkage",
                 fontsize=12, fontweight="bold", y=0.99)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=INK, alpha=0.9, label="predicted"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, linewidth=1.8, label="real GT"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(OUTPUT_DIR, "figure6_vascular_graph.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] {out}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure4_dice_comparison()
    figure5_film_layers()
    figure6_vascular_graph()
    print(f"\n[✓] All figures written to {OUTPUT_DIR}/")
