# analyze_film_layers.py
"""
Per-layer breakdown of SpacingConditionedLoRALinear's FiLM output
(gamma/beta), for a trained checkpoint, at two spacing values: the
in-distribution training spacing (50um) and the OOD kidney_1_voi spacing
(5.2um).

README.md/test.sh already report an AGGREGATE gamma/beta check across all
injected layers (the "FiLM-extrapolation" hypothesis -- refuted: OOD gamma
values were, if anything, tamer than in-distribution ones). This script
answers a narrower question the aggregate check can't: does any SINGLE
layer (out of the 24 injected into qkv/proj) produce an extreme gamma/beta
at the OOD spacing even though the mean across all 24 looks tame -- i.e. is
there a bad layer hiding inside a fine-looking average, rather than the
whole mechanism behaving uniformly.

No forward pass through the image encoder is needed for this -- spacing_film
is a small (3 -> spacing_hidden -> 2*out_features) MLP taking only the
spacing tensor, so it's evaluated directly per layer.
"""

import argparse
import json

import torch

from models.spacing_lora_model import SpacingConditionedLoRAModel


def film_stats(layer, spacing):
    with torch.no_grad():
        film = layer.spacing_film(spacing)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = 1.0 + gamma
    return {
        "gamma_mean": gamma.mean().item(), "gamma_std": gamma.std().item(),
        "gamma_min": gamma.min().item(), "gamma_max": gamma.max().item(),
        "beta_mean": beta.mean().item(), "beta_std": beta.std().item(),
        "beta_min": beta.min().item(), "beta_max": beta.max().item(),
    }


def analyze_film_layers(checkpoint_path, in_dist_spacing_um=50.0, ood_spacing_um=5.2, output_json=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpacingConditionedLoRAModel().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[✓] Loaded {checkpoint_path}")

    in_dist_spacing = torch.tensor([[in_dist_spacing_um] * 3], dtype=torch.float32, device=device)
    ood_spacing = torch.tensor([[ood_spacing_um] * 3], dtype=torch.float32, device=device)

    print("\n" + "=" * 100)
    print(f"Per-layer FiLM output: in-distribution ({in_dist_spacing_um}um) vs OOD ({ood_spacing_um}um)")
    print("=" * 100)
    header = (f"{'layer':>5} | {'gamma_mean(in)':>14} {'gamma_max(in)':>13} | "
              f"{'gamma_mean(ood)':>15} {'gamma_max(ood)':>14} | "
              f"{'beta_mean(in)':>13} {'beta_mean(ood)':>14} | {'|delta gamma_max|':>18}")
    print(header)
    print("-" * len(header))

    worst_layer, worst_delta = None, -1.0
    per_layer = []
    for idx, layer in enumerate(model.lora_layers):
        s_in = film_stats(layer, in_dist_spacing)
        s_ood = film_stats(layer, ood_spacing)
        delta = abs(s_ood["gamma_max"] - s_in["gamma_max"])
        if delta > worst_delta:
            worst_delta, worst_layer = delta, idx
        per_layer.append({"layer": idx, "in_dist": s_in, "ood": s_ood, "abs_delta_gamma_max": delta})

        print(f"{idx:>5} | {s_in['gamma_mean']:>14.4f} {s_in['gamma_max']:>13.4f} | "
              f"{s_ood['gamma_mean']:>15.4f} {s_ood['gamma_max']:>14.4f} | "
              f"{s_in['beta_mean']:>13.4f} {s_ood['beta_mean']:>14.4f} | {delta:>18.4f}")

    print("=" * 100)
    print(f"Largest in-dist -> OOD gamma_max shift: layer {worst_layer} (|delta|={worst_delta:.4f})")
    print("If this is close to the smallest shifts too, no single layer explains the "
          "OOD collapse better than the others -- consistent with the aggregate check "
          "in README.md/test.sh, which already refuted FiLM extrapolation as the cause.")

    if output_json:
        with open(output_json, "w") as f:
            json.dump({
                "checkpoint_path": checkpoint_path,
                "in_dist_spacing_um": in_dist_spacing_um, "ood_spacing_um": ood_spacing_um,
                "worst_layer": worst_layer, "worst_delta_gamma_max": worst_delta,
                "per_layer": per_layer,
            }, f, indent=2)
        print(f"[✓] Full {len(per_layer)}-layer breakdown written to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--in-dist-spacing-um", type=float, default=50.0)
    parser.add_argument("--ood-spacing-um", type=float, default=5.2)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    analyze_film_layers(args.checkpoint, args.in_dist_spacing_um, args.ood_spacing_um, args.output_json)
