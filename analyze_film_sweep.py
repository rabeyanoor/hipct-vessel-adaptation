# analyze_film_sweep.py
"""
FiLM output (gamma, beta) of a Spacing-Conditioned LoRA checkpoint over a range
of spacings, read directly from the state dict (no SAM-Med3D build needed).

For each spacing and each of the 24 adapted layers it reports mean |gamma - 1|,
max |gamma - 1| and mean |beta|, i.e. how far the LoRA branch is from the
unmodulated (plain LoRA) update. With the "raw" spacing input of the first
experiments the FiLM output grows almost linearly with the spacing value; this
script shows that directly.

Usage:
    python analyze_film_sweep.py --checkpoint ckpt.pth --encoding raw --output-json film_sweep.json
"""

import argparse
import json
import re

import torch
import torch.nn.functional as F

from models.spacing_lora import encode_spacing

DEFAULT_SPACINGS = [0.0, 1.0, 2.6, 5.2, 6.5, 13.0, 25.0, 50.0, 50.16, 50.28, 63.08, 100.0]


def film_layers(state_dict):
    prefixes = sorted({k.rsplit(".spacing_film", 1)[0] for k in state_dict if ".spacing_film." in k},
                      key=lambda k: (int(re.search(r"blocks\.(\d+)\.", k).group(1)), k.endswith("proj")))
    return prefixes


def film(state_dict, prefix, spacing_um, encoding):
    w0, b0 = state_dict[prefix + ".spacing_film.0.weight"].float(), state_dict[prefix + ".spacing_film.0.bias"].float()
    w2, b2 = state_dict[prefix + ".spacing_film.2.weight"].float(), state_dict[prefix + ".spacing_film.2.bias"].float()
    s = torch.full((1, 3), float(spacing_um))
    x = encode_spacing(s, encoding) if spacing_um > 0 or encoding == "raw" else torch.full((1, 3), float("nan"))
    out = F.gelu(x @ w0.T + b0) @ w2.T + b2
    gamma, beta = out.chunk(2, dim=-1)
    return 1.0 + gamma[0], beta[0]


def sweep(checkpoint, encoding, spacings=DEFAULT_SPACINGS):
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    layers = film_layers(sd)
    if not layers:
        raise ValueError(f"{checkpoint}: no spacing_film weights (not a Spacing-Conditioned LoRA model?)")
    rows = []
    for s in spacings:
        if s <= 0 and encoding == "log":
            continue
        per_layer = []
        for i, p in enumerate(layers):
            g, b = film(sd, p, s, encoding)
            per_layer.append({"layer": i, "name": p.split("image_encoder.")[-1],
                              "mean_abs_gamma_minus_1": (g - 1).abs().mean().item(),
                              "max_abs_gamma_minus_1": (g - 1).abs().max().item(),
                              "gamma_min": g.min().item(), "gamma_max": g.max().item(),
                              "mean_abs_beta": b.abs().mean().item()})
        allg = torch.cat([(film(sd, p, s, encoding)[0] - 1).abs() for p in layers])
        allb = torch.cat([film(sd, p, s, encoding)[1].abs() for p in layers])
        rows.append({"spacing_um": s, "mean_abs_gamma_minus_1": allg.mean().item(),
                     "max_abs_gamma_minus_1": allg.max().item(), "mean_abs_beta": allb.mean().item(),
                     "per_layer": per_layer})
    return {"checkpoint": checkpoint, "encoding": encoding, "n_layers": len(layers), "sweep": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encoding", choices=["raw", "log"], default="raw")
    p.add_argument("--spacings", type=float, nargs="+", default=DEFAULT_SPACINGS)
    p.add_argument("--output-json", default=None)
    a = p.parse_args()
    res = sweep(a.checkpoint, a.encoding, a.spacings)
    print(f"{res['checkpoint']} ({res['encoding']} spacing input, {res['n_layers']} layers)")
    print(f"{'spacing_um':>10} | {'mean|g-1|':>9} | {'max|g-1|':>8} | {'mean|beta|':>10}")
    for r in res["sweep"]:
        print(f"{r['spacing_um']:>10.2f} | {r['mean_abs_gamma_minus_1']:>9.4f} | "
              f"{r['max_abs_gamma_minus_1']:>8.3f} | {r['mean_abs_beta']:>10.4f}")
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1)
