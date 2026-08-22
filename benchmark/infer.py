# benchmark/infer.py
"""
Runs one of our models on the benchmark ROIs and writes binary predictions
for benchmark/score.py.

Sliding-window inference (MONAI, Gaussian blending). Each window is z-scored
on its own, the same way the training patches are.

Two switches make the spacing / field-of-view experiment possible:
  --resample-um X    resample the ROI to X um before inference and resample
                     the logits back to the native grid afterwards (X=50 is
                     the "FOV-matched" setting: every window covers the same
                     tissue as a training patch).
  --spacing-um X     spacing value given to the model. Default: the spacing of
                     the grid the model actually sees (native, or X from
                     --resample-um). Setting it on purpose to the wrong value
                     separates the effect of the spacing input from the
                     effect of the field of view.

Other options: --tta (average over flips of the three axes) and --save-prob
(also store the probabilities, needed by benchmark/postprocess.py).

Usage:
    python -m benchmark.infer --model spacing_lora --checkpoint ckpt.pth --run-name spacing_seed0
    python -m benchmark.infer ... --resample-um 50 --run-name spacing_seed0_fov50
    python -m benchmark.infer ... --spacing-um 50 --run-name spacing_seed0_tell50
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from benchmark.rois import ROI_DIR, load_manifest

PRED_ROOT = "data/benchmark/preds"


def build_model(name, checkpoint, refine=False, device="cpu", lora_r=4, spacing_encoding="raw"):
    if name == "plain_lora":
        from models.plain_lora_model import PlainLoRAModel
        model = PlainLoRAModel(refine=refine, lora_r=lora_r)
    elif name == "spacing_lora":
        from models.spacing_lora_model import SpacingConditionedLoRAModel
        model = SpacingConditionedLoRAModel(refine=refine, lora_r=lora_r, spacing_encoding=spacing_encoding)
    elif name == "physics_adapter":
        from models.hip_vessel_model import HiPVesselAdaptModel
        model = HiPVesselAdaptModel()
    else:
        raise ValueError(name)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def _predictor(model, spacing_um, device):
    def run(windows):
        mean = windows.mean(dim=(2, 3, 4), keepdim=True)
        std = windows.std(dim=(2, 3, 4), keepdim=True)
        windows = (windows - mean) / std.clamp_min(1e-8)
        sp = torch.full((windows.shape[0], 3), float(spacing_um), device=device)
        out = model(windows, sp)
        return out[-1]  # logits
    return run


@torch.no_grad()
def predict_roi(model, image, native_um, patch, device, resample_um=None, spacing_um=None,
                overlap=0.5, sw_batch_size=2, amp=True, tta=False, return_prob=False):
    """Binary prediction (uint8) on the native grid of `image`; with
    return_prob=True the vessel probability (float32) instead."""
    x = torch.from_numpy(image.astype(np.float32))[None, None].to(device)
    grid_um = native_um
    if resample_um is not None and abs(resample_um - native_um) > 1e-6:
        size = [max(8, int(round(s * native_um / resample_um))) for s in image.shape]
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        grid_um = resample_um
    # pad to at least one window
    pads = []
    for s in reversed(x.shape[2:]):
        pads += [0, max(0, patch - s)]
    if any(pads):
        x = F.pad(x, pads, mode="replicate")
    sp = grid_um if spacing_um is None else spacing_um
    flips = [()] + ([(2,), (3,), (4,), (2, 3, 4)] if tta else [])
    prob = None
    for f in flips:
        xi = torch.flip(x, f) if f else x
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            out = sliding_window_inference(xi, (patch,) * 3, sw_batch_size, _predictor(model, sp, device),
                                           overlap=overlap, mode="gaussian")
        out = torch.sigmoid(out.float())
        out = torch.flip(out, f) if f else out
        prob = out if prob is None else prob + out
    prob = prob / len(flips)
    # back to logits so the resampling below works the same way as before
    logits = torch.logit(prob.clamp(1e-6, 1 - 1e-6))
    if resample_um is not None and abs(resample_um - native_um) > 1e-6:
        valid = [max(8, int(round(s * native_um / resample_um))) for s in image.shape]
        logits = logits[:, :, :valid[0], :valid[1], :valid[2]]
        logits = F.interpolate(logits, size=image.shape, mode="trilinear", align_corners=False)
    else:
        logits = logits[:, :, :image.shape[0], :image.shape[1], :image.shape[2]]
    if return_prob:
        return torch.sigmoid(logits[0, 0]).cpu().numpy()
    return (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()


def run(model_name, checkpoint, run_name, refine=False, patch=64, resample_um=None, spacing_um=None,
        sets=None, roi_dir=ROI_DIR, pred_root=PRED_ROOT, overlap=0.5, device=None, tta=False,
        save_prob=False, lora_r=4, spacing_encoding="raw", model=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = build_model(model_name, checkpoint, refine, device, lora_r, spacing_encoding)
    manifest = load_manifest(roi_dir)
    out_dir = os.path.join(pred_root, run_name)
    log = {"model": model_name, "checkpoint": checkpoint, "refine": refine, "patch": patch,
           "spacing_encoding": spacing_encoding,
           "resample_um": resample_um, "spacing_um": spacing_um, "overlap": overlap, "tta": tta,
           "times_s": {}}
    for sid, entries in manifest.items():
        if sets and sid not in sets:
            continue
        os.makedirs(os.path.join(out_dir, sid), exist_ok=True)
        for e in entries:
            img = np.load(e["path"])["image"]
            t0 = time.time()
            prob = predict_roi(model, img, e["spacing_um"], patch, device, resample_um, spacing_um, overlap,
                               tta=tta, return_prob=True)
            pred = (prob > 0.5).astype(np.uint8)
            log["times_s"][f"{sid}/roi_{e['roi']:02d}"] = round(time.time() - t0, 2)
            if save_prob:
                # probabilities stored as 0-255 to keep the files small
                np.savez_compressed(os.path.join(out_dir, sid, f"roi_{e['roi']:02d}.npz"), pred=pred,
                                    prob=np.round(prob * 255).astype(np.uint8))
            else:
                np.savez_compressed(os.path.join(out_dir, sid, f"roi_{e['roi']:02d}.npz"), pred=pred)
            print(f"  {run_name} {sid} roi_{e['roi']:02d}: {time.time() - t0:.1f}s, "
                  f"pred fg={100 * pred.mean():.3f}%")
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(log, f, indent=1)
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["plain_lora", "spacing_lora", "physics_adapter"])
    p.add_argument("--checkpoint", default="", help="empty: no fine-tuned weights (SAM-Med3D as released)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--resample-um", type=float, default=None)
    p.add_argument("--spacing-um", type=float, default=None)
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--roi-dir", default=ROI_DIR)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--save-prob", action="store_true")
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--spacing-encoding", choices=["raw", "log"], default="raw")
    p.add_argument("--pred-root", default=PRED_ROOT)
    a = p.parse_args()
    run(a.model, a.checkpoint or None, a.run_name, a.refine, a.patch, a.resample_um, a.spacing_um,
        a.sets, a.roi_dir, pred_root=a.pred_root, overlap=a.overlap, tta=a.tta, save_prob=a.save_prob,
        lora_r=a.lora_r, spacing_encoding=a.spacing_encoding)
