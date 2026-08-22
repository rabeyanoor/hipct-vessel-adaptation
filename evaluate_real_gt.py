# evaluate_real_gt.py
"""
Evaluates HiPVesselAdaptModel against the REAL expert-annotated SenNet+HOA
kidney ground truth (utils/real_gt_dataset.py) -- separate from evaluate.py,
which evaluates against the Frangi pseudo-labels for Heart/Lung/Kidney.

Reports Dice, IoU, clDice, HD95, ASSD, foreground statistics, parameter counts,
and per-patch inference time. Supports checkpoint_path=None for a zero-shot
(no fine-tuning) baseline, matching baseline_zeroshot.py's convention.
"""

import os
import time

import torch
import monai.metrics as metrics
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG
from models.hip_vessel_model import HiPVesselAdaptModel
from utils.real_gt_dataset import RealGTKidneyDataset
from utils.load_cldice import load_soft_cldice

soft_cldice = load_soft_cldice()


def _safe_metric(metric_obj, pred, target):
    try:
        metric_obj.reset()
        metric_obj(pred, target)
        val = metric_obj.aggregate().item()
        return val if val == val else None  # filter NaN
    except Exception:
        return None


def evaluate_real_gt(checkpoint_path="__default__", label="Real-GT Kidney Evaluation",
                      sample_ids=None, samples_per_sample=5, patch_size=None,
                      model_class=HiPVesselAdaptModel, raw_tiff_root=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = patch_size or TRAIN_CONFIG["patch_size"]
    print(f"[*] Running {label} on device: {device}")

    model = model_class().to(device)
    if checkpoint_path == "__default__":
        checkpoint_path = TRAIN_CONFIG["best_model_path"]
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] Loaded fine-tuned model weights from {checkpoint_path}")
    elif checkpoint_path:
        print(f"[!] Warning: checkpoint not found at {checkpoint_path}, using initialized weights.")
    else:
        print("[*] Zero-shot: pretrained SAM-Med3D backbone, NO fine-tuning.")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dice_metric = metrics.DiceMetric(include_background=False, reduction="mean")
    iou_metric = metrics.MeanIoU(include_background=False, reduction="mean")
    hd95_metric = metrics.HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    assd_metric = metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
    cldice_fn = soft_cldice()

    dataset = RealGTKidneyDataset(sample_ids=sample_ids, patch_size=patch_size,
                                   samples_per_epoch=samples_per_sample, raw_tiff_root=raw_tiff_root)
    if len(dataset) == 0:
        print("[!] No real-GT kidney data found -- run utils/convert_kidney_gt_to_memmap.py first.")
        return None
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_scores, iou_scores, hd95_scores, assd_scores, cldice_scores = [], [], [], [], []
    fg_fractions, inference_times = [], []
    n_empty_target_patches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            spacings = batch["spacing"].to(device)

            t0 = time.time()
            outputs = model(images, spacings)
            mask_pred, logits = outputs[0], outputs[-1]  # supports both 2-tuple and 3-tuple model returns
            inference_times.append(time.time() - t0)

            binary_pred = (mask_pred > 0.5).float()
            fg_fractions.append(targets.mean().item())

            # Real kidney vasculature is sparse enough that most random 64^3
            # patches contain NO annotated vessel at all -- verified empirically
            # (12/15 test patches had zero target foreground). Segmentation-
            # accuracy and topology metrics (clDice, HD95, ASSD) are mathematically
            # undefined or misleadingly favorable on an empty target (e.g. clDice
            # was observed to silently return ~0.999 regardless of the prediction
            # when the target is empty), so they are only meaningful -- and only
            # averaged here -- over patches that actually contain real vessel.
            if targets.sum().item() == 0:
                n_empty_target_patches += 1
                continue

            d = _safe_metric(dice_metric, binary_pred, targets)
            i = _safe_metric(iou_metric, binary_pred, targets)
            h = _safe_metric(hd95_metric, binary_pred, targets)
            a = _safe_metric(assd_metric, binary_pred, targets)
            # soft_cldice.forward(y_true, y_pred) returns 1-clDice (a LOSS, same
            # convention as soft_dice() in the same file) -- convert to the actual
            # score, and pass args in the documented (y_true, y_pred) order.
            cl = 1.0 - cldice_fn(targets, binary_pred).item()

            if d is not None: dice_scores.append(d)
            if i is not None: iou_scores.append(i)
            if h is not None: hd95_scores.append(h)
            if a is not None: assd_scores.append(a)
            cldice_scores.append(cl)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print("\n" + "=" * 80)
    print(f"{label}")
    print("=" * 80)
    print(f"  Samples evaluated       : {len(loader)}")
    print(f"  Empty-target patches    : {n_empty_target_patches}/{len(loader)} "
          f"(excluded from Dice/IoU/clDice/HD95/ASSD below -- undefined/misleading on empty GT)")
    print(f"  Mean target FG fraction : {_avg(fg_fractions)*100:.4f}% (over ALL patches, incl. empty)")
    print(f"  Dice                    : {_avg(dice_scores):.4f}  (n={len(dice_scores)})")
    print(f"  IoU                     : {_avg(iou_scores):.4f}  (n={len(iou_scores)})")
    print(f"  clDice                  : {_avg(cldice_scores):.4f}  (n={len(cldice_scores)})")
    print(f"  HD95 (voxels)           : {_avg(hd95_scores):.4f}  (n={len(hd95_scores)})")
    print(f"  ASSD (voxels)           : {_avg(assd_scores):.4f}  (n={len(assd_scores)})")
    print(f"  Total parameters        : {total_params:,}")
    print(f"  Trainable parameters    : {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Mean inference time     : {_avg(inference_times)*1000:.1f} ms/patch (CPU)")
    print("=" * 80)

    return {
        "dice": _avg(dice_scores), "iou": _avg(iou_scores), "cldice": _avg(cldice_scores),
        "hd95": _avg(hd95_scores), "assd": _avg(assd_scores),
        "total_params": total_params, "trainable_params": trainable_params,
        "mean_inference_ms": _avg(inference_times) * 1000,
    }


if __name__ == "__main__":
    evaluate_real_gt()
