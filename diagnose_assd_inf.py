# diagnose_assd_inf.py
"""
Root-causes the still-open ASSD=inf finding from the ORIGINAL 10-epoch CPU
run (test.sh section 3.2: PhysicsAwareAdapter and Spacing-Conditioned LoRA
both hit ASSD=inf; Plain LoRA on the identical protocol did not). baki.sh
section 6 records this as still only partially checked: the "completely
empty prediction" hypothesis was tested on 15 patches and REJECTED (0
confirmed) -- so the real cause was still unknown before this script.

Runs each of the two affected 10-epoch CPU checkpoints over many real-GT
patches, calling MONAI's SurfaceDistanceMetric on EACH non-empty-target
patch individually (not just the averaged run evaluate_real_gt.py does), and
records prediction/target foreground fraction alongside the per-patch ASSD
value or raised exception. Tests a second, more specific hypothesis MONAI's
own surface-distance code makes plausible: a FULLY SATURATED prediction
(every voxel in the patch predicted foreground, not zero voxels) has no
detectable surface after erosion either -- same "no boundary to measure
distance from" failure mode as a completely empty mask, but the opposite
extreme, so the earlier all-zero check would have missed it entirely.

Text-only output, no plotting, per project rule.
"""

import warnings

import torch
from torch.utils.data import DataLoader

import monai.metrics as metrics
from models.hip_vessel_model import HiPVesselAdaptModel
from models.spacing_lora_model import SpacingConditionedLoRAModel
from utils.real_gt_dataset import RealGTKidneyDataset

CONFIGS = [
    ("PhysicsAwareAdapter (10ep CPU)", "checkpoints/best_model.pth", HiPVesselAdaptModel),
    ("Spacing-Conditioned LoRA (10ep CPU)", "checkpoints_spacing_lora/best_model_spacing_lora.pth",
     SpacingConditionedLoRAModel),
]


def diagnose(label, checkpoint_path, model_class, samples_per_sample=100, patch_size=(64, 64, 64)):
    device = torch.device("cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = RealGTKidneyDataset(patch_size=patch_size, samples_per_epoch=samples_per_sample)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    assd_metric = metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")

    rows = []
    n_total_voxels = patch_size[0] * patch_size[1] * patch_size[2]
    n_empty_target = 0

    with torch.no_grad():
        for batch in loader:
            images, targets, spacings = batch["image"], batch["mask"], batch["spacing"]
            if targets.sum().item() == 0:
                n_empty_target += 1
                continue

            outputs = model(images, spacings)
            binary_pred = (outputs[0] > 0.5).float()
            pred_sum = int(binary_pred.sum().item())
            target_sum = int(targets.sum().item())

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    assd_metric.reset()
                    assd_metric(binary_pred, targets)
                    val = assd_metric.aggregate().item()
                    warned = "; ".join(str(w.message) for w in caught)
                except Exception as e:
                    val = None
                    warned = f"EXCEPTION: {e}"

            rows.append({
                "sample_id": batch["sample_id"][0],
                "pred_sum": pred_sum, "pred_fraction": pred_sum / n_total_voxels,
                "pred_fully_saturated": pred_sum == n_total_voxels,
                "pred_completely_empty": pred_sum == 0,
                "target_sum": target_sum, "target_fraction": target_sum / n_total_voxels,
                "assd": val, "assd_is_inf": (val is not None and val == float("inf")),
                "warning": warned,
            })

    print("\n" + "=" * 100)
    print(f"{label} -- {checkpoint_path}")
    print("=" * 100)
    print(f"Non-empty-target patches evaluated: {len(rows)} (excluded {n_empty_target} empty-target patches)")

    inf_rows = [r for r in rows if r["assd_is_inf"]]
    finite_rows = [r for r in rows if not r["assd_is_inf"] and r["assd"] is not None]
    print(f"ASSD=inf patches: {len(inf_rows)}/{len(rows)}  |  ASSD finite: {len(finite_rows)}/{len(rows)}")

    if inf_rows:
        print(f"\n{'Sample':<16} | {'pred_sum':<9} | {'pred_frac':<10} | {'fully_sat':<10} | "
              f"{'empty_pred':<11} | {'target_sum':<10} | {'target_frac':<11} | warning")
        print("-" * 100)
        for r in inf_rows:
            print(f"{r['sample_id']:<16} | {r['pred_sum']:<9} | {r['pred_fraction']:<10.4f} | "
                  f"{str(r['pred_fully_saturated']):<10} | {str(r['pred_completely_empty']):<11} | "
                  f"{r['target_sum']:<10} | {r['target_fraction']:<11.4f} | {r['warning'][:40]}")

        n_saturated = sum(1 for r in inf_rows if r["pred_fully_saturated"])
        n_empty = sum(1 for r in inf_rows if r["pred_completely_empty"])
        n_neither = len(inf_rows) - n_saturated - n_empty
        print(f"\nOf {len(inf_rows)} ASSD=inf patches: {n_saturated} fully-saturated prediction, "
              f"{n_empty} completely-empty prediction, {n_neither} neither (some other cause).")
    else:
        print("No ASSD=inf patches occurred in this run (random patch draw -- try a larger "
              "samples_per_sample if this was expected to reproduce it).")

    if finite_rows:
        max_pred_frac_finite = max(r["pred_fraction"] for r in finite_rows)
        min_pred_frac_finite = min(r["pred_fraction"] for r in finite_rows)
        print(f"\nFor comparison, among FINITE-ASSD patches: pred_fraction ranges "
              f"{min_pred_frac_finite:.4f}-{max_pred_frac_finite:.4f}")

    return rows


if __name__ == "__main__":
    all_results = {}
    for label, ckpt, cls in CONFIGS:
        all_results[label] = diagnose(label, ckpt, cls)

    print("\n" + "=" * 100)
    print("CROSS-CONFIG SUMMARY")
    print("=" * 100)
    for label, rows in all_results.items():
        inf_rows = [r for r in rows if r["assd_is_inf"]]
        if not inf_rows:
            print(f"{label}: no ASSD=inf reproduced this run")
            continue
        n_saturated = sum(1 for r in inf_rows if r["pred_fully_saturated"])
        n_empty = sum(1 for r in inf_rows if r["pred_completely_empty"])
        print(f"{label}: {len(inf_rows)} inf patches -- {n_saturated} fully-saturated, "
              f"{n_empty} completely-empty, {len(inf_rows) - n_saturated - n_empty} neither")
