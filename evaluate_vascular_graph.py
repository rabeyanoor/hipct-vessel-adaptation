# evaluate_vascular_graph.py
"""
Vascular graph validation: skeletonize the predicted mask and the kidney GT,
build a networkx graph for each (utils/vascular_graph_metrics.py) and print
components, branch points, endpoints and total length side by side. Shows
fragmentation / extra branching that Dice and Betti numbers don't show well.
"""

import os

import torch
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG
from models.hip_vessel_model import HiPVesselAdaptModel
from utils.real_gt_dataset import RealGTKidneyDataset
from utils.vascular_graph_metrics import compute_vascular_graph_stats


def evaluate_vascular_graph(checkpoint_path="__default__", sample_ids=None,
                             samples_per_sample=5, patch_size=None,
                             model_class=HiPVesselAdaptModel, raw_tiff_root=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = patch_size or TRAIN_CONFIG["patch_size"]
    print(f"[*] Running Vascular Graph Validation on device: {device}")

    model = model_class().to(device)
    if checkpoint_path == "__default__":
        checkpoint_path = TRAIN_CONFIG["best_model_path"]
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] Loaded fine-tuned model weights from {checkpoint_path}")
    elif checkpoint_path:
        print(f"[!] Warning: checkpoint not found at {checkpoint_path}, using initialized weights.")
    model.eval()

    dataset = RealGTKidneyDataset(sample_ids=sample_ids, patch_size=patch_size,
                                   samples_per_epoch=samples_per_sample, raw_tiff_root=raw_tiff_root)
    if len(dataset) == 0:
        print("[!] No real-GT kidney data found -- run utils/convert_kidney_gt_to_memmap.py first.")
        return None
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rows = []
    n_empty_target_patches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            spacings = batch["spacing"].to(device)

            # skip empty targets, same as evaluate_real_gt.py
            if targets.sum().item() == 0:
                n_empty_target_patches += 1
                continue

            outputs = model(images, spacings)
            mask_pred = outputs[0]
            binary_pred = (mask_pred > 0.5).float().cpu().numpy()[0, 0]
            binary_target = targets.cpu().numpy()[0, 0]
            spacing_um = tuple(spacings[0].cpu().tolist())

            pred_stats = compute_vascular_graph_stats(binary_pred, spacing=spacing_um)
            target_stats = compute_vascular_graph_stats(binary_target, spacing=spacing_um)
            rows.append((batch["sample_id"][0], pred_stats, target_stats))

    if not rows:
        print(f"[!] No non-empty target patches found (all {n_empty_target_patches} were empty).")
        return None

    print("\n" + "=" * 118)
    print(f"{'Sample':<16} | {'Pred comp':<9} | {'GT comp':<7} | {'Pred branch':<11} | "
          f"{'GT branch':<9} | {'Pred endp':<9} | {'GT endp':<7} | {'Pred len(um)':<13} | {'GT len(um)':<10}")
    print("=" * 118)
    for sample_id, p, t in rows:
        print(f"{sample_id:<16} | {p['n_components']:<9} | {t['n_components']:<7} | "
              f"{p['n_branch_points']:<11} | {t['n_branch_points']:<9} | "
              f"{p['n_endpoints']:<9} | {t['n_endpoints']:<7} | "
              f"{p['total_length_um']:<13.1f} | {t['total_length_um']:<10.1f}")
    print("=" * 118)
    print(f"Empty-target patches excluded: {n_empty_target_patches}/{len(loader)}")
    print("=" * 118)
    print("[✓] Vascular Graph Validation Complete!")

    return rows


if __name__ == "__main__":
    evaluate_vascular_graph()
