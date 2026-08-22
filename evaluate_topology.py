# evaluate_topology.py
"""
Betti numbers of the predictions vs. the Frangi pseudo-labels, per organ.

b0 = connected components, b1 = loops, for prediction and target plus the
difference. Big differences mean the vessel tree is broken up or has extra
loops, which Dice doesn't show.
"""

import torch
from torch.utils.data import DataLoader
from config import ORGAN_CONFIG, TRAIN_CONFIG
from models.hip_vessel_model import HiPVesselAdaptModel
from utils.dataset import MultiOrganHiPCTDataset
from utils.topology_metrics import compute_betti_numbers


def evaluate_topology(checkpoint_path="__default__", samples_per_organ=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Topology (Betti Number) Evaluation on device: {device}")

    model = HiPVesselAdaptModel().to(device)
    if checkpoint_path == "__default__":
        checkpoint_path = TRAIN_CONFIG["best_model_path"]
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] Loaded fine-tuned model weights from {checkpoint_path}")
    model.eval()

    print("\n" + "=" * 90)
    print(f"{'Organ':<10} | {'Pred b0':<8} | {'Target b0':<10} | {'|Δb0|':<6} | "
          f"{'Pred b1':<8} | {'Target b1':<10} | {'|Δb1|':<6}")
    print("=" * 90)

    with torch.no_grad():
        for organ, meta in ORGAN_CONFIG.items():
            dataset = MultiOrganHiPCTDataset(config={organ: meta}, patch_size=TRAIN_CONFIG["patch_size"],
                                              samples_per_epoch=samples_per_organ)
            if len(dataset) == 0:
                continue
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            pred_b0s, pred_b1s, target_b0s, target_b1s = [], [], [], []
            for batch in loader:
                images = batch["image"].to(device)
                targets = batch["mask"].to(device)
                spacings = batch["spacing"].to(device)

                mask_pred, _, _ = model(images, spacings)
                binary_pred = (mask_pred > 0.5).float().cpu().numpy()[0, 0]
                binary_target = targets.cpu().numpy()[0, 0]

                pb0, pb1 = compute_betti_numbers(binary_pred)
                tb0, tb1 = compute_betti_numbers(binary_target)
                pred_b0s.append(pb0); pred_b1s.append(pb1)
                target_b0s.append(tb0); target_b1s.append(tb1)

            avg_pb0 = sum(pred_b0s) / len(pred_b0s)
            avg_pb1 = sum(pred_b1s) / len(pred_b1s)
            avg_tb0 = sum(target_b0s) / len(target_b0s)
            avg_tb1 = sum(target_b1s) / len(target_b1s)
            print(f"{organ.capitalize():<10} | {avg_pb0:<8.2f} | {avg_tb0:<10.2f} | "
                  f"{abs(avg_pb0-avg_tb0):<6.2f} | {avg_pb1:<8.2f} | {avg_tb1:<10.2f} | "
                  f"{abs(avg_pb1-avg_tb1):<6.2f}")

    print("=" * 90)
    print("[✓] Topology Evaluation Complete!")


if __name__ == "__main__":
    evaluate_topology()
