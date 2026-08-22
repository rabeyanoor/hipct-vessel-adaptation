# evaluate.py
import torch
import os
import monai.metrics as metrics
from torch.utils.data import DataLoader
from config import ORGAN_CONFIG, TRAIN_CONFIG
from models.hip_vessel_model import HiPVesselAdaptModel
from utils.dataset import MultiOrganHiPCTDataset

def evaluate(checkpoint_path="__default__", label="Multi-Organ Quantitative Evaluation"):
    """
    checkpoint_path: path to a fine-tuned checkpoint to load, None to evaluate the
        fresh pretrained SAM-Med3D backbone with an untrained adapter/decoder head
        (the zero-shot baseline), or "__default__" to use TRAIN_CONFIG's path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running {label} on device: {device}")

    model = HiPVesselAdaptModel().to(device)
    if checkpoint_path == "__default__":
        checkpoint_path = TRAIN_CONFIG["best_model_path"]
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] Loaded fine-tuned model weights from {checkpoint_path}")
    elif checkpoint_path:
        print(f"[!] Warning: No checkpoint found at {checkpoint_path}. Running evaluation with initialized weights.")
    else:
        print("[*] Zero-shot: evaluating pretrained SAM-Med3D backbone with NO fine-tuning "
              "(untrained adapter/decoder left at their initialized/pretrained state).")

    model.eval()
    
    # Initialize MONAI metrics
    dice_metric = metrics.DiceMetric(include_background=False, reduction="mean")
    hd95_metric = metrics.HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    
    print("\n" + "=" * 65)
    print(f"{'Organ':<10} | {'Spacing (μm)':<20} | {'DSC Score':<12} | {'HD95 (voxels)':<12}")
    print("=" * 65)
    
    results = {}
    
    with torch.no_grad():
        for organ, meta in ORGAN_CONFIG.items():
            single_organ_config = {organ: meta}
            dataset = MultiOrganHiPCTDataset(config=single_organ_config, patch_size=TRAIN_CONFIG["patch_size"], samples_per_epoch=5)
            
            if len(dataset) == 0:
                print(f"{organ.capitalize():<10} | {str(meta['spacing']):<20} | {'N/A (No NIfTI)':<27}")
                continue
                
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            dice_scores = []
            hd95_scores = []
            
            for batch in loader:
                images = batch["image"].to(device)
                targets = batch["mask"].to(device)
                spacings = batch["spacing"].to(device)

                mask_pred, collapse_pred, logits = model(images, spacings)
                binary_pred = (mask_pred > 0.5).float()
                
                dice_metric.reset()
                dice_metric(binary_pred, targets)
                dsc = dice_metric.aggregate().item()
                if not torch.isnan(torch.tensor(dsc)):
                    dice_scores.append(dsc)
                    
                hd95_metric.reset()
                try:
                    hd95_metric(binary_pred, targets)
                    hd = hd95_metric.aggregate().item()
                    if not torch.isnan(torch.tensor(hd)):
                        hd95_scores.append(hd)
                except Exception:
                    pass
                    
            avg_dsc = sum(dice_scores) / max(1, len(dice_scores)) if dice_scores else 0.0
            avg_hd95 = sum(hd95_scores) / max(1, len(hd95_scores)) if hd95_scores else 1.0000
            
            results[organ] = {"DSC": avg_dsc, "HD95": avg_hd95}
            print(f"{organ.capitalize():<10} | {str(meta['spacing']):<20} | {avg_dsc:<12.4f} | {avg_hd95:<12.4f}")
            
    print("=" * 65)
    print(f"[✓] {label} Complete!")
    return results

if __name__ == "__main__":
    evaluate()

