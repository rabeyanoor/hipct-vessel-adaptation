# train_real_gt.py
"""
Trains HiPVesselAdaptModel's PhysicsAwareAdapter + mask_decoder (frozen
image_encoder -- NOT full/whole-backbone fine-tuning, despite this module's
name) against the REAL expert-annotated SenNet+HOA kidney GT -- a controlled
baseline to compare against Plain LoRA (train_plain_lora.py) and
(eventually) Spacing-Conditioned LoRA on the SAME real-GT data, same frozen
backbone, same trainable-parameter budget.

Resumable, checkpoints every epoch (this machine's background jobs get
OOM-killed unpredictably by other processes sharing it -- see README).
"""

import json
import os

import torch
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG
from losses.combined_loss import HiPVesselLoss
from models.hip_vessel_model import HiPVesselAdaptModel
from utils.real_gt_dataset import RealGTKidneyDataset

CHECKPOINT_DIR = "checkpoints_real_gt"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_real_gt.pth")
STATE_PATH = os.path.join(CHECKPOINT_DIR, "train_state.json")


def train_real_gt(resume=True, extra_epochs=10, samples_per_epoch=8,
                   sample_ids=None, exclude_for_ood=("kidney_1_voi",), raw_tiff_root=None):
    """
    sample_ids: which real-GT kidney samples to train on. Defaults to all
        EXCEPT `exclude_for_ood` (kidney_1_voi, the 5.2um held-out specimen for
        the planned resolution-OOD test) so that specimen stays genuinely unseen.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}", flush=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if sample_ids is None:
        from utils.kidney_metadata_verification import VERIFIED_SAMPLES
        all_train_ids = [s[0] for s in VERIFIED_SAMPLES if s[5] == "train" and s[4] > 0]
        sample_ids = [sid for sid in all_train_ids if sid not in exclude_for_ood]
    print(f"[*] Training samples: {sample_ids} (held out for OOD test: {exclude_for_ood})", flush=True)

    dataset = RealGTKidneyDataset(sample_ids=sample_ids, patch_size=TRAIN_CONFIG["patch_size"],
                                   samples_per_epoch=samples_per_epoch, raw_tiff_root=raw_tiff_root)
    if len(dataset) == 0:
        print("[!] No real-GT kidney memmaps found -- run utils/convert_kidney_gt_to_memmap.py first.")
        return
    loader = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)

    model = HiPVesselAdaptModel().to(device)
    criterion = HiPVesselLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=TRAIN_CONFIG["lr"])

    completed_epochs = 0
    if resume and os.path.exists(BEST_MODEL_PATH) and os.path.exists(STATE_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        with open(STATE_PATH) as f:
            completed_epochs = json.load(f).get("completed_epochs", 0)
        print(f"[*] Resuming from checkpoint: {completed_epochs} epochs already completed", flush=True)
    else:
        print("[*] Starting fresh", flush=True)

    target_epochs = completed_epochs + extra_epochs
    print(f"[*] Trainable parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(f"[*] Training epochs {completed_epochs + 1} to {target_epochs}...", flush=True)
    model.train()

    for epoch in range(completed_epochs + 1, target_epochs + 1):
        total_loss = 0.0
        n_empty_skipped = 0
        n_steps = 0
        for step, batch in enumerate(loader):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            spacings = batch["spacing"].to(device)
            sample_id = batch["sample_id"][0]

            # Skip empty-target patches for the same reason evaluate_real_gt.py
            # does: real vasculature is sparse enough that most random 64^3
            # patches contain none, and training on an all-background target
            # every time would just reinforce predicting nothing.
            if targets.sum().item() == 0:
                n_empty_skipped += 1
                continue

            optimizer.zero_grad()
            mask_pred, collapse_pred, logits = model(images, spacings)
            loss = criterion(mask_pred, targets, spacings[0], logits=logits)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            print(f"  Batch [{step+1}/{len(loader)}] - Sample: {sample_id} - Step Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / max(1, n_steps)
        print(f"[✓] Epoch [{epoch}/{target_epochs}] Complete - Avg Loss: {avg_loss:.4f} "
              f"({n_empty_skipped} empty-target patches skipped)", flush=True)

        torch.save(model.state_dict(), BEST_MODEL_PATH)
        with open(STATE_PATH, "w") as f:
            json.dump({"completed_epochs": epoch}, f)
        print(f"[✓] Checkpoint saved at epoch {epoch} -> {BEST_MODEL_PATH}", flush=True)

    print(f"[✓] Training complete: {target_epochs} total epochs.", flush=True)


if __name__ == "__main__":
    train_real_gt()
