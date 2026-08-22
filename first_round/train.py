# train.py
import torch
import json
import os
from torch.utils.data import DataLoader
from config import ORGAN_CONFIG, TRAIN_CONFIG
from models.hip_vessel_model import HiPVesselAdaptModel
from losses.combined_loss import HiPVesselLoss
from utils.dataset import MultiOrganHiPCTDataset

STATE_PATH = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "train_state.json")


def train(resume=True, extra_epochs=None):
    """
    resume: continue from the saved checkpoint/epoch count if they exist
        (jobs on this machine sometimes get OOM-killed).
    extra_epochs: epochs to run in this call (default TRAIN_CONFIG["num_epochs"]).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}", flush=True)

    os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

    dataset = MultiOrganHiPCTDataset(config=ORGAN_CONFIG, patch_size=TRAIN_CONFIG["patch_size"],
                                      samples_per_epoch=TRAIN_CONFIG["samples_per_epoch"])
    if len(dataset) == 0:
        print("[!] No converted NIfTI volumes found in data/. Please run 'python utils/convert_jp2.py' first.")
        return

    loader = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)

    model = HiPVesselAdaptModel().to(device)
    criterion = HiPVesselLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=TRAIN_CONFIG["lr"])

    completed_epochs = 0
    best_path = TRAIN_CONFIG["best_model_path"]
    if resume and os.path.exists(best_path) and os.path.exists(STATE_PATH):
        model.load_state_dict(torch.load(best_path, map_location=device))
        with open(STATE_PATH) as f:
            completed_epochs = json.load(f).get("completed_epochs", 0)
        print(f"[*] Resuming from checkpoint: {completed_epochs} epochs already completed", flush=True)
    else:
        print("[*] Starting fresh (no resume state found or resume=False)", flush=True)

    epochs_to_run = extra_epochs if extra_epochs is not None else TRAIN_CONFIG["num_epochs"]
    target_epochs = completed_epochs + epochs_to_run

    print(f"[*] Trainable parameters: {sum(p.numel() for p in trainable_params)}", flush=True)
    print(f"[*] Training epochs {completed_epochs + 1} to {target_epochs}...", flush=True)
    model.train()

    for epoch in range(completed_epochs + 1, target_epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            spacings = batch["spacing"].to(device)
            organs = batch["organ"]

            optimizer.zero_grad()
            mask_pred, collapse_pred, logits = model(images, spacings)

            loss = criterion(mask_pred, targets, spacings[0], logits=logits)

            if not loss.requires_grad:
                loss.requires_grad_(True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"  Batch [{step+1}/{len(loader)}] - Organ: {organs[0]} - Step Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / max(1, len(loader))
        print(f"[✓] Epoch [{epoch}/{target_epochs}] Complete - Avg Loss: {avg_loss:.4f}", flush=True)

        # save every epoch so a crash doesn't lose the whole run
        torch.save(model.state_dict(), best_path)
        with open(STATE_PATH, "w") as f:
            json.dump({"completed_epochs": epoch}, f)
        print(f"[✓] Checkpoint saved at epoch {epoch} -> {best_path}", flush=True)

    print(f"[✓] Training complete: {target_epochs} total epochs.", flush=True)


if __name__ == "__main__":
    train()
