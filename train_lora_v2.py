# train_lora_v2.py
"""
Training script for the second round of experiments (MedIA version).

Same models and loss as first_round/train_plain_lora.py and
first_round/train_spacing_lora.py, with the changes that came out of reading
the related work and checking the first results:
  - fixed number of optimiser steps per epoch (--iters), instead of skipping
    empty patches, which left only ~5 real steps per epoch before
  - foreground oversampling (--fg-prob), as in nnU-Net
  - spacing / field-of-view augmentation (--fov-scale lo hi), as in HyperSpace;
    with it the FiLM input actually varies during training
  - 128^3 patches (--patch 128), SAM-Med3D's native input size
  - optional full-resolution refinement head (--refine)
  - mixed precision on GPU, gradient clipping, cosine LR schedule
  - checkpoint every epoch; resumable
  - validation on held-out slices of kidney_1_dense (last 20 %, never used for
    training); the best model by validation Dice is kept as model_best.pth
  - optional extra capacity: higher LoRA rank (--lora-r), unfreezing the last
    encoder blocks (--unfreeze-blocks), or training only the decoder's output
    layers (--decoder output), as suggested by Asokan et al. (2024)

Test volumes (kidney_5, kidney_6, kidney_additional_voi_13um, kidney_1_voi)
are never used here. Validation ROIs are built once with
`python -m benchmark.rois --validation --raw-tiff-root <train folder>`. Evaluation is done afterwards with benchmark/infer.py
and benchmark/score.py.

Example (the main configuration):
    python train_lora_v2.py --model spacing --patch 128 --fg-prob 0.5 \
        --fov-scale 0.25 2.0 --refine --epochs 100 --iters 250 --seed 0 \
        --raw-tiff-root /kaggle/input/.../train --out checkpoints_v2/spacing_full_s0
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses.combined_loss import HiPVesselLoss
from utils.real_gt_dataset import RealGTKidneyDataset
from utils.slab_dataset import SlabPatchDataset

TRAIN_SAMPLES = ["kidney_1_dense", "kidney_2", "kidney_3_dense", "kidney_3_sparse"]
# slices kept out of training for validation (fraction of the volume depth);
# kidney_1_dense is densely labelled, so it gives a clean validation signal
VAL_HOLDOUT = {"kidney_1_dense": (0.8, 1.0)}
TRAIN_Z_FRAC = {"kidney_1_dense": (0.0, 0.8)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s + worker_id)
    np.random.seed((s + worker_id) % 2 ** 32)


def build(args):
    if args.model == "plain":
        from models.plain_lora_model import PlainLoRAModel
        model = PlainLoRAModel(lora_r=args.lora_r, refine=args.refine)
    else:
        from models.spacing_lora_model import SpacingConditionedLoRAModel
        model = SpacingConditionedLoRAModel(lora_r=args.lora_r, refine=args.refine,
                                            spacing_encoding=args.spacing_encoding)
    set_trainable(model, args.unfreeze_blocks, args.decoder)
    return model


def set_trainable(model, unfreeze_blocks=0, decoder="full"):
    """unfreeze_blocks: also train the ordinary weights (attention, MLP, norms)
    of the last N encoder blocks; the LoRA/FiLM parts are trainable anyway.
    decoder: "full" trains the whole mask decoder, "output" only its
    up-sampling and hypernetwork output layers."""
    blocks = model.sam_model.image_encoder.blocks
    for blk in list(blocks)[len(blocks) - unfreeze_blocks:] if unfreeze_blocks else []:
        for p in blk.parameters():
            p.requires_grad = True
    if decoder == "output":
        for name, p in model.sam_model.mask_decoder.named_parameters():
            p.requires_grad = name.startswith(("output_upscaling", "output_hypernetworks_mlps"))


def validate(model, val_manifest, patch, device):
    """Mean Dice on the validation ROIs (sliding window, no TTA)."""
    from benchmark.infer import predict_roi
    dices = []
    model.eval()
    for sid, entries in val_manifest.items():
        for e in entries:
            d = np.load(e["path"])
            pred = predict_roi(model, d["image"], e["spacing_um"], patch, device.type)
            gt = d["label"] > 0
            inter = np.logical_and(pred > 0, gt).sum()
            dices.append(2 * inter / max(1, pred.sum() + gt.sum()))
    model.train()
    return float(np.mean(dices)) if dices else float("nan")


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    fov = tuple(args.fov_scale) if args.fov_scale else None
    if args.sampler == "slab":
        dataset = SlabPatchDataset(TRAIN_SAMPLES, raw_tiff_root=args.raw_tiff_root, patch_size=args.patch,
                                   fg_prob=args.fg_prob, fov_scale_range=fov,
                                   patches_per_slab=args.patches_per_slab, z_frac=TRAIN_Z_FRAC)
        loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers,
                            persistent_workers=args.workers > 0)
    else:
        # the random sampler cannot hold slices out, so it cannot be combined with validation
        if args.val_roi_dir:
            raise ValueError("--sampler random does not keep the validation slices out; use --sampler slab")
        dataset = RealGTKidneyDataset(
            sample_ids=TRAIN_SAMPLES, patch_size=(args.patch,) * 3,
            samples_per_epoch=args.iters * args.batch // len(TRAIN_SAMPLES) + 1,
            raw_tiff_root=args.raw_tiff_root, fg_prob=args.fg_prob, fov_scale_range=fov)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            worker_init_fn=_worker_init, drop_last=True, persistent_workers=args.workers > 0)

    model = build(args).to(device)
    criterion = HiPVesselLoss(alpha=args.loss_alpha, beta=args.loss_beta, gamma=args.loss_gamma,
                              use_cldice=not args.no_cldice)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.iters
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    val_manifest = None
    if args.val_roi_dir:
        from benchmark.rois import load_manifest
        val_manifest = load_manifest(args.val_roi_dir)
    best_val = -1.0

    state_path = os.path.join(args.out, "state.pth")
    start_epoch, log = 0, []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, log = st["epoch"], st["log"]
        best_val = st.get("best_val", -1.0)
        print(f"[*] resuming at epoch {start_epoch}")

    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[*] {args.model} | trainable {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%) | device {device}")

    model.train()
    it = iter(loader)
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            print(f"[!] time limit of {args.time_limit_min} min reached, stopping after {epoch} epochs", flush=True)
            break
        t0, losses, fg = time.time(), [], []
        for step in range(args.iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            optimizer.zero_grad(set_to_none=True)
            step_loss, step_fg = 0.0, 0.0
            for micro in range(args.accum):
                if micro:
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader)
                        batch = next(it)
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)
                spacings = batch["spacing"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    mask_pred, logits = model(images, spacings)
                # the loss has a clDice term and a spacing penalty; keep it in fp32
                loss = criterion(mask_pred.float(), targets, spacings.mean(0), logits=logits.float())
                scaler.scale(loss / args.accum).backward()
                step_loss += loss.item() / args.accum
                step_fg += targets.mean().item() / args.accum
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            losses.append(step_loss)
            fg.append(step_fg)
        entry = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "fg_fraction": float(np.mean(fg)),
                 "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        if val_manifest and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs):
            entry["val_dice"] = validate(model, val_manifest, args.patch, device)
            if entry["val_dice"] > best_val:
                best_val = entry["val_dice"]
                torch.save(model.state_dict(), os.path.join(args.out, "model_best.pth"))
        log.append(entry)
        val_txt = f" val_dice={entry['val_dice']:.4f}" if "val_dice" in entry else ""
        print(f"[✓] epoch {epoch + 1}/{args.epochs} loss={entry['loss']:.4f} "
              f"fg={100 * entry['fg_fraction']:.2f}%{val_txt} ({entry['seconds']}s)", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "log": log, "best_val": best_val}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        torch.save(model.state_dict(), os.path.join(args.out, "model.pth"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    best = os.path.join(args.out, "model_best.pth")
    return best if os.path.exists(best) else os.path.join(args.out, "model.pth")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["plain", "spacing"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-tiff-root", nargs="+", default=None)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = batch * accum)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--unfreeze-blocks", type=int, default=0, help="also train the last N encoder blocks")
    p.add_argument("--decoder", choices=["full", "output"], default="full")
    p.add_argument("--val-roi-dir", default=None, help="validation ROIs (benchmark.rois --validation)")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--fg-prob", type=float, default=0.5)
    p.add_argument("--fov-scale", type=float, nargs=2, default=None)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--no-cldice", action="store_true")
    p.add_argument("--loss-alpha", type=float, default=0.5)
    p.add_argument("--loss-beta", type=float, default=0.05)
    p.add_argument("--loss-gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sampler", choices=["slab", "random"], default="slab")
    p.add_argument("--patches-per-slab", type=int, default=32)
    p.add_argument("--spacing-encoding", choices=["raw", "log"], default="raw",
                   help="FiLM input: spacing in um (first experiments) or log2(spacing / 50um)")
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="stop after the epoch in which this many minutes have passed")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
