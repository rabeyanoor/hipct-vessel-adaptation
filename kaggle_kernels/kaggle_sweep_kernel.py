"""
Kaggle kernel: reduced hyperparameter sweep (learning rate x loss alpha) for
Plain LoRA and Spacing-Conditioned LoRA, with short runs so it fits in one
Kaggle session. Setup is the same as kaggle_train_kernel.py.
"""

import os
import subprocess
import sys
import zipfile
import json

WORK_DIR = "/kaggle/working/hip-vesseladapt"


def find_by_marker(root, is_match, max_depth=4):
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        if not os.path.isdir(path):
            continue
        if is_match(path):
            return path
        if depth >= max_depth:
            continue
        try:
            for name in os.listdir(path):
                stack.append((os.path.join(path, name), depth + 1))
        except PermissionError:
            continue
    return None


CODE_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "train_plain_lora.py")))
if CODE_DIR is None:
    raise RuntimeError("Could not find code dataset under /kaggle/input")
COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train")) and
                           os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
if COMP_DIR is None:
    raise RuntimeError("Could not find competition data under /kaggle/input")

import shutil
os.makedirs(WORK_DIR, exist_ok=True)
for item in os.listdir(CODE_DIR):
    src = os.path.join(CODE_DIR, item)
    dst = os.path.join(WORK_DIR, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
for zip_name in ["models.zip", "utils.zip", "losses.zip", "data.zip"]:
    zip_path = os.path.join(WORK_DIR, zip_name)
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(WORK_DIR)
        os.remove(zip_path)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai", "gdown"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/openmedlab/SAM-Med3D.git", "external/SAM-Med3D"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/jocpae/clDice.git", "external/clDice"], check=True)

from utils.patch_external_repos import patch_sam_med3d
patch_sam_med3d()

os.makedirs("external/SAM-Med3D/ckpt", exist_ok=True)
ckpt_path = "external/SAM-Med3D/ckpt/sam_med3d.pth"
if not os.path.exists(ckpt_path):
    subprocess.run([
        sys.executable, "-m", "gdown",
        "https://drive.google.com/uc?id=1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub",
        "-O", ckpt_path,
    ], check=True)
print(f"[✓] Checkpoint ready: {os.path.getsize(ckpt_path)/1e9:.2f} GB")

RAW_TIFF_ROOT = os.path.join(COMP_DIR, "train")

from train_plain_lora import train_plain_lora
from train_spacing_lora import train_spacing_lora
from evaluate_real_gt import evaluate_real_gt
from models.plain_lora_model import PlainLoRAModel
from models.spacing_lora_model import SpacingConditionedLoRAModel

# smaller grid than colab_run.py's sweep, should finish in under an hour
SWEEP_EPOCHS = 5
LR_GRID = [3e-5, 1e-4, 2e-4]
ALPHA_GRID = [0.25, 0.6]

results = {}
for model_name, train_fn, model_cls, ckpt_name in [
    ("plain_lora", train_plain_lora, PlainLoRAModel, "best_model_plain_lora.pth"),
    ("spacing_lora", train_spacing_lora, SpacingConditionedLoRAModel, "best_model_spacing_lora.pth"),
]:
    for lr in LR_GRID:
        for alpha in ALPHA_GRID:
            run_id = f"{model_name}_lr{lr}_alpha{alpha}"
            ckpt_dir = f"checkpoints_sweep/{run_id}"
            print("\n" + "#" * 80)
            print(f"# SWEEP: {run_id} ({SWEEP_EPOCHS} epochs)")
            print("#" * 80)
            train_fn(resume=False, extra_epochs=SWEEP_EPOCHS, samples_per_epoch=20,
                      raw_tiff_root=RAW_TIFF_ROOT, lr=lr, loss_alpha=alpha,
                      checkpoint_dir=ckpt_dir, seed=0)
            ckpt_file = f"{ckpt_dir}/{ckpt_name}"
            metrics = evaluate_real_gt(checkpoint_path=ckpt_file, label=f"SWEEP EVAL: {run_id}",
                                        model_class=model_cls, samples_per_sample=20,
                                        raw_tiff_root=RAW_TIFF_ROOT)
            results[run_id] = {"lr": lr, "alpha": alpha, "dice": metrics["dice"] if metrics else None}

with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("SWEEP SUMMARY (best Dice per architecture)")
print("=" * 80)
for model_name in ("plain_lora", "spacing_lora"):
    model_results = {k: v for k, v in results.items() if k.startswith(model_name)}
    valid = {k: v for k, v in model_results.items() if v["dice"] is not None and v["dice"] == v["dice"]}
    if valid:
        best_run = max(valid, key=lambda k: valid[k]["dice"])
        print(f"{model_name}: best = {best_run} (lr={valid[best_run]['lr']}, "
              f"alpha={valid[best_run]['alpha']}, Dice={valid[best_run]['dice']:.4f})")
    else:
        print(f"{model_name}: no valid (non-nan) Dice among sweep runs")

print("\n[✓] SWEEP DONE")
