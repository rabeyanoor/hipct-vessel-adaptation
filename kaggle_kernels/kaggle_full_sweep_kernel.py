"""
Kaggle kernel: FULL hyperparameter sweep (learning rate x Dice/clDice loss
balance), for Plain LoRA and Spacing-Conditioned LoRA -- the grid design from
colab_run.py's run_hyperparameter_sweep() (3 LRs x 3 loss_alphas x 2
architectures, 15 epochs each), run here on Kaggle GPU instead of Colab since
this environment already has working Kaggle push/fetch set up.

Follow-up to the reduced sweep (kaggle_sweep_kernel.py, kernel
rabeya100x/hip-vesseladapt-hp-sweep, see sweep_results.json): that 3x2 grid
at 5 epochs found loss_alpha=0.25 beating loss_alpha=0.6 at every learning
rate tested for spacing_lora, which is DIFFERENT from the loss_alpha=0.5
already used in README.md sections 5.2/5.4 -- this full grid adds 0.5 as the
middle point for a direct comparison, plus more epochs (15 vs 5) for a less
noisy signal. Setup steps identical to kaggle_kernels/kaggle_train_kernel.py
-- see that file's comments for why each one is there.
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

# Full grid: colab_run.py's run_hyperparameter_sweep() design.
SWEEP_EPOCHS = 15
LR_GRID = [5e-5, 1e-4, 2e-4]
ALPHA_GRID = [0.25, 0.5, 0.75]

RESULTS_PATH = "full_sweep_results.json"
results = {}
# A single Kaggle session's GPU time limit (or the weekly quota) can run out
# mid-sweep -- this grid is 18 runs at 15 epochs each, several hours, vs. the
# 12-run/5-epoch reduced sweep that already took ~115 minutes. Writing the
# results file after EVERY run (not just once at the end) means a session
# that gets cut off partway still leaves every run completed so far on disk,
# instead of losing the whole sweep's output to one timeout.
for model_name, train_fn, model_cls, ckpt_name in [
    ("plain_lora", train_plain_lora, PlainLoRAModel, "best_model_plain_lora.pth"),
    ("spacing_lora", train_spacing_lora, SpacingConditionedLoRAModel, "best_model_spacing_lora.pth"),
]:
    for lr in LR_GRID:
        for alpha in ALPHA_GRID:
            run_id = f"{model_name}_lr{lr}_alpha{alpha}"
            ckpt_dir = f"checkpoints_full_sweep/{run_id}"
            print("\n" + "#" * 80)
            print(f"# FULL SWEEP: {run_id} ({SWEEP_EPOCHS} epochs)")
            print("#" * 80)
            train_fn(resume=False, extra_epochs=SWEEP_EPOCHS, samples_per_epoch=20,
                      raw_tiff_root=RAW_TIFF_ROOT, lr=lr, loss_alpha=alpha,
                      checkpoint_dir=ckpt_dir, seed=0)
            ckpt_file = f"{ckpt_dir}/{ckpt_name}"
            metrics = evaluate_real_gt(checkpoint_path=ckpt_file, label=f"FULL SWEEP EVAL: {run_id}",
                                        model_class=model_cls, samples_per_sample=20,
                                        raw_tiff_root=RAW_TIFF_ROOT)
            results[run_id] = {"lr": lr, "alpha": alpha, "dice": metrics["dice"] if metrics else None}
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[*] {len(results)}/18 runs done, {RESULTS_PATH} updated")

print("\n" + "=" * 80)
print("FULL SWEEP SUMMARY (best Dice per architecture)")
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

print("\n[✓] FULL SWEEP DONE")
