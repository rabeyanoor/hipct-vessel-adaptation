"""
Kaggle kernel: trains the one ablation variant not already covered by the
main 100-epoch run (rabeya100x/hip-vesseladapt-100ep-gpu-train) -- Spacing-
Conditioned LoRA with the clDice loss term turned OFF (use_cldice=False on
HiPVesselLoss), 100 epochs, same protocol as the main comparison. Combined
with the already-trained zero-shot / Plain LoRA / Spacing-Conditioned LoRA
(with clDice) checkpoints, this completes the zero-shot -> +LoRA -> +spacing
-> +clDice ablation chain without re-training the first three stages.

Setup steps 1-4 are identical to kaggle_train_kernel.py -- see that file's
comments for why each one is there (path detection, disk quota, external-repo
patching).
"""

import os
import subprocess
import sys
import zipfile

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


def _list_tree(root, max_depth=3):
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        lines.append(f"{dirpath} -> dirs={dirnames[:10]} files={filenames[:5]}")
    return "\n".join(lines)


print("[*] /kaggle/input tree:\n" + _list_tree("/kaggle/input"))

CODE_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "train_plain_lora.py")))
if CODE_DIR is None:
    raise RuntimeError("Could not find code dataset (marker file train_plain_lora.py) under /kaggle/input")
print(f"[*] Using code dir: {CODE_DIR}")

COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train")) and
                           os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
if COMP_DIR is None:
    raise RuntimeError("Could not find competition data (train/kidney_1_dense) under /kaggle/input")
print(f"[*] Using competition dir: {COMP_DIR}")

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
print(f"[*] Reading TIFF slices directly from: {RAW_TIFF_ROOT} (no memmap conversion)")

from train_spacing_lora import train_spacing_lora
from evaluate_real_gt import evaluate_real_gt
from models.spacing_lora_model import SpacingConditionedLoRAModel

print("\n" + "#" * 80)
print("# ABLATION: SPACING-CONDITIONED LORA, NO CLDICE LOSS TERM -- 100 epochs")
print("#" * 80)
train_spacing_lora(resume=False, extra_epochs=100, samples_per_epoch=20,
                    raw_tiff_root=RAW_TIFF_ROOT, use_cldice=False,
                    checkpoint_dir="checkpoints_spacing_lora_no_cldice")

print("\n" + "#" * 80)
print("# ABLATION evaluation")
print("#" * 80)
evaluate_real_gt(checkpoint_path="checkpoints_spacing_lora_no_cldice/best_model_spacing_lora.pth",
                  label="Ablation: Spacing-Conditioned LoRA NO CLDICE, IN-DISTRIBUTION (100 epochs, GPU)",
                  samples_per_sample=20, model_class=SpacingConditionedLoRAModel, raw_tiff_root=RAW_TIFF_ROOT)
evaluate_real_gt(checkpoint_path="checkpoints_spacing_lora_no_cldice/best_model_spacing_lora.pth",
                  label="Ablation: Spacing-Conditioned LoRA NO CLDICE, OOD kidney_1_voi (100 epochs, GPU)",
                  sample_ids=["kidney_1_voi"], samples_per_sample=40,
                  model_class=SpacingConditionedLoRAModel, raw_tiff_root=RAW_TIFF_ROOT)

print("\n[✓] ALL DONE")
