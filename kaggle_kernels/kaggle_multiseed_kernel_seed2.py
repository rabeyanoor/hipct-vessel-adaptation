"""
Kaggle kernel: third 100-epoch run of Plain LoRA and Spacing-Conditioned LoRA,
seed=2, for the multi-seed results (baki.sh section 2 / colab_run.py's
run_multiseed()). Same as kaggle_multiseed_kernel.py (seed=1, finished
2026-09-16, test.sh 3.11); only SEED and the output file name differ.

The seed=1 run used only ~9788s of nargisakterr's 21600s weekly quota
(checked with api.quota_view()), so there is enough left (~11800s) to run
seed=2 on the same account without waiting for a reset. With the existing
100-epoch checkpoint (default seed) and seed=1 this gives n=3 per
architecture, as planned.

Setup steps 1-4 are copied from kaggle_train_kernel.py (same six fixes as in
test.sh 3.9: input paths, raw_tiff_root mode for the disk quota, clDice
import, SAM-Med3D import, T4 machine_shape, pos_embed patch).

Before pushing from the other account (see "id" in
kernel-metadata-multiseed-seed2.json):
  1. The private code dataset rabeya100x/hip-vesseladapt-code has to be
     shared with that account (or made public).
  2. That account has to accept the blood-vessel-segmentation competition
     rules on the Kaggle website, otherwise the data won't attach.
Both have to be done on the website, not through the API.
"""

import json
import os
import subprocess
import sys
import zipfile

WORK_DIR = "/kaggle/working/hip-vesseladapt"
SEED = 2


def find_by_marker(root, is_match, max_depth=4):
    """Search under root (limited depth) for a directory where is_match(path)
    is true; the /kaggle/input layout is not always the same."""
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
    raise RuntimeError(
        "Could not find code dataset (marker file train_plain_lora.py) under /kaggle/input -- "
        "check that rabeya100x/hip-vesseladapt-code has been shared with this account."
    )
print(f"[*] Using code dir: {CODE_DIR}")

COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train")) and
                           os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
if COMP_DIR is None:
    raise RuntimeError(
        "Could not find competition data (train/kidney_1_dense) under /kaggle/input -- "
        "check that this account has accepted the blood-vessel-segmentation competition rules."
    )
print(f"[*] Using competition dir: {COMP_DIR}")

# 1. Set up working code directory (copy from read-only input, extract per-dir zips)
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

# 2. Install missing deps + clone external repos
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai", "gdown"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/openmedlab/SAM-Med3D.git", "external/SAM-Med3D"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/jocpae/clDice.git", "external/clDice"], check=True)

from utils.patch_external_repos import patch_sam_med3d
patch_sam_med3d()

# 3. Download SAM-Med3D pretrained checkpoint
os.makedirs("external/SAM-Med3D/ckpt", exist_ok=True)
ckpt_path = "external/SAM-Med3D/ckpt/sam_med3d.pth"
if not os.path.exists(ckpt_path):
    subprocess.run([
        sys.executable, "-m", "gdown",
        "https://drive.google.com/uc?id=1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub",
        "-O", ckpt_path,
    ], check=True)
print(f"[✓] Checkpoint ready: {os.path.getsize(ckpt_path)/1e9:.2f} GB")

# 4. Data source: read TIFF slices directly from the attached competition
#    folder (raw_tiff_root mode), see test.sh section 3.9 bug 2.
RAW_TIFF_ROOT = os.path.join(COMP_DIR, "train")
print(f"[*] Reading TIFF slices directly from: {RAW_TIFF_ROOT} (no memmap conversion)")

# 5. Train + evaluate both models, 100 epochs each, seed=1
from train_plain_lora import train_plain_lora
from train_spacing_lora import train_spacing_lora
from evaluate_real_gt import evaluate_real_gt
from models.plain_lora_model import PlainLoRAModel
from models.spacing_lora_model import SpacingConditionedLoRAModel

results = {}

for model_name, train_fn, model_cls, ckpt_name in [
    ("plain_lora", train_plain_lora, PlainLoRAModel, "best_model_plain_lora.pth"),
    ("spacing_lora", train_spacing_lora, SpacingConditionedLoRAModel, "best_model_spacing_lora.pth"),
]:
    ckpt_dir = f"checkpoints_multiseed/{model_name}_seed{SEED}"
    print("\n" + "#" * 80)
    print(f"# {model_name.upper()}: 100 epochs, seed={SEED}")
    print("#" * 80)
    train_fn(resume=False, extra_epochs=100, samples_per_epoch=20,
              raw_tiff_root=RAW_TIFF_ROOT, checkpoint_dir=ckpt_dir, seed=SEED)

    ckpt_file = f"{ckpt_dir}/{ckpt_name}"
    print("\n" + "#" * 80)
    print(f"# {model_name.upper()}: evaluation, seed={SEED}")
    print("#" * 80)
    in_dist = evaluate_real_gt(checkpoint_path=ckpt_file,
                                label=f"{model_name} seed={SEED} IN-DISTRIBUTION (100 epochs, GPU)",
                                samples_per_sample=20, model_class=model_cls, raw_tiff_root=RAW_TIFF_ROOT)
    ood = evaluate_real_gt(checkpoint_path=ckpt_file,
                            label=f"{model_name} seed={SEED} OOD kidney_1_voi (100 epochs, GPU)",
                            sample_ids=["kidney_1_voi"], samples_per_sample=40,
                            model_class=model_cls, raw_tiff_root=RAW_TIFF_ROOT)
    results[model_name] = {
        "seed": SEED,
        "in_dist_dice": in_dist["dice"] if in_dist else None,
        "ood_dice": ood["dice"] if ood else None,
    }

with open("multiseed_seed2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"MULTISEED (seed={SEED}) SUMMARY")
print("=" * 80)
for model_name, r in results.items():
    print(f"{model_name}: in-dist Dice = {r['in_dist_dice']}, OOD Dice = {r['ood_dice']}")
print("\n[✓] ALL DONE -- results in multiseed_seed2_results.json")
