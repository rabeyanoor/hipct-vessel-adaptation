"""
Kaggle kernel: runs the 50-epoch Plain LoRA and Spacing-Conditioned LoRA
training + in-distribution/OOD evaluation on Kaggle's free GPU, using the
SenNet+HOA competition data attached natively (no re-upload) and this
project's source code attached as a private dataset.
"""

import os
import subprocess
import sys
import zipfile

print("[*] /kaggle/input contents:", os.listdir("/kaggle/input"))

WORK_DIR = "/kaggle/working/hip-vesseladapt"

# Find the code dataset dir regardless of exact mount name (Kaggle sometimes
# suffixes/alters slugs), by looking for our known marker file.
CODE_DIR = None
for name in os.listdir("/kaggle/input"):
    candidate = os.path.join("/kaggle/input", name)
    if os.path.exists(os.path.join(candidate, "train_plain_lora.py")):
        CODE_DIR = candidate
        break
if CODE_DIR is None:
    raise RuntimeError(f"Could not find code dataset under /kaggle/input. Contents: {os.listdir('/kaggle/input')}")
print(f"[*] Using code dir: {CODE_DIR}")

COMP_DIR = None
for name in os.listdir("/kaggle/input"):
    candidate = os.path.join("/kaggle/input", name)
    if os.path.isdir(os.path.join(candidate, "train")):
        COMP_DIR = candidate
        break
if COMP_DIR is None:
    raise RuntimeError(f"Could not find competition data under /kaggle/input. Contents: {os.listdir('/kaggle/input')}")
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

# 4. Convert real-GT kidney samples directly from the attached competition folder
#    (plain files, no zip -- Kaggle mounts competition data unzipped)
import json
import tifffile
import numpy as np

from utils.kidney_metadata_verification import VERIFIED_SAMPLES

MEMMAP_DIR = "data/memmap_kidney_gt"
os.makedirs(MEMMAP_DIR, exist_ok=True)
IMAGE_SOURCE_OVERRIDE = {"kidney_3_dense": "kidney_3_sparse"}

def convert_sample(sample_id, split, n_label_slices):
    if split != "train" or n_label_slices == 0:
        return
    img_path = os.path.join(MEMMAP_DIR, f"{sample_id}_image.dat")
    lbl_path = os.path.join(MEMMAP_DIR, f"{sample_id}_label.dat")
    meta_path = os.path.join(MEMMAP_DIR, f"{sample_id}_meta.json")
    if os.path.exists(meta_path):
        print(f"[SKIP] {sample_id} already converted")
        return

    image_source = IMAGE_SOURCE_OVERRIDE.get(sample_id, sample_id)
    label_dir = os.path.join(COMP_DIR, "train", sample_id, "labels")
    image_dir = os.path.join(COMP_DIR, "train", image_source, "images")
    label_names = sorted(os.listdir(label_dir))

    first_label = tifffile.imread(os.path.join(label_dir, label_names[0]))
    h, w = first_label.shape
    d = len(label_names)

    print(f"[*] Converting {sample_id}: {d} slices, ({h},{w}) each")
    img_mmap = np.memmap(img_path, dtype=np.uint16, mode="w+", shape=(d, h, w))
    lbl_mmap = np.memmap(lbl_path, dtype=np.uint8, mode="w+", shape=(d, h, w))
    for z, name in enumerate(label_names):
        lbl_mmap[z] = (tifffile.imread(os.path.join(label_dir, name)) > 0).astype(np.uint8)
        img_mmap[z] = tifffile.imread(os.path.join(image_dir, name))
        if (z + 1) % 500 == 0:
            print(f"    {sample_id}: {z+1}/{d}")
    img_mmap.flush(); lbl_mmap.flush()
    with open(meta_path, "w") as f:
        json.dump({"shape": [d, h, w], "image_dtype": "uint16", "label_dtype": "uint8"}, f)
    print(f"[✓] {sample_id} done")

for (sid, _spec, _sp, _nimg, n_lbl, split, _dens, _src) in VERIFIED_SAMPLES:
    convert_sample(sid, split, n_lbl)

# 5. Train + evaluate both models, 50 epochs each
from train_plain_lora import train_plain_lora
from train_spacing_lora import train_spacing_lora
from evaluate_real_gt import evaluate_real_gt
from models.plain_lora_model import PlainLoRAModel
from models.spacing_lora_model import SpacingConditionedLoRAModel

print("\n" + "#" * 80)
print("# PLAIN LORA: 50 epochs")
print("#" * 80)
train_plain_lora(resume=False, extra_epochs=50, samples_per_epoch=20)

print("\n" + "#" * 80)
print("# PLAIN LORA: evaluation")
print("#" * 80)
evaluate_real_gt(checkpoint_path="checkpoints_plain_lora/best_model_plain_lora.pth",
                  label="Plain LoRA IN-DISTRIBUTION (50 epochs, GPU)",
                  samples_per_sample=20, model_class=PlainLoRAModel)
evaluate_real_gt(checkpoint_path="checkpoints_plain_lora/best_model_plain_lora.pth",
                  label="Plain LoRA OOD kidney_1_voi (50 epochs, GPU)",
                  sample_ids=["kidney_1_voi"], samples_per_sample=40, model_class=PlainLoRAModel)

print("\n" + "#" * 80)
print("# SPACING-CONDITIONED LORA: 50 epochs")
print("#" * 80)
train_spacing_lora(resume=False, extra_epochs=50, samples_per_epoch=20)

print("\n" + "#" * 80)
print("# SPACING-CONDITIONED LORA: evaluation")
print("#" * 80)
evaluate_real_gt(checkpoint_path="checkpoints_spacing_lora/best_model_spacing_lora.pth",
                  label="Spacing-Conditioned LoRA IN-DISTRIBUTION (50 epochs, GPU)",
                  samples_per_sample=20, model_class=SpacingConditionedLoRAModel)
evaluate_real_gt(checkpoint_path="checkpoints_spacing_lora/best_model_spacing_lora.pth",
                  label="Spacing-Conditioned LoRA OOD kidney_1_voi (50 epochs, GPU)",
                  sample_ids=["kidney_1_voi"], samples_per_sample=40, model_class=SpacingConditionedLoRAModel)

print("\n[✓] ALL DONE")
