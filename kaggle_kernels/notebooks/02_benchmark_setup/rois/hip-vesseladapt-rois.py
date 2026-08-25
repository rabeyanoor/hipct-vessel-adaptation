"""
Kaggle CPU kernel: builds the benchmark ROIs from the Zenodo release
(benchmark/build_rois_zenodo.py) and leaves them in /kaggle/working/rois.
Save the output as a dataset (e.g. rabeya100x/hipct-benchmark-rois) and attach
it to the GPU kernels. No GPU needed; internet must be on.

Only the slices inside the ROIs are downloaded; temporary files go to /tmp so
they do not count toward the 20 GB output limit. Expected output ~7 GB.
"""

import os
import shutil
import subprocess
import sys
import zipfile

WORK_DIR = "/kaggle/working/hip-vesseladapt"


def find_by_marker(root, is_match, max_depth=5):
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        if not os.path.isdir(path):
            continue
        if is_match(path):
            return path
        if depth < max_depth:
            try:
                stack += [(os.path.join(path, n), depth + 1) for n in os.listdir(path)]
            except PermissionError:
                pass
    return None


CODE_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "train_lora_v2.py")))
os.makedirs(WORK_DIR, exist_ok=True)
shutil.copytree(CODE_DIR, WORK_DIR, dirs_exist_ok=True)
for z in ("models.zip", "utils.zip", "losses.zip", "data.zip", "benchmark.zip"):
    zp = os.path.join(WORK_DIR, z)
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(WORK_DIR)
        os.remove(zp)
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1", "tifffile", "imagecodecs"], check=True)

from benchmark import build_rois_zenodo, rois
import train_lora_v2

# kidney_1_voi and the validation ROIs (held-out last 20 % of kidney_1_dense)
# are cut from the competition data, which is mounted read-only; the other
# test sets only exist in the Zenodo release
COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
rois.make_rois(os.path.join(COMP_DIR, "train"), list(train_lora_v2.VAL_HOLDOUT), seed=0,
               out_dir="/kaggle/working/val_rois", n_rois={s: 6 for s in train_lora_v2.VAL_HOLDOUT},
               z_frac=train_lora_v2.VAL_HOLDOUT)
rois.make_rois(os.path.join(COMP_DIR, "train"), ["kidney_1_voi"], seed=0, out_dir="/kaggle/working/rois")
build_rois_zenodo.main(["kidney_6", "kidney_5", "kidney_additional_voi_13um"],
                       seed=0, out_dir="/kaggle/working/rois", workers=8, tmp="/tmp/rois_tmp")
print("[✓] ALL DONE")
