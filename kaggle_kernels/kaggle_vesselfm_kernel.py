"""
Kaggle kernel: vesselFM (Wittmann et al., CVPR 2025) zero-shot on the
benchmark ROIs.

vesselFM's training data includes HiP-CT kidney (2.5um, Table 1 of their
paper) and the 08/2026 checkpoint was trained on an even broader set, so on
this data it is not a strict zero-shot model; the paper says so.

HiP-CT vessels are darker than the surrounding tissue. vesselFM has a
test-time inversion switch (tta.invert); we run both settings and report both
rather than picking one on the test ROIs.

Inputs: code dataset and ROI dataset (see kaggle_v2_kernel.py). GPU + internet.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

WORK_DIR = "/kaggle/working/hip-vesseladapt"
OUT = "/kaggle/working/out"
VFM = "/kaggle/working/vesselFM"


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
ROI_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "manifest.json")))
os.makedirs(WORK_DIR, exist_ok=True)
shutil.copytree(CODE_DIR, WORK_DIR, dirs_exist_ok=True)
for z in ("models.zip", "utils.zip", "losses.zip", "data.zip", "benchmark.zip"):
    zp = os.path.join(WORK_DIR, z)
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(WORK_DIR)
        os.remove(zp)
sys.path.insert(0, WORK_DIR)

subprocess.run(["git", "clone", "-q", "https://github.com/bwittmann/vesselFM.git", VFM], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", VFM], check=True)

os.chdir(WORK_DIR)
from benchmark import nifti_io, score

src = "/kaggle/working/vfm_in"
nifti_io.export(src, roi_dir=ROI_DIR)
for invert in (False, True):
    tag = "invert" if invert else "noinvert"
    dst = f"/kaggle/working/vfm_out_{tag}"
    # ckpt_path is left invalid on purpose: the script then downloads vesselFM_all.pt from Hugging Face
    subprocess.run([sys.executable, "vesselfm/seg/inference.py", f"image_path={src}", f"output_folder={dst}",
                    "ckpt_path=/nonexistent.pt", "device=cuda:0", f"tta.invert={str(invert).lower()}"],
                   cwd=VFM, check=True)
    run_name = f"vesselfm_zeroshot_{tag}"
    nifti_io.import_preds(dst, run_name, roi_dir=ROI_DIR, pred_root=os.path.join(OUT, "preds"),
                          suffix="_pred.nii.gz")
    res = score.score_run(os.path.join(OUT, "preds", run_name), roi_dir=ROI_DIR)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{run_name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    score.print_table([res])
print("[✓] ALL DONE")
