"""
Kaggle kernel: nnU-Net v2 baseline on the same training kidneys and the same
benchmark ROIs as our models.

The whole training kidneys do not fit into /kaggle/working once nnU-Net has
preprocessed them (float32 copies), so training uses fixed crops: CROPS_PER_VOLUME
cubes of CROP voxels per training volume, centred on labelled vessels with a
fixed seed. This is a reduced-data nnU-Net; the paper says so.

Prediction is done twice:
  - true spacing: nnU-Net resamples each ROI to its training spacing (~50um),
    i.e. its standard behaviour, comparable to our FOV-matched setting;
  - native: ROIs are written with a fake 50um spacing, so nnU-Net sees the
    native voxel grid, comparable to our native setting.

Inputs: code dataset, competition data, ROI dataset (see kaggle_v2_kernel.py).
The 12 h session limit allows about 100 nnU-Net epochs on a T4 (estimate);
TRAINER can be changed to nnUNetTrainer_250epochs with continue_training.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import zipfile

import numpy as np

TRAINER = "nnUNetTrainer_100epochs"
CROP = 256
CROPS_PER_VOLUME = 12
SEED = 0
TRAIN_SAMPLES = ["kidney_1_dense", "kidney_2", "kidney_3_dense", "kidney_3_sparse"]

WORK_DIR = "/kaggle/working/hip-vesseladapt"
OUT = "/kaggle/working/out"
NN = "/kaggle/working/nnunet"
os.environ["nnUNet_raw"] = f"{NN}/raw"
os.environ["nnUNet_preprocessed"] = f"{NN}/preprocessed"
os.environ["nnUNet_results"] = f"{NN}/results"


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
COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
ROI_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "manifest.json")))
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nnunetv2", "monai"], check=True)

import SimpleITK as sitk
from utils.real_gt_dataset import RealGTKidneyDataset

# 1. training crops
ds_name = "Dataset001_KidneyVessel"
img_dir = f"{NN}/raw/{ds_name}/imagesTr"
lbl_dir = f"{NN}/raw/{ds_name}/labelsTr"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)
rng = random.Random(SEED)
n_cases = 0
for sid in TRAIN_SAMPLES:
    ds = RealGTKidneyDataset(sample_ids=[sid], raw_tiff_root=os.path.join(COMP_DIR, "train"), samples_per_epoch=1)
    sample = ds.samples[0]
    coords = ds._fg_coords(sample)
    full = sample["shape"]
    side = [min(CROP, s) for s in full]
    sp_mm = float(sample["spacing"][0]) / 1000.0
    for k in range(CROPS_PER_VOLUME):
        c = coords[rng.randrange(len(coords))]
        s = [int(min(max(0, c[a] - side[a] // 2), full[a] - side[a])) for a in range(3)]
        img, lbl = ds._read_cube(sample, s[0], s[1], s[2], *side)
        case = f"{sid}_{k:02d}"
        im = sitk.GetImageFromArray(img.astype(np.float32))
        im.SetSpacing((sp_mm,) * 3)
        sitk.WriteImage(im, f"{img_dir}/{case}_0000.nii.gz")
        lb = sitk.GetImageFromArray((lbl > 0).astype(np.uint8))
        lb.SetSpacing((sp_mm,) * 3)
        sitk.WriteImage(lb, f"{lbl_dir}/{case}.nii.gz")
        n_cases += 1
    print(f"[*] {sid}: {CROPS_PER_VOLUME} crops")
with open(f"{NN}/raw/{ds_name}/dataset.json", "w") as f:
    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "vessel": 1},
               "numTraining": n_cases, "file_ending": ".nii.gz"}, f, indent=1)

# 2. plan, preprocess, train (fold "all": all crops, no cross-validation)
subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "1", "-c", "3d_fullres", "--verify_dataset_integrity"], check=True)
subprocess.run(["nnUNetv2_train", "1", "3d_fullres", "all", "-tr", TRAINER], check=True)

# 3. predict the benchmark ROIs in both settings and score
from benchmark import nifti_io, score
for tag, fake in (("resampled", None), ("native", 50.0)):
    src = f"{NN}/rois_{tag}"
    dst = f"{NN}/pred_{tag}"
    nifti_io.export(src, roi_dir=ROI_DIR, fake_spacing_um=fake)
    subprocess.run(["nnUNetv2_predict", "-i", src, "-o", dst, "-d", "1", "-c", "3d_fullres",
                    "-f", "all", "-tr", TRAINER], check=True)
    run_name = f"nnunet_{TRAINER.split('_')[-1]}_{tag}"
    nifti_io.import_preds(dst, run_name, roi_dir=ROI_DIR, pred_root=os.path.join(OUT, "preds"))
    res = score.score_run(os.path.join(OUT, "preds", run_name), roi_dir=ROI_DIR)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{run_name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    score.print_table([res])
print("[✓] ALL DONE")
