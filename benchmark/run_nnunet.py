# benchmark/run_nnunet.py
"""
nnU-Net v2 baseline on the same training kidneys and benchmark ROIs as our models.

The preprocessed copies of the whole training kidneys do not fit on a Kaggle
disk, so nnU-Net is trained on fixed crops: CROPS_PER_VOLUME cubes of CROP
voxels per training volume, centred on labelled voxels with a fixed seed.
For kidney_1_dense the crops come from the same first 80 % of the slices that
our models train on (the last 20 % are their validation slices). This is a
reduced-data nnU-Net, and the paper says so.

Steps (run with the nnUNet_raw / nnUNet_preprocessed / nnUNet_results
environment variables set):
  prepare  write the training crops as Dataset001_KidneyVessel
  train    plan + preprocess + train 3d_fullres on fold "all"
  predict  predict the ROIs and write benchmark predictions
           --tag resampled: true spacing, so nnU-Net resamples every ROI to its
                            ~50um training spacing (its standard behaviour,
                            comparable to our fov50 setting)
           --tag native:    every ROI is labelled 50um, so nnU-Net works on the
                            native voxel grid (comparable to our native setting)
           Test-time mirroring is off, as for our models.

Usage:
    python -m benchmark.run_nnunet prepare --raw-tiff-root <train>
    python -m benchmark.run_nnunet train
    python -m benchmark.run_nnunet predict --tag resampled --roi-dir <rois> --pred-root <out>/preds
"""

import argparse
import json
import os
import random
import subprocess
import tempfile

import numpy as np

CROP = 256
CROPS_PER_VOLUME = 12
SEED = 0
DATASET = "Dataset001_KidneyVessel"
TRAINER = "nnUNetTrainer_100epochs"
TRAIN_SAMPLES = ["kidney_1_dense", "kidney_2", "kidney_3_dense", "kidney_3_sparse"]
Z_FRAC = {"kidney_1_dense": (0.0, 0.8)}  # same split as train_lora_v2.py


def prepare(raw_tiff_root):
    import SimpleITK as sitk
    from utils.real_gt_dataset import RealGTKidneyDataset

    base = os.path.join(os.environ["nnUNet_raw"], DATASET)
    os.makedirs(f"{base}/imagesTr", exist_ok=True)
    os.makedirs(f"{base}/labelsTr", exist_ok=True)
    rng = random.Random(SEED)
    n = 0
    for sid in TRAIN_SAMPLES:
        ds = RealGTKidneyDataset(sample_ids=[sid], raw_tiff_root=raw_tiff_root, samples_per_epoch=1)
        sample = ds.samples[0]
        full = sample["shape"]
        lo, hi = Z_FRAC.get(sid, (0.0, 1.0))
        z_lo, z_hi = int(lo * full[0]), int(hi * full[0])
        side = [min(CROP, z_hi - z_lo), min(CROP, full[1]), min(CROP, full[2])]
        coords = ds._fg_coords(sample)
        coords = coords[(coords[:, 0] >= z_lo) & (coords[:, 0] < z_hi)]
        sp_mm = float(sample["spacing"][0]) / 1000.0
        for k in range(CROPS_PER_VOLUME):
            c = coords[rng.randrange(len(coords))]
            s = [int(min(max(z_lo, c[0] - side[0] // 2), z_hi - side[0]))]
            s += [int(min(max(0, c[a] - side[a] // 2), full[a] - side[a])) for a in (1, 2)]
            img, lbl = ds._read_cube(sample, s[0], s[1], s[2], *side)
            case = f"{sid}_{k:02d}"
            im = sitk.GetImageFromArray(img.astype(np.float32))
            im.SetSpacing((sp_mm,) * 3)
            sitk.WriteImage(im, f"{base}/imagesTr/{case}_0000.nii.gz")
            lb = sitk.GetImageFromArray((lbl > 0).astype(np.uint8))
            lb.SetSpacing((sp_mm,) * 3)
            sitk.WriteImage(lb, f"{base}/labelsTr/{case}.nii.gz")
            n += 1
        print(f"[*] {sid}: {CROPS_PER_VOLUME} crops of {side}", flush=True)
    with open(f"{base}/dataset.json", "w") as f:
        json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "vessel": 1},
                   "numTraining": n, "file_ending": ".nii.gz"}, f, indent=1)


def train():
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "1", "-c", "3d_fullres",
                    "--verify_dataset_integrity"], check=True)
    subprocess.run(["nnUNetv2_train", "1", "3d_fullres", "all", "-tr", TRAINER], check=True)


def predict(tag, roi_dir, pred_root):
    from benchmark import nifti_io
    fake = {"resampled": None, "native": 50.0}[tag]
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        src, dst = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        nifti_io.export(src, roi_dir=roi_dir, fake_spacing_um=fake)
        subprocess.run(["nnUNetv2_predict", "-i", src, "-o", dst, "-d", "1", "-c", "3d_fullres", "-f", "all",
                        "-tr", TRAINER, "--disable_tta", "-npp", "2", "-nps", "2"], check=True)
        nifti_io.import_preds(dst, f"nnunet_{tag}", roi_dir=roi_dir, pred_root=pred_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["prepare", "train", "predict"])
    p.add_argument("--raw-tiff-root", nargs="+")
    p.add_argument("--tag", choices=["resampled", "native"])
    p.add_argument("--roi-dir")
    p.add_argument("--pred-root")
    a = p.parse_args()
    if a.step == "prepare":
        prepare(a.raw_tiff_root)
    elif a.step == "train":
        train()
    else:
        predict(a.tag, a.roi_dir, a.pred_root)
