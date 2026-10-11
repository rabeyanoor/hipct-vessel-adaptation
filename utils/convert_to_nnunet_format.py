# utils/convert_to_nnunet_format.py
"""
Exports the SenNet+HOA kidney memmaps (from utils/convert_kidney_gt_to_memmap.py)
to the nnU-Net v2 raw dataset format, for the nnU-Net baseline (README
section 7).

Uses the same split as train_plain_lora.py/train_spacing_lora.py
(kidney_1_dense, kidney_2, kidney_3_dense, kidney_3_sparse for training,
kidney_1_voi held out) so the comparison is fair.

Needs `pip install nnunetv2`. Only exports the data; the nnU-Net commands for
preprocessing and training are printed at the end (they need a GPU).
"""

import gc
import json
import os

import numpy as np
import SimpleITK as sitk

from utils.kidney_metadata_verification import VERIFIED_SAMPLES

MEMMAP_DIR = "data/memmap_kidney_gt"
NNUNET_RAW_DIR = "data/nnUNet_raw"
DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_KidneyVessel"

# same split as the LoRA training scripts; kidney_1_voi (5.2um) is OOD only
TRAIN_SAMPLES = ["kidney_1_dense", "kidney_2", "kidney_3_dense", "kidney_3_sparse"]
OOD_SAMPLES = ["kidney_1_voi"]

_SPACING_UM = {s[0]: s[2] for s in VERIFIED_SAMPLES}


def _write_case(sample_id, images_dir, labels_dir):
    meta_path = os.path.join(MEMMAP_DIR, f"{sample_id}_meta.json")
    img_path = os.path.join(MEMMAP_DIR, f"{sample_id}_image.dat")
    lbl_path = os.path.join(MEMMAP_DIR, f"{sample_id}_label.dat")
    if not (os.path.exists(meta_path) and os.path.exists(img_path) and os.path.exists(lbl_path)):
        print(f"[WARNING] Memmap not found for '{sample_id}' -- run "
              f"utils/convert_kidney_gt_to_memmap.py first. Skipping.")
        return False

    with open(meta_path) as f:
        meta = json.load(f)
    shape = tuple(meta["shape"])
    spacing_mm = _SPACING_UM[sample_id] / 1000.0  # NIfTI/SimpleITK spacing convention is mm

    print(f"[*] Exporting {sample_id}: shape={shape}, spacing={spacing_mm:.5f}mm isotropic")

    img_mmap = np.memmap(img_path, dtype=meta["image_dtype"], mode="r", shape=shape)
    img_arr = np.array(img_mmap)  # SimpleITK needs a normal array
    img_sitk = sitk.GetImageFromArray(img_arr)
    img_sitk.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    sitk.WriteImage(img_sitk, os.path.join(images_dir, f"{sample_id}_0000.nii.gz"))
    del img_mmap, img_arr, img_sitk
    gc.collect()

    lbl_mmap = np.memmap(lbl_path, dtype=meta["label_dtype"], mode="r", shape=shape)
    lbl_arr = np.array(lbl_mmap)
    lbl_sitk = sitk.GetImageFromArray(lbl_arr)
    lbl_sitk.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    sitk.WriteImage(lbl_sitk, os.path.join(labels_dir, f"{sample_id}.nii.gz"))
    del lbl_mmap, lbl_arr, lbl_sitk
    gc.collect()

    print(f"[✓] {sample_id} written")
    return True


def convert_all():
    dataset_dir = os.path.join(NNUNET_RAW_DIR, DATASET_NAME)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    # keep kidney_1_voi out of imagesTr/labelsTr so nnU-Net never trains on it
    images_ood = os.path.join(dataset_dir, "imagesOOD_not_for_training")
    labels_ood = os.path.join(dataset_dir, "labelsOOD_not_for_training")
    for d in (images_tr, labels_tr, images_ood, labels_ood):
        os.makedirs(d, exist_ok=True)

    n_train = 0
    for sample_id in TRAIN_SAMPLES:
        if _write_case(sample_id, images_tr, labels_tr):
            n_train += 1
    for sample_id in OOD_SAMPLES:
        _write_case(sample_id, images_ood, labels_ood)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n[✓] nnU-Net raw dataset ready: {dataset_dir} ({n_train} training cases)")
    print("\nNext steps (need nnUNet_raw/nnUNet_preprocessed/nnUNet_results env vars set, and a GPU):")
    print(f"  export nnUNet_raw={os.path.abspath(NNUNET_RAW_DIR)}")
    print(f"  export nnUNet_preprocessed={os.path.abspath('data/nnUNet_preprocessed')}")
    print(f"  export nnUNet_results={os.path.abspath('data/nnUNet_results')}")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0")
    print(f"  # then evaluate the trained model against imagesOOD_not_for_training/"
          f"labelsOOD_not_for_training the same way evaluate_real_gt.py evaluates the other models")


if __name__ == "__main__":
    convert_all()
