# utils/convert_kidney_gt_to_memmap.py
"""
Converts the SenNet+HOA kidney samples (image + label TIFFs inside
data/blood-vessel-segmentation.zip) to np.memmap for fast patch reading, like
utils/convert_to_memmap.py does for the other organs.

Reads straight from the zip (zipfile allows random access) so we don't need
another ~30-40GB for an extracted copy.

kidney_3_dense only has label files; its images are the same slices as
kidney_3_sparse with the same numbers (kidney_3_dense/labels/0496.tif goes with
train/kidney_3_sparse/images/0496.tif), so images are read from there.
"""

import io
import json
import os
import zipfile

import numpy as np
import tifffile

from utils.kidney_metadata_verification import VERIFIED_SAMPLES

ZIP_PATH = "data/blood-vessel-segmentation.zip"
MEMMAP_DIR = "data/memmap_kidney_gt"

# sample_id -> folder to read images from (differs from sample_id only for kidney_3_dense)
IMAGE_SOURCE_OVERRIDE = {
    "kidney_3_dense": "kidney_3_sparse",
}


def convert_sample(zf, sample_id, split, n_label_slices):
    if split != "train" or n_label_slices == 0:
        print(f"[SKIP] {sample_id}: split={split}, n_label_slices={n_label_slices} (no public GT)")
        return

    image_source = IMAGE_SOURCE_OVERRIDE.get(sample_id, sample_id)
    label_prefix = f"train/{sample_id}/labels/"
    image_prefix = f"train/{image_source}/images/"

    label_names = sorted(
        n for n in zf.namelist() if n.startswith(label_prefix) and n.endswith(".tif")
    )
    if not label_names:
        print(f"[WARNING] No label files found for {sample_id} under {label_prefix}")
        return

    first_label = tifffile.imread(io.BytesIO(zf.read(label_names[0])))
    h, w = first_label.shape
    d = len(label_names)

    os.makedirs(MEMMAP_DIR, exist_ok=True)
    img_path = os.path.join(MEMMAP_DIR, f"{sample_id}_image.dat")
    lbl_path = os.path.join(MEMMAP_DIR, f"{sample_id}_label.dat")
    meta_path = os.path.join(MEMMAP_DIR, f"{sample_id}_meta.json")

    print(f"[*] Converting {sample_id}: {d} slices, ({h}, {w}) each -> {img_path} / {lbl_path}")
    img_mmap = np.memmap(img_path, dtype=np.uint16, mode="w+", shape=(d, h, w))
    lbl_mmap = np.memmap(lbl_path, dtype=np.uint8, mode="w+", shape=(d, h, w))

    for z, label_name in enumerate(label_names):
        basename = os.path.basename(label_name)
        image_name = f"{image_prefix}{basename}"

        label_arr = tifffile.imread(io.BytesIO(zf.read(label_name)))
        image_arr = tifffile.imread(io.BytesIO(zf.read(image_name)))

        lbl_mmap[z] = (label_arr > 0).astype(np.uint8)  # binarize: 0/255 -> 0/1
        img_mmap[z] = image_arr

        if (z + 1) % 200 == 0 or z == d - 1:
            print(f"    {sample_id}: {z + 1}/{d} slices written")

    img_mmap.flush()
    lbl_mmap.flush()
    del img_mmap, lbl_mmap

    with open(meta_path, "w") as f:
        json.dump({"shape": [d, h, w], "image_dtype": "uint16", "label_dtype": "uint8"}, f)

    print(f"[✓] {sample_id} ready: {os.path.getsize(img_path) / 1e9:.2f} GB image + "
          f"{os.path.getsize(lbl_path) / 1e9:.2f} GB label")


def convert_all():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for (sample_id, _specimen, _spacing, _n_img, n_lbl, split, _density, _source) in VERIFIED_SAMPLES:
            convert_sample(zf, sample_id, split, n_lbl)


if __name__ == "__main__":
    convert_all()
