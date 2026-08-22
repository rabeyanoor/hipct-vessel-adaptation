# utils/convert_kidney_gt_to_memmap.py
"""
Converts the real expert-annotated SenNet+HOA kidney samples (image + label
TIFF slices, stored inside data/blood-vessel-segmentation.zip) into np.memmap
arrays for fast random-access patch extraction -- same rationale as
utils/convert_to_memmap.py for the Frangi-pseudo-labeled organs.

Reads directly from the zip via Python's zipfile (which supports efficient
random access to individual members, unlike gzip streams) rather than bulk-
extracting to a separate raw folder first, to avoid ~30-40GB of redundant
disk usage.

kidney_3_dense is a special case: it ships ONLY label files, because its dense
annotations are a more thoroughly-curated subset of the same raw slices as
kidney_3_sparse -- label and image slices share the same numeric index (e.g.
kidney_3_dense/labels/0496.tif corresponds to train/kidney_3_sparse/images/
0496.tif). Its images are read from kidney_3_sparse instead.
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
