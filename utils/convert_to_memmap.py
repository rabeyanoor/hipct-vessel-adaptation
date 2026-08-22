# utils/convert_to_memmap.py
"""
Converts each organ's .jp2 slice stack into an uncompressed np.memmap file
(+ a small JSON file with shape/dtype).

Reason: .nii.gz is gzip-compressed and SimpleITK can't seek in it, so every
patch read decompresses from the start of the file. That took 15-56 seconds
per 64^3 patch, way too slow for training.

Slices are read one by one so RAM use stays at one 2D slice (this machine has
15GB RAM, ~8.6GB free).
"""

import glob
import json
import os

import numpy as np
import SimpleITK as sitk

from config import ORGAN_CONFIG

MEMMAP_DIR = "data/memmap"


def convert_organ_to_memmap(organ, meta):
    jp2_files = sorted(glob.glob(os.path.join(meta["raw_dir"], "**", "*.jp2"), recursive=True))
    if not jp2_files:
        print(f"[WARNING] No .jp2 files found for organ '{organ}' in {meta['raw_dir']}")
        return

    first_slice = sitk.GetArrayFromImage(sitk.ReadImage(jp2_files[0]))
    h, w = first_slice.shape
    d = len(jp2_files)
    dtype = first_slice.dtype

    os.makedirs(MEMMAP_DIR, exist_ok=True)
    data_path = os.path.join(MEMMAP_DIR, f"{organ}.dat")
    meta_path = os.path.join(MEMMAP_DIR, f"{organ}_meta.json")

    print(f"[*] Converting {organ}: {d} slices, ({h}, {w}) each, dtype={dtype} -> {data_path}")
    mmap = np.memmap(data_path, dtype=dtype, mode="w+", shape=(d, h, w))

    for z, slice_path in enumerate(jp2_files):
        slice_arr = sitk.GetArrayFromImage(sitk.ReadImage(slice_path))
        mmap[z] = slice_arr
        if (z + 1) % 200 == 0 or z == d - 1:
            print(f"    {organ}: {z + 1}/{d} slices written")

    mmap.flush()
    del mmap

    with open(meta_path, "w") as f:
        json.dump({"shape": [d, h, w], "dtype": str(dtype)}, f)

    print(f"[✓] {organ} memmap ready: {data_path} ({os.path.getsize(data_path) / 1e9:.2f} GB)")


def convert_all():
    for organ, meta in ORGAN_CONFIG.items():
        convert_organ_to_memmap(organ, meta)


if __name__ == "__main__":
    convert_all()
