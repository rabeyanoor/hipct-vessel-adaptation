# utils/convert_hoa_parquet_to_memmap.py
"""
Converts the Human Organ Atlas raw kidney_right (donor LADAF-2021-17, same
specimen as kidney_1_dense/kidney_1_voi) volumes from the eternalaudrey/
hipct_3d HF mirror into memmap arrays MultiOrganHiPCTDataset can read --
addresses baki.sh item 3 / README.md section 7 ("train on a wider range of
real spacings"). See config.py's HOA_KIDNEY_MULTISPACING_CONFIG for why these
three specific volumes were chosen.

Data layout discovered by direct inspection (not documented in the dataset
card): each row is one (32, 224, 224) uint16 TIFF-encoded 3D patch ("z-slab"),
NOT a single 2D slice -- and patches for a given volume are scattered across
shards, not grouped by shard. `image.path` encodes the source volume name and
a 0-indexed, contiguous patch number (e.g. "..._patch37.tiff"); reconstructing
a volume means collecting every row across every shard where `name` matches,
sorting by that patch index, and stacking along z. All three volumes here
have exactly 68 contiguous patches (0-67) once collected from the right
shards (see SHARD_MAP below -- found by scanning all 66 shards' `name`
column, expensive enough over a slow connection that it's hardcoded here
rather than re-scanned every run).

Shard files must already be downloaded to data/raw/hoa_kidney_right_shards/
(gitignored -- see README setup instructions). Download with e.g.:
    from huggingface_hub import hf_hub_download
    for i in (42, 43, 44):
        hf_hub_download(repo_id="eternalaudrey/hipct_3d", repo_type="dataset",
                         filename=f"data/train-{i:05d}-of-00066.parquet",
                         local_dir="data/raw/hoa_kidney_right_shards")
(if the connection is unreliable, curl with --retry/-C - against
https://huggingface.co/datasets/eternalaudrey/hipct_3d/resolve/main/data/
train-{i:05d}-of-00066.parquet worked more reliably than the Python client
in this environment -- see baki.sh.)
"""

import io
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
import tifffile

from config import HOA_KIDNEY_MULTISPACING_CONFIG
from utils.convert_to_memmap import MEMMAP_DIR

SHARD_DIR = "data/raw/hoa_kidney_right_shards"

# organ_key (matches HOA_KIDNEY_MULTISPACING_CONFIG) -> (volume name in the
# parquet's `name` column, shard indices it's scattered across).
VOLUME_MAP = {
    "kidney_hoa17_25um": ("LADAF-2021-17_kidney_right_complete-organ_25.0um_bm05", [42, 43]),
    "kidney_hoa17_6um": ("LADAF-2021-17_kidney_right_VOI-02_6.5um_bm05", [43, 44]),
    "kidney_hoa17_2um": ("LADAF-2021-17_kidney_right_VOI-01.1_2.6um_bm05", [42]),
}

_PATCH_RE = re.compile(r"patch(\d+)\.tiff")


def _collect_patches(volume_name, shard_indices):
    """Returns [(patch_index, tiff_bytes), ...] for every row matching
    `volume_name` across the given shards, unsorted (caller sorts)."""
    patches = []
    for i in shard_indices:
        path = os.path.join(SHARD_DIR, f"train-{i:05d}-of-00066.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- download the required shards first "
                f"(see this module's docstring)."
            )
        pf = pq.ParquetFile(path)
        table = pf.read(columns=["name", "image"])
        names = table.column("name").to_pylist()
        images = table.column("image").to_pylist()
        for name, img in zip(names, images):
            if name != volume_name:
                continue
            m = _PATCH_RE.search(img["path"])
            if m is None:
                raise ValueError(f"Could not parse patch index from '{img['path']}'")
            patches.append((int(m.group(1)), img["bytes"]))
    return patches


def _write_meta(organ_key, shape, dtype):
    meta_path = os.path.join(MEMMAP_DIR, f"{organ_key}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"shape": list(shape), "dtype": str(dtype)}, f)


def convert_all():
    for organ_key in HOA_KIDNEY_MULTISPACING_CONFIG:
        if organ_key not in VOLUME_MAP:
            print(f"[WARNING] No VOLUME_MAP entry for '{organ_key}', skipping")
            continue
        volume_name, shard_indices = VOLUME_MAP[organ_key]
        patches = _collect_patches(volume_name, shard_indices)
        patches.sort(key=lambda t: t[0])
        expected = list(range(len(patches)))
        if [p[0] for p in patches] != expected:
            raise ValueError(f"'{organ_key}': patch indices not contiguous from 0")

        print(f"[*] {organ_key}: decoding {len(patches)} patches...")
        slabs = [tifffile.imread(io.BytesIO(b)) for _, b in patches]
        volume = np.concatenate(slabs, axis=0)
        del slabs

        os.makedirs(MEMMAP_DIR, exist_ok=True)
        data_path = os.path.join(MEMMAP_DIR, f"{organ_key}.dat")
        mmap = np.memmap(data_path, dtype=volume.dtype, mode="w+", shape=volume.shape)
        mmap[:] = volume
        mmap.flush()
        del mmap
        _write_meta(organ_key, volume.shape, volume.dtype)
        del volume

        print(f"[✓] {organ_key} ready: {os.path.getsize(data_path) / 1e9:.2f} GB -> {data_path}")


if __name__ == "__main__":
    convert_all()
