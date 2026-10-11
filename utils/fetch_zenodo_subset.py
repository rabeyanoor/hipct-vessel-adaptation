# utils/fetch_zenodo_subset.py
"""
Downloads only the parts we need from the Jain et al. (2026) data release on
Zenodo (10.5281/zenodo.19685381, CC-BY-4.0) instead of the whole 51 GB zip.

The zip is read over HTTP range requests: first the central directory, then
only the members under the requested folders. Useful subsets:

  kidney_5                    competition-data/test/kidney_5   (50.28um, labels, ~6.5 GB)
  kidney_6                    competition-data/test/kidney_6   (63.08um, BM18, labels, ~1.9 GB)
  kidney_additional_voi_13um  additional_dataset_files/...     (13um, labels, ~38 GB;
                              in this release only slices 0-2225 have image data,
                              the other 3729 image files are empty)
  team_predictions            the five winning teams' RLE predictions for the three
                              sets above (~0.5 GB)
  kidney_1_voi                competition-data/train/kidney_1_voi (5.2um, ~11 GB), the
                              same data as in the Kaggle competition download

Files are written as data/zenodo/<name>/{images,labels}/*.tif, which is the
layout RealGTKidneyDataset's raw_tiff_root mode expects (use
raw_tiff_root="data/zenodo").

Usage:
    python -m utils.fetch_zenodo_subset kidney_5 kidney_6
    python -m utils.fetch_zenodo_subset kidney_additional_voi_13um --z-range 0 2226
"""

import argparse
import concurrent.futures
import io
import os
import pickle
import struct
import time
import urllib.request
import zipfile
import zlib

ZIP_URL = ("https://zenodo.org/api/records/19685382/files/"
           "final-data-repository-for-paper.zip/content")
ROOT = "final-data-repository-for-paper/"
SUBSETS = {
    "kidney_5": "competition-data/test/kidney_5/",
    "kidney_6": "competition-data/test/kidney_6/",
    "kidney_additional_voi_13um": "additional_dataset_files/kidney_additional_voi_13um/",
    "team_predictions": "",
    "kidney_1_voi": "competition-data/train/kidney_1_voi/",
}
_TEAM_PRED_PREFIXES = (
    "prediction-files-for-teams/competition-test-set/",
    "additional_dataset_files/k4_predictions_additional_data/team-",
)
OUT_DIR = "data/zenodo"


class _RangeFile(io.RawIOBase):
    """Read-only file object backed by HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read_range(self, start, length, attempts=10):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) == length:
                    return data
                raise IOError(f"short read {len(data)}/{length}")
            except Exception as e:  # network hiccups and Zenodo rate limits (429)
                if attempt == attempts - 1:
                    raise
                wait = min(300, 5 * 2 ** attempt)
                print(f"    retry in {wait}s: {e}", flush=True)
                time.sleep(wait)

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.read_range(self.pos, n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _read_member(rf, info):
    # local header is 30 bytes + name + extra; the extra length can differ
    # from the central directory, so read it from the local header itself
    head = rf.read_range(info.header_offset, 30)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    raw = rf.read_range(start, info.compress_size) if info.compress_size else b""
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")


def _slice_index(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return int(stem) if stem.isdigit() else None


def _infos(rf, out_dir):
    # reading the central directory (33k entries) takes a while, so cache it
    index = os.path.join(out_dir, ".zip_index.pkl")
    if os.path.exists(index):
        with open(index, "rb") as f:
            return pickle.load(f)
    infos = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)).infolist()
    os.makedirs(out_dir, exist_ok=True)
    with open(index, "wb") as f:
        pickle.dump(infos, f)
    return infos


def list_slices(name, out_dir=OUT_DIR):
    """[(slice_name, image_bytes, label_bytes)] for a volume, sorted by name,
    without downloading anything but the zip index."""
    rf = _RangeFile(ZIP_URL)
    prefix = ROOT + SUBSETS[name]
    sizes = {}
    for i in _infos(rf, out_dir):
        if i.filename.startswith(prefix) and not i.is_dir():
            kind, fname = i.filename[len(prefix):].split("/", 1)
            sizes.setdefault(fname, {})[kind] = i.file_size
    return [(n, s.get("images", 0), s.get("labels", 0)) for n, s in sorted(sizes.items())]


def fetch_files(name, slice_names, kinds=("images", "labels"), out_dir=OUT_DIR, workers=4):
    """Download specific slices (by file name) of one volume."""
    rf = _RangeFile(ZIP_URL)
    prefix = ROOT + SUBSETS[name]
    by_name = {i.filename: i for i in _infos(rf, out_dir)}
    todo = [by_name[f"{prefix}{k}/{n}"] for n in slice_names for k in kinds if f"{prefix}{k}/{n}" in by_name]

    def get(info):
        dst = os.path.join(out_dir, name, info.filename[len(prefix):])
        if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst + ".part", "wb") as f:
            f.write(_read_member(rf, info))
        os.replace(dst + ".part", dst)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, todo))


def fetch(names, every=1, z_range=None, out_dir=OUT_DIR, dry_run=False, workers=4):
    rf = _RangeFile(ZIP_URL)
    infos = _infos(rf, out_dir)
    for name in names:
        prefix = ROOT + SUBSETS[name]
        if name == "team_predictions":
            members = [i for i in infos if not i.is_dir() and i.filename.endswith(".csv")
                       and any(i.filename.startswith(ROOT + p) for p in _TEAM_PRED_PREFIXES)]
        else:
            members = [i for i in infos if i.filename.startswith(prefix) and not i.is_dir()]
        selected = []
        for i in members:
            z = _slice_index(i.filename)
            if z is not None:
                if z_range and not (z_range[0] <= z < z_range[1]):
                    continue
                if z % every:
                    continue
            selected.append(i)
        total = sum(i.file_size for i in selected)
        print(f"[*] {name}: {len(selected)}/{len(members)} files, {total / 1e9:.2f} GB")
        if dry_run:
            continue
        def get(info):
            rel = (os.path.basename(info.filename) if name == "team_predictions"
                   else info.filename[len(prefix):])
            dst = os.path.join(out_dir, name, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                return
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            data = _read_member(rf, info)
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)

        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for k, _ in enumerate(pool.map(get, selected)):
                if (k + 1) % 100 == 0 or k == len(selected) - 1:
                    print(f"    {name}: {k + 1}/{len(selected)}", flush=True)
    print("[✓] done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=sorted(SUBSETS))
    p.add_argument("--every", type=int, default=1, help="keep every n-th slice")
    p.add_argument("--z-range", type=int, nargs=2, default=None, help="slice index range [start, end)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    fetch(a.names, every=a.every, z_range=a.z_range, out_dir=a.out_dir, dry_run=a.dry_run,
          workers=a.workers)
