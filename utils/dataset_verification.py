# utils/dataset_verification.py
"""
Step 0 data verification: scans all HiP-CT data actually present in this project,
cross-checks the voxel spacing declared in config.py against the spacing embedded
in the raw ESRF filenames, checks for real ground-truth annotation availability,
and writes the result to dataset_verification.csv.

This is a factual inventory, not a judgment call -- it exists so that any
downstream research-design decision (e.g. whether a real multi-resolution
out-of-distribution experiment is possible with the data on this machine) is
based on verified facts rather than assumption.
"""

import csv
import glob
import json
import os
import re
import sys

sys.path.append(os.path.abspath("."))
from config import ORGAN_CONFIG

OUTPUT_CSV = "dataset_verification.csv"

# Any file whose name contains one of these tokens would indicate a real
# annotation/label file shipped alongside the raw scan.
ANNOTATION_TOKENS = ["mask", "label", "seg", "annotation", "gt", "groundtruth"]


def find_embedded_spacing(raw_dir):
    """Parse the '<number>um' prefix that ESRF embeds in its raw filenames."""
    jp2_files = glob.glob(os.path.join(raw_dir, "**", "*.jp2"), recursive=True)
    if not jp2_files:
        return None
    match = re.search(r"([\d.]+)um", os.path.basename(jp2_files[0]))
    return float(match.group(1)) if match else None


def find_annotation_files(raw_dir):
    """Look for any file under raw_dir whose name suggests a real annotation."""
    all_files = glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
    hits = []
    for f in all_files:
        name_lower = os.path.basename(f).lower()
        if any(token in name_lower for token in ANNOTATION_TOKENS):
            hits.append(f)
    return hits


def get_memmap_shape(organ):
    meta_path = os.path.join("data", "memmap", f"{organ}_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)["shape"]  # (D, H, W)


def verify():
    rows = []
    for organ, meta in ORGAN_CONFIG.items():
        raw_dir = meta["raw_dir"]
        declared_spacing = meta["spacing"][0]  # isotropic in this dataset
        embedded_spacing = find_embedded_spacing(raw_dir)
        n_slices = len(glob.glob(os.path.join(raw_dir, "**", "*.jp2"), recursive=True))
        annotation_files = find_annotation_files(raw_dir)
        shape = get_memmap_shape(organ)

        spacing_match = (
            "MATCH" if embedded_spacing is not None
            and abs(embedded_spacing - declared_spacing) < 1e-3
            else "MISMATCH" if embedded_spacing is not None
            else "UNKNOWN (no jp2 files found)"
        )

        row = {
            "organ": organ,
            "raw_dir": raw_dir,
            "n_slices": n_slices,
            "declared_spacing_um": declared_spacing,
            "embedded_filename_spacing_um": embedded_spacing,
            "spacing_check": spacing_match,
            "memmap_shape_dhw": shape,
            "real_annotation_files_found": len(annotation_files),
            "vessel_gt_available": "NO -- pseudo-label only (Frangi vesselness)" if not annotation_files else f"POSSIBLE ({len(annotation_files)} files, needs manual check)",
        }
        rows.append(row)
        print(f"[*] {organ}: {row}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✓] Wrote {OUTPUT_CSV} with {len(rows)} organ(s).")

    # Explicit decision-rule verdict, per the multi-resolution OOD verification protocol.
    unique_spacings = {row["declared_spacing_um"] for row in rows}
    has_real_gt = any("POSSIBLE" in row["vessel_gt_available"] for row in rows)
    print("\n[Decision Rule Verdict]")
    print(f"  Organs available: {len(rows)}, each at ONE distinct spacing "
          f"({sorted(unique_spacings)} um) -- NOT multiple resolutions of the SAME organ.")
    print(f"  Real expert vessel annotations found: {has_real_gt}")
    if not has_real_gt:
        print("  => Case B/C: no real multi-resolution-same-organ + GT setup exists in "
              "this project's downloaded data. A same-organ multi-resolution OOD experiment "
              "would require downloading additional scans of the same specimen at other "
              "resolutions from ESRF, or using semi-synthetic downsampling (must be labeled "
              "as such, not claimed as real generalization).")


if __name__ == "__main__":
    verify()
