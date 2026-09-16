# utils/kidney_metadata_verification.py
"""
Verified metadata table for the SenNet+HOA "blood-vessel-segmentation" Kaggle
dataset (data/blood-vessel-segmentation.zip), providing REAL expert-annotated
kidney vessel ground truth (unlike the Frangi pseudo-labels used for the other
3 organs in this project).

Voxel spacing is NOT embedded in the TIFF files (checked directly: XResolution/
YResolution tags are dummy placeholder values (1,1)) and is not present as a
metadata file inside the zip (checked: only train_rles.csv and
sample_submission.csv exist outside train/test folders). It is documented in
the dataset's official companion paper:

  "Vasculature segmentation in 3D hierarchical phase-contrast tomography
  images of human kidneys" -- https://pmc.ncbi.nlm.nih.gov/articles/PMC13470036/
  (Table 1 / Methods)

Every spacing value below is taken directly from that source and is marked
"documented" -- none are guessed or inferred. Slice/label counts are counted
directly from the actual zip contents (`unzip -l`), not assumed.
"""

import csv

OUTPUT_CSV = "data/kidney_metadata_verified.csv"

# (sample_id, kidney_specimen_id, spacing_um, n_image_slices, n_label_slices,
#  split, annotation_density, spacing_source)
VERIFIED_SAMPLES = [
    ("kidney_1_dense", "LADAF-2021-17-Right", 50.00, 2279, 2279, "train", "dense",
     "documented (PMC13470036, Table 1)"),
    ("kidney_1_voi", "LADAF-2021-17-Right", 5.20, 1397, 1397, "train", "dense",
     "documented (PMC13470036, Table 1)"),
    ("kidney_2", "S-20-28-Kidney", 50.00, 2217, 2217, "train", "sparse (~65%)",
     "documented (PMC13470036, Table 1)"),
    ("kidney_3_dense", "LADAF-2020-27-Kidney-2", 50.16, 0, 501, "train", "dense (85%+)",
     "documented (PMC13470036, Table 1); shares raw images with kidney_3_sparse"),
    ("kidney_3_sparse", "LADAF-2020-27-Kidney-2", 50.16, 1035, 1035, "train", "sparse",
     "documented (PMC13470036, Table 1)"),
    ("kidney_5", "LADAF-2021-17-Left", 50.28, 3, 0, "public_test", "dense (labels NOT public)",
     "documented (PMC13470036, Table 1); zip contains only 3 placeholder images, no labels"),
    ("kidney_6", "LADAF-2022-13-Bottom", 63.08, 3, 0, "private_test", "dense (labels NOT public)",
     "documented (PMC13470036, Table 1); zip contains only 3 placeholder images, no labels"),
]


def build_csv():
    rows = []
    for (sample_id, specimen_id, spacing_um, n_img, n_lbl, split, density, source) in VERIFIED_SAMPLES:
        gt_available = "YES (real expert annotation)" if n_lbl > 0 and split == "train" else "NO (test set, labels withheld by competition)"
        rows.append({
            "sample_id": sample_id,
            "kidney_specimen_id": specimen_id,
            "image_path": f"train/{sample_id}/images/" if split == "train" else f"test/{sample_id}/images/",
            "mask_path": f"train/{sample_id}/labels/" if (split == "train" and n_lbl > 0) else "N/A",
            "voxel_spacing_um": spacing_um,
            "n_image_slices": n_img,
            "n_label_slices": n_lbl,
            "split": split,
            "annotation_density": density,
            "vessel_gt_available": gt_available,
            "spacing_documented_or_inferred": source,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] Wrote {OUTPUT_CSV} with {len(rows)} samples.")
    for row in rows:
        print(f"  {row['sample_id']:16s} | {row['voxel_spacing_um']:>6.2f} um | "
              f"{row['n_label_slices']:>5d} label slices | {row['vessel_gt_available']}")

    print("\n[Multi-resolution same-specimen check]")
    print("  kidney_1_dense (50.00 um) and kidney_1_voi (5.20 um) are the SAME physical "
          "specimen (LADAF-2021-17-Right) at two real, distinct resolutions, both with real "
          "dense expert annotation. This is a genuine same-organ multi-resolution pair -- "
          "not synthetic, not pseudo-labeled. It supports a real (if limited: one specimen, "
          "two resolutions, ~10x apart) same-organ resolution-generalization experiment: "
          "train on kidney_1_dense/kidney_2/kidney_3_* (~50 um) and evaluate on the held-out "
          "kidney_1_voi (5.2 um) as a genuinely unseen resolution.")


if __name__ == "__main__":
    build_csv()
