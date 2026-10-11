# utils/kidney_metadata_verification.py
"""
Metadata table for the SenNet+HOA "blood-vessel-segmentation" Kaggle dataset
(data/blood-vessel-segmentation.zip), which has expert kidney vessel labels
(the other organs only have Frangi pseudo-labels).

Voxel spacing is not in the TIFF files (XResolution/YResolution are just
(1,1)) and there is no metadata file for it in the zip (only train_rles.csv and
sample_submission.csv besides train/test). It is given in the dataset paper:

  "Vasculature segmentation in 3D hierarchical phase-contrast tomography
  images of human kidneys" -- https://pmc.ncbi.nlm.nih.gov/articles/PMC13470036/
  (Table 1 / Methods)

All spacings below are from that paper. Slice/label counts are from
`unzip -l` on the zip.

The test kidneys (kidney_5, kidney_6) had no public labels during the
competition. Jain et al. later released them, together with an extra 13um VOI
of donor 1 (6.5um scan binned 2x), on Zenodo (10.5281/zenodo.19685381); get
them with utils/fetch_zenodo_subset.py. These are test-only and are never used
for training (the training scripts only pick split == "train").

Note on the HOA volumes in config.py (HOA_KIDNEY_MULTISPACING_CONFIG): the 6.5um
VOI-02 is the same scan as kidney_additional_voi_13um, and kidney_1_voi (5.2um)
is very likely the 2.6um VOI-01.1 binned 2x. Training on those would leak test
data.
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
    ("kidney_5", "LADAF-2021-17-Left", 50.28, 1012, 1012, "public_test", "dense",
     "documented (PMC13470036, Table 1); labels released on Zenodo 19685381"),
    ("kidney_6", "LADAF-2022-13-Bottom", 63.08, 501, 501, "private_test", "dense",
     "documented (PMC13470036, Table 1); beamline BM18; labels released on Zenodo 19685381"),
    ("kidney_additional_voi_13um", "LADAF-2021-17-Right", 13.00, 2226, 5955, "extra_test",
     "labelled (density not stated)",
     "PMC13470036 (6.5um VOI binned to 13um); Zenodo 19685381; image data only for slices 0-2225"),
]

TEST_SPLITS = ("public_test", "private_test", "extra_test")


def build_csv():
    rows = []
    for (sample_id, specimen_id, spacing_um, n_img, n_lbl, split, density, source) in VERIFIED_SAMPLES:
        if split == "train":
            gt_available = "YES (expert annotation)"
            folder = f"train/{sample_id}"
        else:
            gt_available = "YES (released after the competition, test only)"
            folder = f"zenodo/{sample_id}"
        rows.append({
            "sample_id": sample_id,
            "kidney_specimen_id": specimen_id,
            "image_path": f"{folder}/images/",
            "mask_path": f"{folder}/labels/" if n_lbl > 0 else "N/A",
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
        print(f"  {row['sample_id']:26s} | {row['voxel_spacing_um']:>6.2f} um | "
              f"{row['n_label_slices']:>5d} label slices | {row['vessel_gt_available']}")

    print("\n[Multi-resolution same-specimen check]")
    print("  kidney_1_dense (50.00 um) and kidney_1_voi (5.20 um) are the same specimen "
          "(LADAF-2021-17-Right) at two resolutions (~10x apart), both with dense expert labels. "
          "Train on kidney_1_dense/kidney_2/kidney_3_* (~50 um) and test on the held-out "
          "kidney_1_voi (5.2 um) as an unseen resolution.")


if __name__ == "__main__":
    build_csv()
