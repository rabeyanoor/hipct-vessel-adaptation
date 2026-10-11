# benchmark/nifti_io.py
"""
NIfTI bridge for external methods (nnU-Net, vesselFM).

  export: writes every benchmark ROI image as <out>/<sample_id>__roi_XX_0000.nii.gz
          (nnU-Net naming; vesselFM reads any .nii.gz in a folder). Spacing is
          stored in mm. nnU-Net resamples inputs to its training spacing, so
          with the true spacing it effectively runs "FOV-matched"; with
          --fake-spacing-um 50 every ROI is labelled 50um and processed on its
          native voxel grid.
  import: reads <in>/<sample_id>__roi_XX.nii.gz (or ..._0000.nii.gz) predictions
          back into data/benchmark/preds/<run>/ for benchmark/score.py.

Usage:
    python -m benchmark.nifti_io export --out data/benchmark/nifti_rois
    python -m benchmark.nifti_io import --pred-in nnunet_out --run-name nnunet_250ep
"""

import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk

from benchmark.rois import ROI_DIR, load_manifest

PRED_ROOT = "data/benchmark/preds"


def export(out_dir, roi_dir=ROI_DIR, sets=None, labels_dir=None, fake_spacing_um=None):
    manifest = load_manifest(roi_dir)
    os.makedirs(out_dir, exist_ok=True)
    for sid, entries in manifest.items():
        if sets and sid not in sets:
            continue
        for e in entries:
            d = np.load(e["path"])
            sp_mm = (fake_spacing_um or e["spacing_um"]) / 1000.0
            img = sitk.GetImageFromArray(d["image"].astype(np.float32))
            img.SetSpacing((sp_mm,) * 3)
            sitk.WriteImage(img, os.path.join(out_dir, f"{sid}__roi_{e['roi']:02d}_0000.nii.gz"))
            if labels_dir:
                os.makedirs(labels_dir, exist_ok=True)
                lbl = sitk.GetImageFromArray(d["label"])
                lbl.SetSpacing((sp_mm,) * 3)
                sitk.WriteImage(lbl, os.path.join(labels_dir, f"{sid}__roi_{e['roi']:02d}.nii.gz"))
    print(f"[✓] exported ROIs to {out_dir}")


def import_preds(pred_in, run_name, roi_dir=ROI_DIR, pred_root=PRED_ROOT, threshold=0.5, suffix=".nii.gz"):
    manifest = load_manifest(roi_dir)
    n = 0
    for sid, entries in manifest.items():
        for e in entries:
            stem = f"{sid}__roi_{e['roi']:02d}"
            cands = glob.glob(os.path.join(pred_in, stem + "*" + suffix))
            if not cands:
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(sorted(cands)[0]))
            pred = (arr > threshold).astype(np.uint8)
            if pred.shape != tuple(e["size_zyx"]):
                raise ValueError(f"{cands[0]}: shape {pred.shape} != {e['size_zyx']}")
            os.makedirs(os.path.join(pred_root, run_name, sid), exist_ok=True)
            np.savez_compressed(os.path.join(pred_root, run_name, sid, f"roi_{e['roi']:02d}.npz"), pred=pred)
            n += 1
    print(f"[✓] imported {n} predictions into {os.path.join(pred_root, run_name)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["export", "import"])
    p.add_argument("--out", default="data/benchmark/nifti_rois")
    p.add_argument("--labels-out", default=None)
    p.add_argument("--pred-in")
    p.add_argument("--run-name")
    p.add_argument("--roi-dir", default=ROI_DIR)
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--fake-spacing-um", type=float, default=None)
    p.add_argument("--suffix", default=".nii.gz", help='e.g. "_pred.nii.gz" for vesselFM output')
    a = p.parse_args()
    if a.mode == "export":
        export(a.out, a.roi_dir, a.sets, a.labels_out, a.fake_spacing_um)
    else:
        import_preds(a.pred_in, a.run_name, a.roi_dir, suffix=a.suffix)
