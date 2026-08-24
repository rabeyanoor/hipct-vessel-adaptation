# benchmark/run_vesselfm.py
"""
vesselFM (Wittmann et al., CVPR 2025) on the benchmark ROIs, without any
fine-tuning, on the native voxel grid.

vesselFM's training data include HiP-CT kidney (2.5um), so on these data it is
not a strict zero-shot model. HiP-CT vessels are darker than the tissue around
them; vesselFM has a test-time switch that inverts bright-background images
(tta.invert). We run it with and without the switch and report both, instead
of choosing one on the test ROIs.

Needs a clone of https://github.com/bwittmann/vesselFM (pinned commit below)
run from its OWN virtual environment (the clone is put on PYTHONPATH): vesselFM pins
torch 2.5.1, numpy 1.26 and monai 1.3 and imports skimage's skeletonize_3d,
which newer scikit-image versions no longer have. --python points to that
environment's interpreter. Internet access is needed (the checkpoint is
downloaded from Hugging Face by vesselFM's inference script).

Usage:
    uv venv -p 3.11 /tmp/vfm_env   # scikit-image 0.24 has no wheels for newer Pythons
    uv pip install --python /tmp/vfm_env/bin/python torch==2.5.1 numpy==1.26.4 monai==1.3.2 hydra-core==1.3.2 scikit-image==0.24.0 \
        scikit-learn==1.5.2 SimpleITK==2.4.0 tqdm==4.67.0 huggingface_hub==0.29.1 nibabel   # what inference.py imports
    python -m benchmark.run_vesselfm --vesselfm-dir /tmp/vesselFM --python /tmp/vfm_env/bin/python \
        --roi-dir <rois> --pred-root <out>/preds
"""

import argparse
import os
import subprocess
import sys
import tempfile

VESSELFM_COMMIT = "d33b952223220fb40caa1f4daa4b627256de9508"


def run(vesselfm_dir, roi_dir, pred_root, invert, python=sys.executable, device="cuda:0", batch_size=2):
    from benchmark import nifti_io
    tag = "invert" if invert else "noinvert"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        src, dst = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        nifti_io.export(src, roi_dir=roi_dir)
        # ckpt_path does not exist on purpose: the script then downloads vesselFM_all.pt from Hugging Face
        subprocess.run([python, "vesselfm/seg/inference.py", f"image_path={src}", f"output_folder={dst}",
                        "ckpt_path=/nonexistent.pt", f"device={device}", f"tta.invert={str(invert).lower()}",
                        f"batch_size={batch_size}"],
                       cwd=vesselfm_dir, check=True,
                       env=dict(os.environ, PYTHONPATH=os.path.abspath(vesselfm_dir),
                                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        nifti_io.import_preds(dst, f"vesselfm_{tag}", roi_dir=roi_dir, pred_root=pred_root,
                              suffix="_pred.nii.gz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vesselfm-dir", required=True)
    p.add_argument("--roi-dir", required=True)
    p.add_argument("--pred-root", required=True)
    p.add_argument("--python", default=sys.executable, help="interpreter of the vesselFM environment")
    p.add_argument("--device", default="cuda:0")
    # patches per sliding-window step; vesselFM's default of 8 runs out of memory on a 16 GB T4 for
    # the 640^3 ROIs. It changes only speed and memory, not the prediction.
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--invert", choices=["yes", "no", "both"], default="both")
    a = p.parse_args()
    for inv in {"yes": [True], "no": [False], "both": [False, True]}[a.invert]:
        run(a.vesselfm_dir, a.roi_dir, a.pred_root, inv, a.python, a.device, a.batch_size)
