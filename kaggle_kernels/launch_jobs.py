"""
Pushes kaggle_jobs_kernel.py as a Kaggle kernel with a given job list.

Usage:
    python kaggle_kernels/launch_jobs.py jobs.json --slug hip-vesseladapt-jobs-a [--gpu] [--rois]

jobs.json: {"gpu0": [job, ...], "gpu1": [job, ...]} with job = {"name": str, "cmds": [str, ...]};
for a TPU session (--tpu) {"tpu0": [...], ..., "tpu7": [...]}, one queue per TPU chip.
Credentials come from the usual Kaggle environment variables / config.
"""

import argparse
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = "rabeya100x"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--slug", required=True)
    p.add_argument("--write", metavar="PATH", help="only write the filled-in kernel script to PATH (e.g. for Colab)")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--tpu", action="store_true", help="TPU v5e-8 session (96 CPU cores, 8 TPU chips)")
    p.add_argument("--keep-state", action="store_true", help="keep state.pth of training runs (to resume them later)")
    p.add_argument("--no-competition", action="store_true",
                   help="do not attach the competition data (jobs that only run inference on the ROIs)")
    # sam-med3d-weights: private mirror of sam_med3d.pth (the Google Drive link gets rate-limited)
    p.add_argument("--datasets", nargs="*", default=[f"{OWNER}/hip-vesseladapt-code", f"{OWNER}/sam-med3d-weights"])
    p.add_argument("--kernels", nargs="*", default=[f"{OWNER}/hip-vesseladapt-100ep-gpu-train"])
    a = p.parse_args()

    jobs = json.load(open(a.jobs))
    src = open(os.path.join(HERE, "kaggle_jobs_kernel.py")).read()
    src = src.replace("JOBS_GPU0 = []", f"JOBS_GPU0 = {json.dumps(jobs.get('gpu0', []), indent=1)}", 1)
    src = src.replace("JOBS_GPU1 = []", f"JOBS_GPU1 = {json.dumps(jobs.get('gpu1', []), indent=1)}", 1)
    src = src.replace("KEEP_STATE = False", f"KEEP_STATE = {a.keep_state}", 1)
    tpu = [jobs[k] for k in sorted(jobs, key=lambda k: int(k[3:])) if k.startswith("tpu")]
    src = src.replace("JOBS_TPU = []", f"JOBS_TPU = {json.dumps(tpu, indent=1)}", 1)
    if a.write:
        open(a.write, "w").write(src)
        return
    meta = {
        "id": f"{OWNER}/{a.slug}", "title": a.slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not (a.no_gpu or a.tpu), "enable_tpu": a.tpu,
        "enable_internet": True, "dataset_sources": a.datasets,
        "competition_sources": [] if a.no_competition else ["blood-vessel-segmentation"], "kernel_sources": a.kernels, "model_sources": [],
    }
    if a.tpu:
        meta["machine_shape"] = "Tpu1VmV38"  # the name Kaggle uses for its TPU VM (a v5e-8 at present)
    elif not a.no_gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "kernel.py"), "w").write(src)
        json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
        subprocess.run(["kaggle", "kernels", "push", "-p", d], check=True)


if __name__ == "__main__":
    main()
