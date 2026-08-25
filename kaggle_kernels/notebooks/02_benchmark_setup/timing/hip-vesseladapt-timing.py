"""
Kaggle GPU kernel that runs a list of jobs on both GPUs of a "T4 x2" session,
one queue per GPU, so a session hour gives two GPU hours of work.

A job is a list of shell commands run in the project folder with
CUDA_VISIBLE_DEVICES set to its GPU. Typical jobs:
  - train one configuration with train_lora_v2.py, then write its benchmark
    predictions (benchmark/infer.py) for the validation and test ROIs;
  - write benchmark predictions for an existing checkpoint (first-round
    models, SAM-Med3D without fine-tuning).
Scoring (benchmark/score.py) is CPU-only and is done afterwards from the saved
predictions, so it does not use GPU time.

Inputs attached to the kernel (found by marker file, not by path):
  - code dataset (marker: train_lora_v2.py)
  - competition data blood-vessel-segmentation (marker: train/kidney_1_dense)
  - output of kernel hip-vesseladapt-100ep-gpu-train (marker: sam_med3d.pth);
    it has the SAM-Med3D checkpoint and the first-round models
  - optional: ROI dataset (marker: manifest.json next to kidney_6/)
Everything is written to /kaggle/working/out.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

JOBS_GPU0 = [
 {
  "name": "timing_spacing_log_aug_refine",
  "cmds": [
   "python train_lora_v2.py --model spacing --spacing-encoding log --patch 128 --fg-prob 0.5 --fov-scale 0.25 2.0 --refine --epochs 2 --iters 50 --seed 0 --workers 4 --raw-tiff-root $COMP_TRAIN --out $OUT/checkpoints/timing_a",
   "python -c \"import time,torch,numpy as np; from benchmark.infer import build_model, predict_roi; m=build_model('spacing_lora','$OUT/checkpoints/timing_a/model.pth',refine=True,device='cuda',spacing_encoding='log'); [ (lambda s: (lambda t: (predict_roi(m,np.random.randint(0,60000,(s,s,s)).astype(np.uint16),5.2,128,'cuda'), print('patch128 native', s, round(time.time()-t,1),'s', flush=True)))(time.time()))(s) for s in (192,640)]; t=time.time(); predict_roi(m,np.random.randint(0,60000,(640,640,640)).astype(np.uint16),5.2,128,'cuda',resample_um=50.0); print('patch128 fov50 640', round(time.time()-t,1),'s')\""
  ]
 }
]  # filled in by the launcher (kaggle_kernels/launch_jobs.py)
JOBS_GPU1 = [
 {
  "name": "timing_plain_noaug",
  "cmds": [
   "python train_lora_v2.py --model plain --patch 128 --fg-prob 0.5 --epochs 2 --iters 50 --seed 0 --workers 4 --raw-tiff-root $COMP_TRAIN --out $OUT/checkpoints/timing_b",
   "python -c \"import time,torch,numpy as np; from benchmark.infer import build_model, predict_roi; m=build_model('plain_lora','$FIRST_ROUND/checkpoints_plain_lora/best_model_plain_lora.pth',device='cuda'); [ (lambda s: (lambda t: (predict_roi(m,np.random.randint(0,60000,(s,s,s)).astype(np.uint16),5.2,64,'cuda'), print('patch64 native', s, round(time.time()-t,1),'s', flush=True)))(time.time()))(s) for s in (192,640)]\""
  ]
 }
]

SAM_MED3D_COMMIT = "bd7d1620df0f85fade74aa9e44262de953b559f1"
CLDICE_COMMIT = "47d31a6cc4a8101b1ffe8052994821961e57af9f"
WORK_DIR = "/kaggle/working/hip-vesseladapt"
OUT = "/kaggle/working/out"


def find_by_marker(root, is_match, max_depth=9):
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        if not os.path.isdir(path):
            continue
        if is_match(path):
            return path
        if depth < max_depth:
            try:
                stack += [(os.path.join(path, n), depth + 1) for n in sorted(os.listdir(path))]
            except PermissionError:
                pass
    return None


t0 = time.time()
CODE_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "train_lora_v2.py")))
COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
SAM_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "sam_med3d.pth")))
FIRST_ROUND = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "checkpoints_plain_lora")))
ROI_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "manifest.json"))
                         and os.path.isdir(os.path.join(p, "kidney_6")))
VAL_ROI_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "manifest.json"))
                             and os.path.isdir(os.path.join(p, "kidney_1_dense")))
for name, d in (("code", CODE_DIR), ("competition", COMP_DIR), ("SAM-Med3D ckpt", SAM_DIR),
                ("first-round output", FIRST_ROUND), ("test ROIs", ROI_DIR), ("val ROIs", VAL_ROI_DIR)):
    print(f"[*] {name}: {d}", flush=True)
if CODE_DIR is None or COMP_DIR is None:
    raise RuntimeError("code or competition data not attached")

os.makedirs(WORK_DIR, exist_ok=True)
shutil.copytree(CODE_DIR, WORK_DIR, dirs_exist_ok=True)
for z in ("models.zip", "utils.zip", "losses.zip", "data.zip", "benchmark.zip"):
    zp = os.path.join(WORK_DIR, z)
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(WORK_DIR)
        os.remove(zp)
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1", "tifffile", "imagecodecs"], check=True)
for url, dst, commit in (("https://github.com/openmedlab/SAM-Med3D.git", "external/SAM-Med3D", SAM_MED3D_COMMIT),
                         ("https://github.com/jocpae/clDice.git", "external/clDice", CLDICE_COMMIT)):
    if not os.path.isdir(dst):
        subprocess.run(["git", "clone", "-q", url, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", commit], check=True)
from utils.patch_external_repos import patch_sam_med3d
patch_sam_med3d()
os.makedirs("external/SAM-Med3D/ckpt", exist_ok=True)
ckpt = "external/SAM-Med3D/ckpt/sam_med3d.pth"
if not os.path.exists(ckpt):
    if SAM_DIR:
        os.symlink(os.path.join(SAM_DIR, "sam_med3d.pth"), ckpt)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        subprocess.run([sys.executable, "-m", "gdown", "https://drive.google.com/uc?id=1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub",
                        "-O", ckpt], check=True)

import torch
print(f"[*] torch {torch.__version__}, {torch.cuda.device_count()} GPU(s): "
      f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)
subprocess.run(["nvidia-smi"])

env_vars = {
    "COMP_TRAIN": os.path.join(COMP_DIR, "train"),
    "ROI_DIR": ROI_DIR or "",
    "VAL_ROI_DIR": VAL_ROI_DIR or "",
    "FIRST_ROUND": FIRST_ROUND or "",
    "OUT": OUT,
}
os.makedirs(OUT, exist_ok=True)
print(f"[*] setup took {time.time() - t0:.0f}s", flush=True)


def run_queue(gpu, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_ALLOC_CONF="expandable_segments:True", **env_vars)
    for job in jobs:
        name = job["name"]
        log_path = os.path.join(OUT, f"{name}.log")
        t = time.time()
        with open(log_path, "w") as log:
            for cmd in job["cmds"]:
                log.write(f"$ {cmd}\n")
                log.flush()
                r = subprocess.run(cmd, shell=True, cwd=WORK_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    log.write(f"[!] exit code {r.returncode}\n")
                    print(f"[!] GPU{gpu} job {name}: command failed ({r.returncode}): {cmd}", flush=True)
                    break
        print(f"[✓] GPU{gpu} job {name} finished in {(time.time() - t) / 60:.1f} min", flush=True)
        with open(log_path) as f:
            tail = f.read()[-3000:]
        print(f"----- tail of {name}.log -----\n{tail}\n-----", flush=True)


n_gpu = torch.cuda.device_count()
queues = [JOBS_GPU0, JOBS_GPU1]
if n_gpu < 2:
    queues = [JOBS_GPU0 + JOBS_GPU1]
threads = [threading.Thread(target=run_queue, args=(g, q)) for g, q in enumerate(queues) if q]
for th in threads:
    th.start()
for th in threads:
    th.join()
# checkpoint folders can be large; keep the model weights and logs, drop the resume state
for root, _, files in os.walk(os.path.join(OUT, "checkpoints")):
    for f in files:
        if f in ("state.pth",):
            os.remove(os.path.join(root, f))
print(f"[✓] ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
