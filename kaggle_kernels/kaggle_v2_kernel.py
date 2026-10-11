"""
Kaggle kernel for the second round of experiments (MedIA version).

One run = train one configuration with train_lora_v2.py (best epoch chosen on
validation ROIs from the held-out last 20 % of kidney_1_dense), then run the
benchmark on the held-out test ROIs in three settings: native, FOV-matched
(resampled to 50um), and native with the spacing input set to 50um. Each
setting is scored twice: plain (threshold 0.5) and with flip TTA plus the
threshold / small-component removal chosen on the validation ROIs
(benchmark/postprocess.py). Everything is written to /kaggle/working/out.

Attach three inputs to the kernel:
  - the code dataset (rabeya100x/hip-vesseladapt-code, marker: train_lora_v2.py)
  - the competition data (blood-vessel-segmentation)
  - the ROI dataset built locally with benchmark/rois.py
    (folder with manifest.json; upload data/benchmark/rois as a dataset)

Edit RUN below for each configuration. A 100 epoch x 250 step run with 128^3
patches should take roughly 6-8 h on a T4 (estimate, check the first epochs
in the log); use RESUME_FROM to continue a run from a previous kernel output
if the 12 h limit is hit.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

RUN = {
    "name": "spacing_full_s0",          # plain_base_s0, spacing_base_s0, spacing_full_s0, ...
    "model": "spacing",                  # plain | spacing
    "patch": 128,
    "fg_prob": 0.5,
    "fov_scale": [0.25, 2.0],            # None for no spacing augmentation
    "refine": True,
    "epochs": 100,
    "iters": 250,
    "seed": 0,
    "lora_r": 4,
    "unfreeze_blocks": 0,      # e.g. 2 to also train the last two encoder blocks
    "decoder": "full",         # "full" or "output" (only the decoder's output layers)
    # set to a .pth path to skip training and only benchmark an existing model,
    # e.g. the first-round 100 epoch checkpoints (then use patch 64, refine False,
    # fov_scale None, and the matching model type)
    "checkpoint": None,
}
RESUME_FROM = None  # e.g. "/kaggle/input/<previous-output>/out/checkpoints/spacing_full_s0"

WORK_DIR = "/kaggle/working/hip-vesseladapt"
OUT = "/kaggle/working/out"


def find_by_marker(root, is_match, max_depth=5):
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        if not os.path.isdir(path):
            continue
        if is_match(path):
            return path
        if depth < max_depth:
            try:
                stack += [(os.path.join(path, n), depth + 1) for n in os.listdir(path)]
            except PermissionError:
                pass
    return None


CODE_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "train_lora_v2.py")))
COMP_DIR = find_by_marker("/kaggle/input", lambda p: os.path.isdir(os.path.join(p, "train", "kidney_1_dense")))
ROI_DIR = find_by_marker("/kaggle/input", lambda p: os.path.exists(os.path.join(p, "manifest.json")))
for name, d in (("code", CODE_DIR), ("competition data", COMP_DIR), ("ROI dataset", ROI_DIR)):
    if d is None:
        raise RuntimeError(f"could not find the {name} under /kaggle/input")
    print(f"[*] {name}: {d}")

# 1. code
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

# 2. dependencies, external repos, SAM-Med3D checkpoint
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai", "gdown"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/openmedlab/SAM-Med3D.git", "external/SAM-Med3D"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/jocpae/clDice.git", "external/clDice"], check=True)
from utils.patch_external_repos import patch_sam_med3d
patch_sam_med3d()
os.makedirs("external/SAM-Med3D/ckpt", exist_ok=True)
ckpt = "external/SAM-Med3D/ckpt/sam_med3d.pth"
if not os.path.exists(ckpt):
    subprocess.run([sys.executable, "-m", "gdown", "https://drive.google.com/uc?id=1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub",
                    "-O", ckpt], check=True)

# 3. validation ROIs (held-out slices of kidney_1_dense; built once per run, small)
from benchmark import rois as roi_mod
import train_lora_v2
VAL_DIR = "/kaggle/working/val_rois"
if not os.path.exists(os.path.join(VAL_DIR, "manifest.json")):
    roi_mod.make_rois(os.path.join(COMP_DIR, "train"), list(train_lora_v2.VAL_HOLDOUT), seed=0, out_dir=VAL_DIR,
                      n_rois={s: 6 for s in train_lora_v2.VAL_HOLDOUT}, z_frac=train_lora_v2.VAL_HOLDOUT)

# 4. train
run_dir = os.path.join(OUT, "checkpoints", RUN["name"])
if RESUME_FROM and not os.path.exists(os.path.join(run_dir, "state.pth")):
    shutil.copytree(RESUME_FROM, run_dir, dirs_exist_ok=True)
argv = ["--model", RUN["model"], "--out", run_dir, "--raw-tiff-root", os.path.join(COMP_DIR, "train"),
        "--patch", str(RUN["patch"]), "--fg-prob", str(RUN["fg_prob"]), "--epochs", str(RUN["epochs"]),
        "--iters", str(RUN["iters"]), "--seed", str(RUN["seed"]), "--lora-r", str(RUN["lora_r"]),
        "--unfreeze-blocks", str(RUN["unfreeze_blocks"]), "--decoder", RUN["decoder"],
        "--val-roi-dir", VAL_DIR]
if RUN["fov_scale"]:
    argv += ["--fov-scale", *map(str, RUN["fov_scale"])]
if RUN["refine"]:
    argv += ["--refine"]
if RUN["checkpoint"]:
    model_path = RUN["checkpoint"]
else:
    model_path = train_lora_v2.train(train_lora_v2.get_parser().parse_args(argv))

# 5. benchmark
from benchmark import infer, score, postprocess
model_name = "spacing_lora" if RUN["model"] == "spacing" else "plain_lora"
common = dict(refine=RUN["refine"], patch=RUN["patch"], pred_root=os.path.join(OUT, "preds"),
              lora_r=RUN["lora_r"], save_prob=True)

# threshold / min component size chosen on the validation ROIs (with TTA), never on test
val_dir = infer.run(model_name, model_path, f"{RUN['name']}_val", roi_dir=VAL_DIR, tta=True, **common)
postprocess.tune(val_dir, VAL_DIR)
pp_params = os.path.join(val_dir, "postprocess.json")

settings = {"native": {}, "fov50": {"resample_um": 50.0}}
if RUN["model"] == "spacing":
    settings["tell50"] = {"spacing_um": 50.0}
for tag, kw in settings.items():
    run_name = f"{RUN['name']}_{tag}"
    pred_dir = infer.run(model_name, model_path, run_name, roi_dir=ROI_DIR, tta=True, **common, **kw)
    # plain: TTA probabilities thresholded at 0.5; pp: tuned threshold + small-piece removal
    for label, d in ((run_name, pred_dir), (run_name + "_pp", postprocess.apply(pred_dir, pp_params, ROI_DIR))):
        res = score.score_run(d, roi_dir=ROI_DIR)
        with open(os.path.join(OUT, f"{label}.json"), "w") as f:
            json.dump(res, f, indent=1)
        score.print_table([res])
print("\n[✓] ALL DONE")
