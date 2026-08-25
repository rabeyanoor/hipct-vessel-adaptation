# Spacing-aware LoRA for multi-resolution HiP-CT vessel segmentation

Code for the paper *"Does Knowing the Voxel Size Help? Spacing-Aware Low-Rank
Adaptation of a 3D Foundation Model for Multi-Resolution HiP-CT Vessel
Segmentation"* (manuscript in preparation; not part of this repository).

We adapt a frozen [SAM-Med3D](https://github.com/openmedlab/SAM-Med3D) encoder
with LoRA, with and without FiLM modulation of the LoRA branch by the voxel
spacing, and test it on held-out, expert-labelled human kidney vessels at four
voxel sizes (63.08, 50.28, 13 and 5.2 µm) from the SenNet + HOA challenge
(Jain et al., 2026). A 2 × 2 test separates the effect of the spacing input
from that of the field of view.

> **Status:** experiments of the second round are running; the result tables
> in the manuscript are filled from `results/` as runs finish. Every number in
> the paper is produced by the scripts below.

## Repository layout

```
train_lora_v2.py        training (plain / spacing-conditioned LoRA, spacing augmentation, refinement head)
analyze_film_sweep.py   FiLM gamma/beta of a trained model over a range of spacings
config.py               dataset paths and spacings
models/                 LoRA, spacing-conditioned LoRA (FiLM), full-resolution refinement head
losses/                 Dice + clDice + spacing-scaled gradient penalty + weighted BCE
utils/                  datasets (slab sampler), SAM-Med3D loader and patch, Zenodo download, metrics
benchmark/              test ROIs, inference (native / tell50 / fov50), scoring, baselines, paper tables
kaggle_kernels/         job-based Kaggle GPU/CPU pipeline used for all second-round runs
results/                scores (one JSON per run), FiLM sweeps, training logs
first_round/            first set of experiments (64^3 patches, raw-spacing FiLM); kept for Section 5.6
docs/                   dated research and results logs
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# external code, pinned to the commits used in the paper
git clone https://github.com/openmedlab/SAM-Med3D.git external/SAM-Med3D
git -C external/SAM-Med3D checkout bd7d1620df0f85fade74aa9e44262de953b559f1
git clone https://github.com/jocpae/clDice.git external/clDice
git -C external/clDice checkout 47d31a6cc4a8101b1ffe8052994821961e57af9f
python -m utils.patch_external_repos          # lets SAM-Med3D take inputs other than 128^3
# SAM-Med3D weights (sam_med3d.pth) -> external/SAM-Med3D/ckpt/sam_med3d.pth
```

## Data

* Training: `kidney_1_dense`, `kidney_2`, `kidney_3_dense/sparse` from the
  [SenNet + HOA Kaggle challenge](https://www.kaggle.com/competitions/blood-vessel-segmentation).
  The last 20 % of the slices of `kidney_1_dense` are held out for validation.
* Test: `kidney_5`, `kidney_6`, the 13 µm region and `kidney_1_voi`, plus the
  winning teams' predictions, from the Zenodo release (doi:10.5281/zenodo.19685381, CC-BY 4.0):

```bash
python -m utils.fetch_zenodo_subset kidney_5 kidney_6 kidney_1_voi team_predictions
python -m utils.fetch_zenodo_subset kidney_additional_voi_13um --z-range 0 2226
python -m benchmark.rois --raw-tiff-root <competition>/train data/zenodo        # test ROIs (seed 0)
python -m benchmark.rois --validation --raw-tiff-root <competition>/train      # validation ROIs
```

No test volume is used for training or for choosing settings.

## Reproducing the experiments

Main configuration (one run; seeds 0, 1, 2 are used in the paper):

```bash
python train_lora_v2.py --model spacing --spacing-encoding log --fov-scale 0.25 2.0 \
    --patch 128 --fg-prob 0.5 --refine --epochs 12 --iters 250 --val-every 2 --workers 3 \
    --raw-tiff-root <competition>/train --val-roi-dir data/benchmark/val_rois \
    --seed 0 --out checkpoints_v2/s_aug_s0
```

Drop `--fov-scale` for the runs without spacing augmentation and use
`--model plain` for plain LoRA. Evaluation in the three settings of the 2 × 2 test:

```bash
CKPT=checkpoints_v2/s_aug_s0/model_best.pth
ARGS="--model spacing_lora --checkpoint $CKPT --patch 128 --refine --spacing-encoding log"
python -m benchmark.infer $ARGS --run-name s_aug_s0_native                   # native grid, true spacing
python -m benchmark.infer $ARGS --run-name s_aug_s0_tell50 --spacing-um 50   # native grid, told 50 µm
python -m benchmark.infer $ARGS --run-name s_aug_s0_fov50 --resample-um 50   # resampled to 50 µm
python -m benchmark.score data/benchmark/preds/s_aug_s0_native --out results/scores/s_aug_s0_native.json
python -m benchmark.paper_tables --results results/scores --out results/paper_tables.md
```

Baselines: `benchmark/import_team_rle.py` (Kaggle top 5), `benchmark/run_nnunet.py`,
`benchmark/run_vesselfm.py`; SAM-Med3D without fine-tuning is `benchmark.infer` with an empty `--checkpoint`.

### On Kaggle

All second-round runs were done on Kaggle T4 GPUs with job lists:

```bash
kaggle_kernels/upload_code.sh "message"                       # code as a Kaggle dataset
python kaggle_kernels/make_jobs.py ...                        # training jobs -> kaggle_kernels/jobs/*.json
python kaggle_kernels/launch_jobs.py kaggle_kernels/jobs/train_a.json --slug hip-vesseladapt-train-a
python kaggle_kernels/make_bench_jobs.py ... --no-score       # inference on GPU
python kaggle_kernels/make_score_jobs.py ...                  # scoring on CPU
```

The exact job lists used are in `kaggle_kernels/jobs/`.

## First round

`first_round/` holds the first experiments (64³ patches, FiLM on the raw
spacing, random-patch evaluation), which motivated this study and are reported
in Section 5.6 of the paper. See [`first_round/README.md`](first_round/README.md).
The repository as it was before this layout is tagged `v0-flat-layout`.

## Citation

The paper is not published yet; please cite this repository and Jain et al. (2026) for the data.

## Acknowledgements

Data: the Human Organ Atlas and the SenNet + HOA challenge organisers (Jain et
al., 2026, *Nature Communications* 17:8273). Models: SAM-Med3D, clDice,
nnU-Net and vesselFM. Compute: Kaggle.
