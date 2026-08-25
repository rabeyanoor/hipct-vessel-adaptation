# Kaggle notebooks

The source code and `kernel-metadata.json` of every Kaggle notebook used in
this project, exactly as the last version ran on Kaggle (downloaded with
`kaggle kernels pull -m`), sorted by the stage of the project. All are private
notebooks of the account `rabeya100x`; the Kaggle name is
`hip-vesseladapt-<folder name>` unless noted.

Most notebooks of the second round are the same script,
[`../kaggle_jobs_kernel.py`](../kaggle_jobs_kernel.py), with a job list filled
in by [`../launch_jobs.py`](../launch_jobs.py); the job lists are also in
[`../jobs/`](../jobs). Their logs are kept locally in `logs/kaggle/<folder name>/`
(not in git).

Status is the status of the last version on Kaggle.

## 00_exploration — first attempts (before this repository's method)

| Notebook | What it did | Status |
|---|---|---|
| `hi-rabeya` (Kaggle name `hi-rabeya`) | vesselFM zero-shot on the K292 kidney (16.3–52 µm), collapse score and a physics-aware adapter on vesselFM, first SenNet kidney_1_dense dataset class | exploration |
| `notebookfad00c0539` (Kaggle name as is) | Kaggle port of the original Colab notebook "HiP-VesselAdapt": vesselFM on K292, adapter training for 5 epochs | exploration |

The physics-aware adapter of this stage survives as `models/physics_adapter.py` /
`models/hip_vessel_model.py`; it is not part of the paper.

## 01_first_round — 64³ patches, FiLM on the raw spacing (paper Section 5.6)

| Notebook | What it did | Status | Results |
|---|---|---|---|
| `hp-sweep` | reduced learning-rate × loss sweep | complete | `first_round/results/` |
| `full-hp-sweep` | full learning-rate × Dice/clDice sweep | complete | `first_round/results/` |
| `50ep-gpu-train` | 50-epoch plain and spacing-conditioned LoRA | error: code dataset not found (old mount path); replaced by `100ep-gpu-train` | — |
| `100ep-gpu-train` | 100-epoch plain and spacing-conditioned LoRA with in-distribution / out-of-distribution evaluation; its output (SAM-Med3D checkpoint, first-round models) is an input of later notebooks | complete | `first_round/results/`, `docs/results_log.txt` |
| `ablation-no-cldice` | spacing-conditioned LoRA without clDice | complete | `first_round/results/` |

## 02_benchmark_setup — test ROIs, Kaggle team scores, timing

| Notebook | What it did | Status | Results |
|---|---|---|---|
| `rois` | builds the 28 test ROIs (kidney_5, kidney_6, 13 µm, kidney_1_voi) from the Zenodo release (`benchmark/rois.py`, seed 0) | complete | input of every benchmark notebook |
| `score-teams` | scores the Kaggle top-5 predictions on the ROIs | complete | `results/scores/kaggle_team*.json` |
| `timing` | time of 2 × 50 training steps (spacing-conditioned with augmentation + refinement head; plain) and of inference on 192³ and 640³ volumes (native and fov50) | complete | log |
| `bench-cpu1` – `bench-cpu3` | first-round models and SAM-Med3D zero-shot on CPU | cancelled; the same runs are in `bench-g1` on GPU | — |
| `bench-cpu4` | p_base_s0 / s_base_s0 on CPU | complete; superseded by `bench-g1` | — |

## 03_second_round_training — the 2 × 2 design, 3 seeds (12 runs)

`p` = plain LoRA, `s` = spacing-conditioned LoRA (log-spacing FiLM), `base` =
no spacing augmentation, `aug` = spacing augmentation (`--fov-scale 0.25 2.0`).

| Notebook | GPU 0 | GPU 1 | Status |
|---|---|---|---|
| `train-a` | s_aug_s0, s_base_s0, s_aug_s1 | p_aug_s0 (out of memory, rerun in train-b), p_base_s0, p_aug_s1 | complete |
| `train-b` | p_aug_s0, s_base_s1 | p_base_s1, p_aug_s2 | complete |
| `train-c` | s_aug_s2, s_base_s2 | p_base_s2 | complete |

Training logs and configurations: `results/train_logs/`.

## 04_second_round_benchmark — predictions, scores, FiLM analysis

| Notebook | What it did | Status | Results |
|---|---|---|---|
| `bench-g1` | GPU predictions (native / tell50 / fov50): s_base_s0, s_aug_s0, p_base_s0, p_aug_s1, s_aug_s1, SAM-Med3D zero-shot, first-round models | running | scored by `score-g1` |
| `bench-g2` | GPU predictions: p_aug_s0, s_base_s1, p_base_s1, p_aug_s2, s_aug_s2, s_base_s2, p_base_s2 (17 runs × 28 ROIs) | complete | scored by `score-g2` |
| `score-g2` | CPU scoring of the 17 `bench-g2` runs | running | `results/scores/` |
| `film-a` | FiLM sweep of s_base_s0, s_aug_s0, s_aug_s1 | complete | `results/film/` |
| `film` | FiLM sweep of s_base_s1, s_base_s2, s_aug_s2 | complete | `results/film/` |

## 05_baselines

| Notebook | What it did | Status |
|---|---|---|
| `nnunet` | nnU-Net (100 epochs) trained on the challenge data, predictions on the ROIs; also had vesselFM on GPU 1 | error twice: Kaggle could not mount the competition data (`ERRORED_MOUNTING_DATASET`); to be rerun |
| `vesselfm` | vesselFM (with and without intensity inversion) on the ROIs | v1: `venv` without pip; v2: out of GPU memory; v3 (uv + Python 3.11, batch size 2): running |

## Updating this folder

```bash
kaggle kernels pull rabeya100x/hip-vesseladapt-<name> -p kaggle_kernels/notebooks/<stage>/<name> -m
kaggle kernels output rabeya100x/hip-vesseladapt-<name> -p logs/kaggle/<name> --file-pattern '.*\.log$'
```
