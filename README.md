# HiP-VesselAdapt

**Paper title:** Spacing-Aware Parameter-Efficient Adaptation of Frozen 3D
Foundation Models for Multi-Resolution HiP-CT Vessel Segmentation
(`HiP-VesselAdapt` is the short name of the code.)

A small physics-aware adapter for **SAM-Med3D** (a pretrained 3D segmentation
foundation model), fine-tuned to segment blood vessels in high-resolution
synchrotron tomography (HiP-CT) scans of human organs (Heart, Lung, Kidney),
each scanned at a different voxel spacing.

**Status: work in progress.** All numbers in this README come from runs I
actually did. They are still quite low in absolute terms (see "Known
limitations"). The goal of this file is to keep track of what works, what has
been checked and what is still open.

---

## 1. What the project does

- Takes a **frozen, pretrained 3D foundation model** (SAM-Med3D, ViT image
  encoder + promptable mask decoder) and adapts it to a new imaging domain
  (ex-vivo synchrotron CT) with a small trainable **Physics-Aware Adapter**
  that is conditioned on the voxel spacing of each scan (in micrometers).
- Trains **one joint model** for three organs with three different
  resolutions (Heart 26.04μm, Lung 169.36μm, Kidney 50μm) instead of one model
  per organ.
- Evaluates with Dice, Hausdorff distance (HD95) and Betti numbers (topology:
  connected components and loops).

## 2. Data

Source: ESRF Human Organ Atlas (HiP-CT), raw `.jp2` slice stacks, downloaded
into `data/raw/<organ>_jp2/`.

**Important:** these downloads are only grayscale volumes. **They do not come
with vessel annotations.** All "ground truth" masks for these organs are
**weak pseudo-labels** from a multiscale Frangi (Hessian) vesselness filter
(`utils/pseudo_labels.py`), not manual labels. Keep this in mind for every
Dice/HD95/Betti number on these organs: they measure agreement with a
heuristic label, not with clinical ground truth.

### 2.1 Real ground truth: SenNet+HOA kidney dataset

To get around the pseudo-label problem I used the expert-annotated vessel
masks from the Kaggle **SenNet + HOA "Hacking the Human Vasculature in 3D"**
competition (`data/blood-vessel-segmentation.zip`, 21GB compressed / 43.5GB
uncompressed, 14,365 files). I checked the labels: the label TIFFs only
contain the values `{0, 255}`, and the foreground fraction in sample slices
from 3 different kidneys was 0.29%–1.54%, so they are real, non-empty masks.

Voxel spacing is **not** stored in the TIFF files (the `XResolution`/
`YResolution` tags are just placeholder values) and there is no metadata file
for it in the zip. All spacing values come from the dataset paper,
["Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys"](https://pmc.ncbi.nlm.nih.gov/articles/PMC13470036/),
Table 1, and are saved with the source in `data/kidney_metadata_verified.csv`
(made by `utils/kidney_metadata_verification.py`):

| Sample | Specimen | Spacing (μm) | Label slices | Split | Real GT? |
|---|---|---|---|---|---|
| kidney_1_dense | LADAF-2021-17-Right | 50.00 | 2279 | train | Yes |
| kidney_1_voi | LADAF-2021-17-Right | 5.20 | 1397 | train | Yes |
| kidney_2 | S-20-28-Kidney | 50.00 | 2217 | train | Yes |
| kidney_3_dense | LADAF-2020-27-Kidney-2 | 50.16 | 501 | train | Yes |
| kidney_3_sparse | LADAF-2020-27-Kidney-2 | 50.16 | 1035 | train | Yes |
| kidney_5 | LADAF-2021-17-Left | 50.28 | 0 | public_test | No (labels withheld) |
| kidney_6 | LADAF-2022-13-Bottom | 63.08 | 0 | private_test | No (labels withheld) |

**Useful detail:** `kidney_1_dense` and `kidney_1_voi` are the **same
specimen** at two different resolutions (50.00μm and 5.20μm, ~10x apart), and
both have dense expert labels. It is only one specimen, but it gives a
same-organ multi-resolution pair: train on the ~50μm kidneys and test on the
held-out 5.2μm `kidney_1_voi` as an unseen resolution, without any synthetic
downsampling.

**Data pipeline (2025-09-13):** `utils/convert_kidney_gt_to_memmap.py`
converts the 5 train samples with labels (kidney_1_dense, kidney_1_voi,
kidney_2, kidney_3_dense, kidney_3_sparse; ~45GB of memmaps in total)
straight from the zip. There is no separate extraction step, since Python's
`zipfile` allows random access (gzip streams do not). `utils/real_gt_dataset.py`
(`RealGTKidneyDataset`) loads them as a **separate dataset mode**; the
Frangi-based `MultiOrganHiPCTDataset` pipeline for Heart/Lung/Kidney is
unchanged.

Checked with `diagnose_real_gt_dataset.py` (data only, no training): shapes
match, spacing is correct for each sample, and the masks are binary. I also
looked at one foreground patch by eye during development (not part of the
code; the diagnostic scripts only print numbers). The label follows a vessel
cross-section and tissue boundary visible in the raw image quite precisely,
much cleaner than the Frangi pseudo-labels.

**Baselines on real GT (2025-09-13)**: see section 5.2 for the numbers.
`evaluate_real_gt.py` (Dice/IoU/clDice/HD95/ASSD, separate from `evaluate.py`
which is for the pseudo-labeled organs) and `train_real_gt.py` (resumable,
saves a checkpoint every epoch to `checkpoints_real_gt/`) both use
`RealGTKidneyDataset`. By default `train_real_gt.py` leaves `kidney_1_voi`
(5.2μm) out of training so it stays held out for the resolution test.

**Bug fixed in `evaluate_real_gt.py`:** `soft_cldice.forward(y_true, y_pred)`
(from `external/clDice/cldice_loss/pytorch/cldice.py`) returns `1 - clDice`,
i.e. a **loss** (same as `soft_dice()` in that file), not the clDice score.
The first run showed clDice≈0.97 together with Dice≈0.03, which does not make
sense, because it was printing the loss. Changed to
`1.0 - cldice_fn(targets, binary_pred).item()`; clDice is now in the same
range as Dice, as expected. (`losses/combined_loss.py` uses the same function
as a *training loss*, which was already correct.)

**Sparse vessels:** kidney vessels are sparse, so most random 64³ patches
contain none (71-80 of 100 random patches had no annotated foreground).
Dice/IoU/clDice/HD95/ASSD are undefined or misleadingly high on an empty
target, so `evaluate_real_gt.py` skips empty-target patches when averaging and
`train_real_gt.py` skips them for gradient steps. Both print how many were
skipped.

**Not done yet:** Plain LoRA is implemented (`models/lora.py`,
`models/plain_lora_model.py`) and runs, but I have not trained or evaluated it
on real GT yet (no time so far, see section 7). The same-specimen resolution
test, Spacing-Conditioned LoRA and the ablation study are also still to do.

## 3. Pipeline

```
data/raw/<organ>_jp2/*.jp2          (raw ESRF download)
        |
utils/convert_to_memmap.py          -> data/memmap/<organ>.dat (raw np.memmap)
        |                               (NOT .nii.gz -- see "Why memmap" below)
utils/dataset.py (MultiOrganHiPCTDataset)
        |  - grid-aligned random patch sampling (64^3 default)
        |  - Frangi pseudo-label generation, cached per grid block
        |  - z-score normalization
        v
models/hip_vessel_model.py (HiPVesselAdaptModel)
        |  - SAM-Med3D image_encoder (frozen, pretrained, torch.no_grad())
        |  - models/physics_adapter.py: PhysicsAwareAdapter (trainable)
        |  - SAM-Med3D prompt_encoder (auto-prompted, see below)
        |  - SAM-Med3D mask_decoder (trainable)
        |  - models/collapse_head.py: CollapseDetectionHead (trainable, not trained/validated yet)
        v
losses/combined_loss.py (HiPVesselLoss)
        |  - Soft Dice + clDice (topology) + anisotropic gradient penalty
        |  - + class-weighted BCE-on-logits (see "Bugs fixed" below)
        v
train.py  (resumable, checkpoints every epoch)  ->  checkpoints/best_model.pth
evaluate.py            -> Dice / HD95 per organ
evaluate_topology.py   -> Betti numbers (b0, b1) per organ
baseline_zeroshot.py   -> same metrics with NO fine-tuning, for comparison
```

### Why `np.memmap` instead of `.nii.gz`?

`.nii.gz` is gzip-compressed, and neither SimpleITK nor ITK can seek randomly
inside a compressed stream, so every patch read decompresses the file from
the start. It took **15–56 seconds to read a single 64³ patch**, which makes
training impossible. `utils/convert_to_memmap.py` converts each organ's slice
stack once into an uncompressed flat binary file (one 2D slice at a time, so
RAM stays low). After that, reading a patch is practically instant.

### Why an "auto-prompt" for SAM-Med3D's decoder?

SAM-Med3D's `mask_decoder` is a **promptable** head (points/box), not an
automatic dense-segmentation head. Without any prompt
(`points=None, boxes=None`) it has no spatial cue, and in practice it ended up
predicting **every voxel as foreground** after training. So I give it two
corner points covering the whole volume, which act like a whole-volume box.
(SAM-Med3D's box-embedding code has a hard-coded 2D reshape that crashes in
this 3D port, so I use points instead; see the comments in
`models/hip_vessel_model.py`.)

## 4. Bugs found and fixed (in order)

1. **No real ground truth.** The original pipeline compared predictions with
   a threshold of the *input image itself*
   (`targets = (images > mean+0.5*std)`). Replaced with Frangi pseudo-labels.
2. **Wrong vessel polarity.** Frangi's `black_ridges` was `False` by default
   (vessels assumed brighter than tissue). Looking at the scans showed that
   vessels are *darker*: a clear airway/vessel cross-section in the lung was
   only picked up with `black_ridges=True`. Fixed for all 3 organs.
3. **I/O bottleneck.** See "Why np.memmap" above.
4. **Backbone was not actually pretrained.** `checkpoint_path` was `None`
   everywhere, so the frozen backbone was frozen *random* weights and the
   adapter was learning on top of noise. I downloaded the SAM-Med3D checkpoint
   (`external/SAM-Med3D/ckpt/sam_med3d.pth`, 1.2GB) and fixed the loading
   code. The released checkpoint is a full training-state dict (weights are
   under the `model_state_dict` key) and needs the `vit_b_ori` variant
   (768-dim, 128³ pretraining), not the registry default `vit_b` (384-dim,
   256³). With that it loads with 0 missing/unexpected keys.
5. **Decoder predicting everything as foreground.** See "auto-prompt" above.
   Cause: sigmoid saturation + no prompt + weak gradient from the sparse
   target. Fixed with the auto-prompt *and* a class-weighted
   `BCEWithLogitsLoss` term (on logits, so it stays numerically stable even
   when saturated; Dice/clDice on probabilities cannot recover once the
   sigmoid saturates).
6. **Long runs getting killed.** This is a shared desktop, not a GPU machine,
   and Chrome takes a lot of its ~15GB RAM, so background training jobs get
   OOM-killed at random. `train.py` now saves a checkpoint **every epoch**
   and supports `resume=True`, so a crash loses at most one epoch.
7. **Three bugs when moving the 100-epoch run to a Kaggle GPU kernel.** (The
   local CPU cannot do 100 epochs × 2 models in a reasonable time, so the
   controlled retrain was moved to Kaggle's free GPU.) Each one was found
   from the kernel's stderr log:
   - **Input paths.** Kaggle mounts datasets/competitions under
     `/kaggle/input/datasets/<owner>/<name>/` and
     `/kaggle/input/competitions/<slug>/`, not directly under
     `/kaggle/input/`, so the original hard-coded path found nothing. Fixed
     with a bounded recursive search (`find_by_marker()`) for a folder that
     contains a known file/subfolder.
   - **Disk quota.** Making `np.memmap` copies of kidney_1_dense +
     kidney_1_voi needs ~23GB, much more than the `/kaggle/working` quota, and
     the conversion filled it half way (`Bus error`, `No space left on
     device`). Fixed by adding a `raw_tiff_root` mode to
     `RealGTKidneyDataset` that reads TIFF slices directly from the read-only
     competition mount in `__getitem__`, with no memmap conversion. The mount
     does not count toward the quota.
   - **clDice import.** `external/clDice/cldice_loss/pytorch/cldice.py`
     (upstream, cloned fresh by the kernel each run) does
     `from .soft_skeleton import SoftSkeletonize`, a relative import that
     only works when the file is imported as part of a package. I load it as
     a standalone file, so this always fails. It only worked locally because
     I had earlier edited my local copy (not committed) to add a
     `try/except ImportError` fallback, which the fresh clone does not have.
     Fixed in `utils/load_cldice.py`: it loads `soft_skeleton.py` first and
     registers it in `sys.modules`, then reads `cldice.py`, rewrites the
     relative import into an absolute one and executes it. This works whether
     or not upstream ever adds the fallback.
8. **SAM-Med3D "not importable" on Kaggle, with no real error.**
   `external/SAM-Med3D/segment_anything/__init__.py` does
   `from utils import *`, an absolute import meant to reach SAM-Med3D's own
   `external/SAM-Med3D/utils/`. That only works if SAM-Med3D's repo root is
   the first entry on `sys.path` with a top-level `utils`. This project also
   has a top-level `utils/` package, and the model files used
   `sys.path.append(sam_med3d_path)` (append, so my `utils/`, which comes
   earlier on `sys.path`, wins) inside a bare
   `except ImportError: HAS_SAM_MED3D = False`, which hid the real error.
   Fixed with `utils/load_sam_med3d.py`: `sam_med3d_path` is put at the front
   of `sys.path` only while importing `segment_anything`, and any
   `utils`/`utils.*` entries added to `sys.modules` during that import are
   removed afterwards, so `from utils.xxx import yyy` in the rest of the
   project still finds this project's `utils/`. I also removed the bare
   `except ImportError`, so a future import failure shows the real traceback.
9. **CUDA "no kernel image is available for execution on the device".**
   After fixing the two bugs above the kernel got to training and failed on
   the GPU: Kaggle had given it a Tesla P100 (compute capability `sm_60`),
   which the PyTorch build in the kernel image does not support (`sm_70`
   minimum). `kernel-metadata.json` did not ask for a specific accelerator,
   so Kaggle picked whatever was free. Fixed by setting
   `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` (`sm_75`,
   supported).
10. **`RuntimeError: The size of tensor a (4) must match the size of tensor
    b (8)` in SAM-Med3D's image encoder.** With a working GPU, training
    reached the first forward pass and crashed when adding the positional
    embedding sized for 128³ pretraining (8 tokens/axis at patch 16) to a
    64³ input (4 tokens/axis). The fix (interpolating `pos_embed`, and in
    `mask_decoder3D.py` the dense/image prompt embeddings, to the input size
    when it differs from the pretraining size) had only been made as an
    uncommitted local edit to `external/SAM-Med3D`, so it was lost every time
    the repo was cloned fresh, as on Kaggle. Same problem as the clDice
    import in bug 7. Now fixed properly in `utils/patch_external_repos.py`,
    an idempotent text patch that checks itself (it fails loudly if the
    upstream source no longer looks as expected instead of silently doing
    nothing) and is applied after every fresh clone, locally or on Kaggle.

## 5. Results

The numbers in this first part are the printed output of `evaluate.py` /
`baseline_zeroshot.py` / `evaluate_topology.py` on HiP-CT data with Frangi
pseudo-labels, on CPU (no GPU on this machine).

### Zero-shot vs. fine-tuned (20 epochs, 24 patches/epoch)

| Organ | Zero-Shot DSC | Fine-tuned DSC | Zero-Shot HD95 (voxels) | Fine-tuned HD95 |
|---|---|---|---|---|
| Heart | 0.021 | 0.060 | 40.3 | 27.4 |
| Lung | 0.037 | 0.077 | 17.5 | 8.5 |
| Kidney | 0.051 | 0.082 | 19.4 | 24.8 |

Fine-tuning improves Dice over the zero-shot backbone for **all three
organs** (1.6x–2.9x), and this held at every checkpoint I measured during
training (10, 18 and 20 epochs). This is the main evidence that the physics
adapter is actually learning something.

### Betti numbers (topology), 20-epoch model, 5 patches/organ

| Organ | Pred b0 | Target b0 | \|Δb0\| | Pred b1 | Target b1 | \|Δb1\| |
|---|---|---|---|---|---|---|
| Heart | 32.6 | 34.4 | 1.8 | 0.2 | 2.6 | 2.4 |
| Lung | 21.8 | 49.8 | 28.0 | 4.6 | 3.2 | 1.4 |
| Kidney | 41.0 | 89.6 | 48.6 | 1.8 | 12.8 | 11.0 |

### 5.2 Real-GT kidney baselines (SenNet+HOA, no pseudo-labels)

Output of `evaluate_real_gt.py` on the expert vessel masks (section 2.1):
`kidney_1_dense`, `kidney_2`, `kidney_3_dense`, `kidney_3_sparse` (100 random
64³ patches per run; 71-79 skipped as empty-target, see above), on CPU.
`kidney_1_voi` is not used in any training here; it is kept for the
resolution test.

**Terminology:** none of these is "full" fine-tuning of the backbone. The
SAM-Med3D image encoder is frozen in every row, zero-shot included. The only
differences are (a) whether there is a trainable adapter after the frozen
encoder (PhysicsAwareAdapter) or trainable low-rank updates inside the
encoder's attention layers (LoRA), and (b) the mask_decoder, which is always
trained. An earlier version of this README and of `train_real_gt.py`'s
docstring called this "full fine-tuning", which was wrong.

| Model | Trainable | Trainable component | Dice | IoU | clDice | HD95 (voxels) | ASSD (voxels) |
|---|---|---|---|---|---|---|---|
| Zero-shot SAM-Med3D | 0% | none (no training) | 0.088 | 0.052 | 0.076 | 48.1 | 25.2 |
| PhysicsAwareAdapter (10 epochs) | 7.65% | adapter + decoder | 0.157 | 0.094 | 0.139 | 36.2 | inf* |
| Plain LoRA, r=4 (10 epochs) | 7.75% | LoRA (24 attn layers) + decoder | 0.067 | 0.038 | 0.057 | 45.7 | 24.8 |
| **Spacing-Conditioned LoRA, r=4 (10 epochs)** | **8.88%** | spacing-modulated LoRA + decoder | **0.206** | **0.131** | **0.261** | **23.4** | inf* |

**In-distribution, Spacing-Conditioned LoRA is best on every metric**,
including against Plain LoRA, its closest ablation (same architecture and
training; the *only* difference is whether the LoRA branch is modulated by
the voxel spacing). The extra trainable cost is small (8.88% vs. 7.65-7.75%).

**⚠️ This does NOT hold out-of-distribution. Section 5.3 reverses this
ranking**, so the in-distribution result alone is not the conclusion.

**Plain LoRA result:** after 10 epochs Plain LoRA is worse than even the
untrained zero-shot baseline, and well below both adapter-based methods.
Possible reasons (none confirmed yet):
- LoRA (rank 4) changes attention *inside* the frozen encoder, while the
  adapter changes the encoder's *output* features right before the decoder.
  That is a shorter and maybe easier optimization path for the same number
  of steps (~170-220 in each case).
- All three trained runs used the same loss weights and learning rate
  (originally tuned for PhysicsAwareAdapter) with no per-architecture tuning,
  which is a confound. Spacing-Conditioned LoRA still won despite this, but I
  have not done a proper sweep.
- 10 epochs with ~20 non-empty gradient steps per epoch may just be too few
  for Plain LoRA to change the frozen encoder enough, since it sits several
  layers away from the loss. On the other hand, Spacing-Conditioned LoRA,
  with the same LoRA plus spacing modulation, reached a much lower loss (1.56
  vs. Plain LoRA's 2.05) with the same setup, which suggests the spacing
  conditioning itself is what made the difference, not just more LoRA
  training.

This needs more work (more epochs, a learning-rate sweep per architecture,
several seeds) before the ranking above can be taken as final.

\* **ASSD=inf for PhysicsAwareAdapter and Spacing-Conditioned LoRA:** MONAI
warned "the prediction of class 0 is all 0", i.e. at least one patch had a
completely empty *prediction* (surface distance is undefined if one side has
no surface). Plain LoRA's ASSD was finite (24.8) with the same evaluation.
This needs checking (section 7) before trusting the ASSD of the two better
models.

Total parameters: 100,622,065 (PhysicsAwareAdapter) / 100,730,416 (Plain
LoRA) / 101,985,328 (Spacing-Conditioned LoRA; the per-layer spacing FiLM
networks add ~1.25M params over Plain LoRA). In a first version of
Spacing-Conditioned LoRA the FiLM hidden size grew with the attention layer
width, which for the qkv layers (2304-dim) alone added ~31M trainable params
(31.79% of the model). I noticed this from the parameter count before
training and changed it to a fixed small hidden size (16), which gives the
8.88% above (see the comments in `models/spacing_lora.py`).

### 5.3 Resolution-OOD test (the key experiment) — reverses the ranking of 5.2

All three trained models from section 5.2 evaluated on `kidney_1_voi`, the
5.2μm specimen, which is **never used in training** (`exclude_for_ood`
default in `train_real_gt.py`/`train_plain_lora.py`/`train_spacing_lora.py`,
see section 2.1). All training data was ~50μm, so this is a ~10x resolution
shift on real data. 40 patches sampled, 34-35 skipped as empty-target (this
specimen's vessels are sparser than in the 50μm training kidneys: mean FG
fraction 0.14-3.56% vs. 0.25-0.73% in training).

| Model | In-dist. Dice (50μm) | **OOD Dice (5.2μm)** | OOD IoU | OOD clDice | OOD HD95 | OOD ASSD |
|---|---|---|---|---|---|---|
| PhysicsAwareAdapter | 0.157 | 0.064 | 0.035 | 0.059 | 53.1 | 25.3 |
| Plain LoRA | 0.067 | **0.067** | **0.036** | 0.210 | 50.7 | 25.9 |
| Spacing-Conditioned LoRA | **0.206** | 0.002 | 0.001 | 0.008 | 41.6 | **18.7** |

**This reverses the ranking of section 5.2.** Spacing-Conditioned LoRA, the
clear in-distribution winner, **drops to almost zero Dice (0.002)** at the
unseen resolution, the worst of the three on Dice/IoU/clDice. It still has
the best HD95/ASSD, but that is not a contradiction: a model that predicts
almost nothing can still get a good boundary distance if the few voxels it
predicts are roughly in the right place. These metrics are known to be
misleading when Dice is this low.

**First hypothesis, tested and rejected:** my first guess was that the
spacing-FiLM network (trained on basically a single spacing, ~50μm, for all 4
training kidneys) would extrapolate badly ~10x out of range and give
large/erratic gamma/beta at 5.2μm, corrupting the LoRA branch. I checked by
computing `spacing_film(spacing)` for both values on the trained model: at
50μm, gamma∈[0.70, 1.54] (mean 1.05); at 5.2μm, gamma∈[0.97, 1.06] (mean
1.01). So the OOD modulation is actually *smaller* (closer to identity) than
the in-distribution one, the opposite of the hypothesis. **The FiLM network
is not confirmed as the cause, and the cause is still open.**

A more likely explanation, partly checked: a 64³ patch at 5.2μm covers a
region ~10x *smaller* than the same patch at 50μm (≈333μm vs. ≈3200μm
across). The model sees a much more zoomed-in view than anything in training,
which is a visual domain shift regardless of the spacing *value* given to
FiLM. Supporting this, the target foreground fraction is ~20x higher on
average at 5.2μm (mean 0.67%, up to 6.7% per patch, n=10 patches) than at
50μm (mean 0.032%, up to 0.67%, n=10 patches), which fits the same vessel
filling much more of a much smaller window. z-score normalization is not the
problem (both resolutions normalize to mean≈0, std≈0.90-0.98).

It is also not an obvious numerical problem in the frozen backbone: the
image_encoder output statistics are almost the same for both resolutions
(50μm: mean=0.0013, std=0.218; 5.2μm: mean=0.0014, std=0.221; similar
min/max), so there is no activation blow-up or degenerate output. The issue
is more likely about *content* than magnitude: the frozen encoder (and the
LoRA weights inside it, trained only on 50μm patches) may recognize patterns
at the wrong physical scale for a 10x zoomed-in view, which mean/std cannot
show. Checking this would need a per-channel/spatial analysis, not done yet.
It might hit Spacing-Conditioned LoRA hardest if its LoRA weights inside the
encoder are more scale-specific than PhysicsAwareAdapter's later adaptation
point or Plain LoRA's (unmodulated, but also 50μm-only) weights. This is
still an untested hypothesis.

**This matters more than the positive result in 5.2**, and it changes what
the project can claim: the method is NOT shown to generalize across
resolutions. If anything, it currently generalizes *worse* than plain LoRA on
real OOD data. Next steps: find the actual cause (compare field of view /
image statistics at 50μm vs 5.2μm directly), train on more real spacings if
possible, and repeat this OOD test afterwards instead of relying on one
checkpoint.

### 5.4 Controlled 100-epoch GPU retrain — confirms 5.3, not under-training

The numbers in 5.2/5.3 are from a 10-epoch CPU run, so the open question was
whether the OOD collapse was real or just from too little training. This
retrain (100 epochs, both architectures, GPU, same data split and protocol;
see section 3.4 of test.sh for all the problems getting the kernel to run)
answers that. Output of `evaluate_real_gt.py` from Kaggle kernel version 7
(`rabeya100x/hip-vesseladapt-100ep-gpu-train`), taken from
`kaggle kernels_logs` after the kernel finished (`COMPLETE`):

| Model | Trainable | In-dist Dice | In-dist IoU | In-dist clDice | OOD Dice | OOD IoU | OOD clDice |
|---|---|---|---|---|---|---|---|
| Plain LoRA (100 ep, GPU) | 7.75% | 0.1838 (n=26/100) | 0.1201 | 0.2466 | 0.1672 (n=5/40) | 0.1008 | 0.1575 |
| Spacing-Conditioned LoRA (100 ep, GPU) | 8.88% | **0.3145** (n=18/100) | **0.2279** | **0.3268** | **0.0058** (n=5/40) | 0.0029 | 0.0162 |

**Same pattern as 5.2/5.3, and more training makes it stronger:**
- In-distribution, Spacing-Conditioned LoRA's lead over Plain LoRA *grew*
  from 10 to 100 epochs (0.206 vs 0.067 at 10 epochs → 0.3145 vs 0.1838 at
  100 epochs).
- Out-of-distribution, Spacing-Conditioned LoRA is still worse by more than
  an order of magnitude (0.0058 vs Plain LoRA's 0.1672), and far below its
  own in-distribution score (0.0058 vs 0.3145, a ~98% relative drop). Plain
  LoRA's OOD score (0.1672) is close to its in-distribution score (0.1838),
  so it degrades gracefully, while Spacing-Conditioned LoRA collapses.
- So "10 epochs was too few" does not explain 5.3. The collapse is a property
  of the trained model. This is what the OOD investigation
  (evaluate_fov_confound.py, analyze_film_layers.py, run_ablation.py; all
  written, see section 7) has to explain, and it should be run on these
  100-epoch checkpoints, not the old 10-epoch ones.

**Caveat:** only 5 of the 40 OOD patches contained any annotated vessel, for
*both* models, so each OOD Dice/IoU/clDice above is an average over just 5
patches. The gap (0.1672 vs 0.0058) is large enough to mean something even
at this n, but a larger OOD sample is still on the to-do list (section 7).

**Two other things in this run's output:**
- **Final training loss is almost the same for both models** (Plain LoRA
  1.4737, Spacing-Conditioned LoRA 1.4900, both at epoch 100/100). In the
  10-epoch run Spacing-Conditioned LoRA's loss was clearly lower (1.56 vs
  Plain LoRA's 2.05). At 100 epochs the training loss no longer reflects the
  big Dice gap on real GT, so only the evaluation metrics should be used to
  compare them.
- **HD95 and ASSD were `nan` (n=0) in all four evaluations** (both models,
  in-dist and OOD). The earlier CPU run gave real HD95 values and only ASSD
  was `inf` for two of the three models. `_safe_metric()` in
  `evaluate_real_gt.py` was silently catching whatever exception MONAI's
  `HausdorffDistanceMetric`/`SurfaceDistanceMetric` threw on every non-empty
  patch. Dice/IoU/clDice worked on the same GPU tensors in the same loop, so
  it is not a problem with moving the tensors. I changed `_safe_metric()` to
  print the actual exception the first time each metric fails instead of
  just returning `None`, so the next run will show the cause. Not diagnosed
  yet at the time of writing.

### 5.5 Field-of-view confound test — strong support for the FOV hypothesis

I downloaded the two 100-epoch checkpoints from the Kaggle kernel output
(only the two `best_model_*.pth` files, through signed URLs, not the whole
working directory) and ran `evaluate_fov_confound.py` on both locally with
`samples=40`. The method: instead of a native 64³ crop of kidney_1_voi
(5.2μm), take a much bigger cube whose *physical* size matches a 64³ patch at
the training spacing (~50μm) and downsample it to 64³. Same 5.2μm data, but
the field of view matches training.

| Model | In-dist (5.2) | OOD, native-scale crop (5.3) | OOD, FOV-matched (this test) |
|---|---|---|---|
| Plain LoRA | 0.1838 (n=26/100) | 0.1672 (n=5/40) | **0.2895** (n=40/40) |
| Spacing-Conditioned LoRA | 0.3145 (n=18/100) | **0.0058** (n=5/40) | **0.2215** (n=40/40) |

**This clearly supports the field-of-view hypothesis.** Main points:
- All 40 FOV-matched patches contained vessel (0 empty), compared with 35/40
  *empty* patches in the native-scale OOD crop. At 5.2μm, a 64³ window covers
  too little tissue to reliably contain a vessel, which is a big part of why
  that evaluation had only 5 usable patches.
- Spacing-Conditioned LoRA's OOD Dice goes from 0.0058 (a ~98% drop from its
  in-distribution score) to 0.2215, about 70% of its in-distribution
  performance, just by matching the field of view on the *same* OOD data.
  Plain LoRA also improves a lot (0.1672 → 0.2895), so field of view is not
  the *whole* story (Plain LoRA never collapsed in the first place), but it
  is a major factor for both models and the main one for
  Spacing-Conditioned LoRA.
- With matched FOV, Plain LoRA is slightly better than Spacing-Conditioned
  LoRA, so the in-distribution ranking (section 5.4) flips again once the
  confound is removed. **This is significant**: a separate scoring run
  (`significance_test_fov.py`, 40 new patches per model; the means are a bit
  different from the table because `evaluate_fov_confound.py` draws a new
  random cube each call, but direction and size agree: Plain LoRA 0.3681 vs
  Spacing-Conditioned LoRA 0.2618) gives a Mann-Whitney U p=0.0216,
  significant at α=0.05.

**Interpretation (a hypothesis, not proven):** most (not all) of the "OOD
collapse" in 5.3/5.4 looks like an ordinary shift to an unseen visual scale,
not something specific to how the FiLM mechanism uses the spacing value. The
per-layer FiLM analysis below gives a second reason *why*
Spacing-Conditioned LoRA suffers more than Plain LoRA from the FOV shift that
is left after this correction.

### 5.6 Per-layer FiLM breakdown — a second lead besides FOV

`analyze_film_layers.py` on the 100-epoch Spacing-Conditioned LoRA
checkpoint, comparing gamma/beta of each of the 24 injected layers at the
in-distribution spacing (50μm) and the OOD spacing (5.2μm). No image encoder
pass is needed, because `spacing_film` is a small MLP that only takes the
3-value spacing.

The earlier aggregate check (section 5.3, 10-epoch checkpoint) rejected the
idea that "FiLM gives wild values at an unseen spacing"; OOD gamma was, if
anything, closer to identity. At 100 epochs the per-layer results go in the
same direction but **much more strongly**: gamma at the *training* spacing
has become large during training (e.g. layer 19: gamma_mean=6.52,
gamma_max=21.6 at 50μm; layer 2: gamma_max=19.7 at 50μm), while gamma in the
same layers at the *OOD* spacing stays near identity (layer 19:
gamma_mean=1.59, gamma_max=3.23 at 5.2μm; layer 2: gamma_max=3.06 at 5.2μm).
The pattern holds across all 24 layers, not just one.

**Updated hypothesis:** FiLM does not produce *dangerous* values OOD.
Instead, training pushed it toward large modulation *tuned to* the training
spacing, and after 100 epochs the model may depend on that modulation to
segment 50μm data well. At an unseen spacing the FiLM output goes back toward
identity (gamma≈1, beta≈0), which is roughly Plain LoRA's behaviour *for that
layer*, but not exactly, because the LoRA A/B weights were trained together
with the strong modulation. Together with 5.5, the best picture right now is:
most of the OOD failure is an ordinary field-of-view shift, and the FiLM
mechanism makes it worse by falling back toward (but not quite reaching)
plain-LoRA-like behaviour that the model was never trained for. Still a
hypothesis. The obvious next test is training both models on a wider range of
real spacings, so that "unseen spacing" is no longer mixed up with "unseen
field of view".

### 5.7 Ablation: clDice loss term, 100 epochs GPU

Stage 4 of `run_ablation.py` (Spacing-Conditioned LoRA with the clDice loss
switched off, using the new `use_cldice=False` flag in `HiPVesselLoss`),
trained on Kaggle GPU for 100 epochs, otherwise the same as section 5.4.

| Model | In-dist Dice | In-dist clDice | OOD Dice | OOD clDice |
|---|---|---|---|---|
| Spacing-Conditioned LoRA, WITH clDice (5.4) | 0.3145 (n=18/100) | 0.3268 | 0.0058 (n=5/40) | 0.0162 |
| Spacing-Conditioned LoRA, NO clDice | **0.1898** (n=19/100) | 0.1927 | 0.0015 (n=3/40) | 0.0035 |

**The clDice loss clearly helps in-distribution.** Without it, in-distribution
Dice drops by about 40% relative (0.3145 → 0.1898). This is not only the
clDice metric (which would be expected, since the loss optimizes something
close to it) but also Dice, which the loss does not optimize directly.
**It does not explain the OOD collapse**: OOD Dice is very low either way
(0.0058 vs 0.0015), both close to noise at n=3–5. So the clDice ablation and
the FOV/FiLM findings (5.5, 5.6) are separate things: clDice improves
segmentation quality in general, while the OOD collapse seems to come mostly
from the field-of-view mismatch and the spacing-specific FiLM modulation.

### 5.8 Cause found: HD95/ASSD `nan` on Kaggle GPU (sections 5.4, 5.7)

With the better `_safe_metric()` logging (section 5.4), the ablation run
finally showed the actual exception. It is an environment problem, not a bug
in this code:

```
HausdorffDistanceMetric raised: .../cupy/_core/include/cupy/_cccl/thrust/
thrust/detail/config/cpp_dialect.h(74): catastrophic error: #error
directive: Thrust requires at least C++17.
```

For GPU tensors, MONAI's `HausdorffDistanceMetric`/`SurfaceDistanceMetric`
use a CUDA kernel compiled on the fly by CuPy for the distance transform. On
Kaggle's image this compilation fails (a CuPy/Thrust/CUDA toolkit version
mismatch in their Docker image, nothing I can change). Dice/IoU/clDice do not
use that code, which is why only HD95/ASSD were `nan`. Fixed in
`evaluate_real_gt.py` by moving `binary_pred`/`targets` to CPU just for the
HD95/ASSD calls (`.cpu()` right before `_safe_metric()`), so the broken GPU
path is never used. Checked locally (HD95/ASSD still come out correctly).
Not re-checked on Kaggle, since re-running a 100-epoch kernel only for this
would use a lot of GPU quota for a fix I am fairly confident in.

## 6. Known limitations

- **Absolute DSC is low (0.06–0.08).** Two likely reasons: (a) very little
  training so far (20 epochs × 24 patches/epoch = 480 gradient steps in
  total, on CPU, where each extra epoch takes hours); (b) the Frangi
  pseudo-labels are a noisy heuristic target. They are often sparse,
  fragmented blobs rather than a clean connected vessel tree. That is not a
  bug, but it limits label quality unless the labels are cleaned up or real
  annotations are used.
- **DSC jumps around between epochs** instead of improving steadily (e.g.
  Heart: 0.077 at 10ep → 0.043 at 18ep → 0.060 at 20ep). With only 5
  evaluation patches per organ each DSC estimate has high variance, so the
  zero-shot vs. fine-tuned *comparison* is more reliable than any single
  number.
- **Betti-number gaps for Lung/Kidney are large**, but that is probably
  partly due to fragmented pseudo-labels (the target itself has many spurious
  components) rather than the model being wrong. A cleaner reference is
  needed to tell these apart.
- **`CollapseDetectionHead` exists but is not trained or validated.** It runs
  (extra output next to the mask), but there is no supervision signal or
  evaluation for it yet.
- **`clDice` loss** is in `losses/combined_loss.py`, but its effect has not
  been ablated separately from Dice/BCE.

## 7. Not done yet

- **Find the actual cause of the OOD collapse** (sections 5.3, 5.4). This is
  the most important open question. It is confirmed NOT to be caused by too
  little training (5.4: it still collapses, even more clearly, after 10x more
  training on GPU). "FiLM gives wild values OOD" was tested and rejected
  (5.3). Two leads now (5.5, 5.6): most of the collapse looks like an
  ordinary field-of-view shift (Dice goes from 0.0058 to 0.2215 once the
  physical FOV is matched), and FiLM seems to have learned strong modulation
  tied to the training spacing that goes back to near-identity OOD, making
  the FOV effect worse. Neither is confirmed as the full explanation;
  training on a wider range of real spacings is the next check. Until this
  is understood and fixed, Spacing-Conditioned LoRA is not a validated
  method, even though its in-distribution result is strong.
- **Hyperparameter tuning per architecture.** All trained models (5.2, 5.4)
  used the same loss weights/LR, originally tuned for PhysicsAwareAdapter.
  Each architecture needs its own sweep. Script written (`colab_run.py`,
  `run_hyperparameter_sweep()`), **not run yet**. Most of Kaggle's free GPU
  quota (6 hours/week) went into the 100-epoch retrain and the OOD
  experiments (5.4–5.6), so this one is meant to run on Google Colab. It
  reuses the same `train_plain_lora`/`train_spacing_lora`/`evaluate_real_gt`
  functions and pulls the code and the competition data through the Kaggle
  API, so no GitHub repo or Drive upload is needed. See the docstring in the
  file for how to run it.
- **Multiple seeds.** Every Dice/IoU/clDice number in this README so far is
  from a single run. Script written (`colab_run.py`, `run_multiseed()`),
  **not run yet**, also for Colab. It trains both architectures with N seeds
  (default 3), 100 epochs each, evaluates in-distribution and OOD every time
  and reports mean ± std.
- **ASSD=inf / ASSD=nan.** The 100-epoch GPU case is **DONE** (5.4, 5.7,
  5.8): the cause is that Kaggle's CuPy/Thrust toolchain cannot compile the
  GPU distance-transform kernel used by MONAI's HD95/ASSD, and it is fixed by
  moving the tensors to CPU for those two metrics (checked locally). The
  ORIGINAL 10-epoch CPU case (5.2, `ASSD=inf`, not `nan`) is a different
  issue and only **partly checked**: the empty-prediction idea was tested on
  15 patches with 0 confirmed, so inconclusive.
- Other baselines: **nnU-Net, VesselFM, VesselSDF** (only zero-shot
  SAM-Med3D has been run as a baseline so far, on both pseudo-label and
  real-GT data).
- **Vascular graph validation** (skeletonize + networkx connectivity
  comparison).
- **Ablation study** (zero-shot → +LoRA → +spacing conditioning → +clDice).
  **DONE** for stage 4 (5.7): Spacing-Conditioned LoRA without clDice, 100
  epochs on GPU (clDice helps in-distribution but does not explain the OOD
  collapse). Stages 1–3 already have 100-epoch numbers from 5.2/5.4
  (zero-shot, Plain LoRA, Spacing-Conditioned LoRA with clDice), so the
  ablation chain is complete with the existing checkpoints. `run_ablation.py`
  is still there for a full re-run if needed, but re-running stages 1–3
  would just repeat existing results.
- **OOD investigation scripts.** `evaluate_fov_confound.py` and
  `analyze_film_layers.py` are **DONE** and were run on the 100-epoch
  checkpoints (5.5, 5.6): strong, significant support for the FOV hypothesis
  (5.5) and an updated, not yet confirmed FiLM hypothesis (5.6). Still to do:
  train on a wider range of real spacings so unseen spacing and unseen FOV
  are no longer mixed.
- Multiple seeds: see the bullet above (`colab_run.py`, written, not run).
  Few-shot experiments; cross-organ generalization (pseudo-label organs vs.
  real-GT kidney, which should not be compared directly, see section 6): not
  done.
- **Controlled 100-epoch retrain on a Kaggle GPU kernel.** DONE (5.4).
  Confirms the OOD collapse is real and not from too little training. Getting
  the kernel to work turned up six environment-specific bugs (input paths,
  disk quota, clDice import, SAM-Med3D import, GPU/CUDA mismatch,
  pos_embed/mask_decoder shapes); see items 7–10 in section 4.

## 8. How to check the results

- `ls -la checkpoints/best_model.pth` (pseudo-label track) /
  `ls -la checkpoints_real_gt/best_model_real_gt.pth` (real-GT track). The
  timestamps only change when a training run finishes and calls
  `torch.save()`.
- `cat checkpoints/train_state.json` / `cat checkpoints_real_gt/train_state.json`
  shows the epoch count, updated after every epoch.
- `python evaluate.py`, `python baseline_zeroshot.py`,
  `python evaluate_topology.py` (pseudo-label track) and
  `python evaluate_real_gt.py` (real-GT track, `checkpoint_path=None` for
  zero-shot). They print the numbers directly; nothing is cached.
- `python diagnose_dataloader.py` (pseudo-labels) /
  `python diagnose_real_gt_dataset.py` (real GT) print shape/spacing/
  foreground statistics per sample (no image files are written).
- `python utils/dataset_verification.py` /
  `python utils/kidney_metadata_verification.py` rebuild the metadata CSVs
  from the files on disk.

## 9. Repository layout

**Cleanup (2025-09-13):** removed `models/adapters/` (a duplicate
`PhysicsAwareAdapter`; `models/physics_adapter.py` had become a small
re-export of it, so I merged it back into one file at the import path
everything already used), an empty `pipeline/` folder, an old notes file
saved as `test.sh`, two old screenshots, and `experiments.ipynb` (out of
date: it was written before most of the fixes in this README, e.g. it expects
the model to return 2 values while it now returns 3; the `diagnose_*.py`
scripts replace it). Added a `.gitignore` for `venv/`, the generated `data/`
folders, checkpoints and the cloned `external/` repos (see section 10 for
how to regenerate them). The tracked repo was 31 files, 176KB: source code
plus two small metadata CSVs.

```
config.py                  organ metadata (spacing, ridge polarity), training hyperparameters
train.py                   resumable training loop (pseudo-label track), checkpoints every epoch
train_real_gt.py           resumable training loop, PhysicsAwareAdapter on real-GT track
train_plain_lora.py        resumable training loop, Plain LoRA on real-GT track
train_spacing_lora.py      resumable training loop, Spacing-Conditioned LoRA on real-GT track
evaluate.py                Dice/HD95 evaluation, pseudo-label track (checkpoint_path=None -> zero-shot)
evaluate_real_gt.py        Dice/IoU/clDice/HD95/ASSD evaluation, real-GT track (model_class= selects architecture)
evaluate_topology.py       Betti number evaluation (pseudo-label track)
evaluate_fov_confound.py   OOD field-of-view confound test (large 5.2um cube downsampled to match training FOV)
analyze_film_layers.py     per-layer gamma/beta breakdown for Spacing-Conditioned LoRA's FiLM output
run_ablation.py            zero-shot -> +LoRA -> +spacing -> +clDice progressive ablation, one script
baseline_zeroshot.py       thin wrapper: evaluate() with no fine-tuning (pseudo-label track)
diagnose_dataloader.py     dataset sanity-check, numeric output only (pseudo-labeled organs)
diagnose_real_gt_dataset.py  dataset sanity-check, numeric output only (real-GT kidney)
models/
  hip_vessel_model.py      HiPVesselAdaptModel wrapper (SAM-Med3D + PhysicsAwareAdapter + collapse head)
  physics_adapter.py       PhysicsAwareAdapter (spacing-conditioned bottleneck adapter)
  collapse_head.py         CollapseDetectionHead (not trained/validated, see limitations)
  lora.py                  LoRALinear + inject_lora(): generic LoRA for attention qkv/proj layers
  plain_lora_model.py      PlainLoRAModel: frozen backbone + LoRA + decoder, NO spacing conditioning
  spacing_lora.py           SpacingConditionedLoRALinear + inject_spacing_lora(): FiLM-modulated LoRA
  spacing_lora_model.py     SpacingConditionedLoRAModel: frozen backbone + spacing-modulated LoRA + decoder
losses/
  combined_loss.py         HiPVesselLoss: Dice + clDice + anisotropic penalty + weighted BCE (use_cldice= toggle)
utils/
  convert_to_memmap.py     raw .jp2 -> np.memmap converter (pseudo-label track)
  dataset.py               MultiOrganHiPCTDataset (patch sampling, pseudo-label caching)
  pseudo_labels.py         Frangi vesselness pseudo-label generator
  topology_metrics.py      Betti number computation (gudhi cubical complex)
  dataset_verification.py       spacing/GT check for the 3 pseudo-labeled organs -> dataset_verification.csv
  kidney_metadata_verification.py  spacing/GT table for SenNet+HOA kidneys -> data/kidney_metadata_verified.csv
  convert_kidney_gt_to_memmap.py   real-GT kidney TIFFs (in zip) -> np.memmap converter
  real_gt_dataset.py            RealGTKidneyDataset: loads expert-annotated kidney patches (memmap or raw_tiff_root backend)
  load_cldice.py                 loader for external/clDice's soft_cldice (see section 4, bug 7)
  load_sam_med3d.py              loader for external/SAM-Med3D's segment_anything (see section 4, bug 8)
  patch_external_repos.py        applies the needed fixes to a fresh clone of external/SAM-Med3D (see section 4, bug 10)
external/
  SAM-Med3D/               cloned backbone repo + downloaded pretrained checkpoint
  clDice/                  cloned topology-loss repo
data/
  raw/<organ>_jp2/         raw ESRF downloads (Heart/Lung/Kidney, pseudo-labeled)
  memmap/<organ>.dat       converted uncompressed volumes (fast random access)
  pseudo_cache/            cached Frangi pseudo-labels per grid block
  blood-vessel-segmentation.zip   SenNet+HOA real-GT kidney dataset (source zip)
  memmap_kidney_gt/        converted real-GT kidney image+label memmaps (~45GB, 5 samples)
  kidney_metadata_verified.csv    per-sample spacing + GT availability (see README 2.1)
checkpoints/                current fine-tuned model, pseudo-label track
  best_model.pth
  train_state.json          resume state (completed_epochs)
checkpoints_real_gt/         PhysicsAwareAdapter fine-tuned model, real-GT track
  best_model_real_gt.pth
  train_state.json
checkpoints_plain_lora/      Plain LoRA fine-tuned model, real-GT track
  best_model_plain_lora.pth
  train_state.json
checkpoints_spacing_lora/    Spacing-Conditioned LoRA fine-tuned model, real-GT track (10-epoch CPU run)
  best_model_spacing_lora.pth
  train_state.json
checkpoints_plain_lora_100ep_gpu/     Plain LoRA, 100-epoch GPU retrain (section 5.4), downloaded from Kaggle
  best_model_plain_lora.pth
checkpoints_spacing_lora_100ep_gpu/   Spacing-Conditioned LoRA, 100-epoch GPU retrain (section 5.4), downloaded from Kaggle
  best_model_spacing_lora.pth
kaggle_kernels/              scripts that ran on Kaggle for sections 5.4-5.8
  kaggle_train_kernel.py       main 100-epoch run: Plain LoRA + Spacing-Conditioned LoRA, train + eval
  kaggle_ablation_kernel.py    ablation stage 4: Spacing-Conditioned LoRA with clDice off
  kernel-metadata-main.json    kernel-metadata.json for kaggle_train_kernel.py (kaggle kernels push -p .)
  kernel-metadata-ablation.json  kernel-metadata.json for kaggle_ablation_kernel.py
colab_run.py                  hyperparameter sweep + multi-seed runs (section 7), for Google Colab
```

## 10. Setup

All entry points and modules are plain `.py` files (for a short time the
entry points were `.ipynb` notebooks, but I switched back to `.py`).

```bash
pip install monai SimpleITK scikit-image scipy gudhi
# external/SAM-Med3D and external/clDice already cloned; checkpoint already
# downloaded to external/SAM-Med3D/ckpt/sam_med3d.pth
python utils/convert_to_memmap.py     # one-time: raw jp2 -> memmap (~15-20 min)

python train.py                        # resumes automatically from checkpoints/
python evaluate.py
python baseline_zeroshot.py
python evaluate_topology.py

# Real-GT track (SenNet+HOA kidney)
python utils/convert_kidney_gt_to_memmap.py   # one-time
python train_real_gt.py                        # PhysicsAwareAdapter
python train_plain_lora.py                     # Plain LoRA
python train_spacing_lora.py                   # Spacing-Conditioned LoRA
python evaluate_real_gt.py
```

## 11. Second round: held-out benchmark (for the MedIA paper)

The numbers in sections 5.2-5.8 were measured on random patches from the
training kidneys (in-distribution) and from kidney_1_voi (OOD). For the paper
everything is re-evaluated on held-out data. Plan and reasons: `baki.sh`
section 7. Paper draft: `PAPER_MEDIA.md`.

Test data (labels released by Jain et al. 2026 on Zenodo 10.5281/zenodo.19685381):

| Test volume | Voxel (µm) | What it tests |
|---|---|---|
| kidney_5 | 50.28 | other kidney, same donor |
| kidney_6 | 63.08 | other donor, other beamline (BM18) |
| kidney_additional_voi_13um | 13 | 4x finer (only slices 0-2225 have images) |
| kidney_1_voi | 5.2 | ~10x finer |

New code:

```
utils/fetch_zenodo_subset.py     download parts of the Zenodo zip (HTTP range requests)
benchmark/rois.py                fixed evaluation ROIs from local data
benchmark/build_rois_zenodo.py   same ROIs, downloading only the slices needed
benchmark/infer.py               sliding-window inference; --resample-um / --spacing-um for the spacing/FOV test
benchmark/score.py               Dice, clDice, NSD (t=0, t=1), HD95, ASSD, graph stats, bootstrap CIs, paired tests
benchmark/import_team_rle.py     Kaggle top-5 predictions -> ROI predictions
benchmark/nifti_io.py            ROI export/import for nnU-Net and vesselFM
utils/slab_dataset.py            fast training sampler (slabs in memory)
models/refine_head.py            full-resolution refinement head (--refine)
train_lora_v2.py                 training: 128^3, foreground sampling, spacing augmentation
kaggle_kernels/kaggle_rois_kernel.py     builds the ROIs on Kaggle (CPU)
kaggle_kernels/kaggle_v2_kernel.py       train + benchmark one configuration (GPU)
kaggle_kernels/kaggle_nnunet_kernel.py   nnU-Net baseline (GPU)
kaggle_kernels/kaggle_vesselfm_kernel.py vesselFM zero-shot baseline (GPU)
```

Local example (CPU is enough for the Kaggle team baselines):

```bash
python -m utils.fetch_zenodo_subset kidney_6 team_predictions
python -m benchmark.rois --raw-tiff-root data/zenodo --sets kidney_6
python -m benchmark.import_team_rle --teams 1 2 3 4 5
python -m benchmark.score data/benchmark/preds/kaggle_team1 --out benchmark_results/kaggle_team1.json
```

Results so far are in `benchmark_results/` and test.sh section 3.12.
