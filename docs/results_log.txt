HiP-VesselAdapt — Technical data & results log
================================================================
All values below are output of scripts I ran on the data in this
repository. Nothing is estimated or rounded up. Where a script can
regenerate a number, the script is named so it can be checked again.

================================================================
1. DATASETS
================================================================

--------------------------------------------------------------
1.1 Track A: ESRF HiP-CT (Heart / Lung / Kidney) — pseudo-labels
--------------------------------------------------------------
Source: European Synchrotron Radiation Facility (ESRF) Human Organ Atlas,
raw .jp2 slice stacks. No vessel annotations come with these downloads
(checked: no file under data/raw/ has mask/label/seg/annotation/gt/
groundtruth in its name).

Regenerate this table: python utils/dataset_verification.py
(writes dataset_verification.csv)

  Organ  | Slices | Voxel spacing (um) | Spacing source                | Memmap shape (D,H,W)
  -------|--------|---------------------|--------------------------------|----------------------
  Heart  | 2074   | 26.04               | filename-embedded, verified   | (2074, 960, 960)
  Lung   | 1229   | 169.36              | filename-embedded, verified   | (1229, 870, 870)
  Kidney | 2141   | 50.00               | filename-embedded, verified   | (2141, 1108, 1644)

Ground truth for training/eval: NO real labels. All masks are pseudo-labels
from a multiscale Frangi (Hessian) vesselness filter
(utils/pseudo_labels.py: gaussian denoise, sigma=1.0 -> frangi(sigmas=(1,2,3))
-> Otsu threshold -> remove_small_objects(max_size=30) -> morphological
closing). Ridge polarity (black_ridges=True for all 3 organs) was chosen by
looking at the scans: vessel/airway lumen is darker than the surrounding
tissue. (The diagnostic scripts only print numbers; the code does not save
any images.)

--------------------------------------------------------------
1.2 Track B: SenNet+HOA Kidney (Kaggle "blood-vessel-segmentation") — expert GT
--------------------------------------------------------------
Source: Kaggle competition "SenNet + HOA - Hacking the Human Vasculature in
3D" (data/blood-vessel-segmentation.zip, 21GB compressed / 43.5GB
uncompressed / 14,365 files, download checked complete).

Voxel spacing is NOT stored in the TIFF files (XResolution/YResolution tags
are just placeholder values of (1,1)) and there is no metadata file with it
in the zip (the only non-train/test files are sample_submission.csv and
train_rles.csv, neither has spacing). All spacing values below are from
the dataset paper (Nature Methods / PMC): "Vasculature segmentation in 3D
hierarchical phase-contrast tomography images of human kidneys",
https://pmc.ncbi.nlm.nih.gov/articles/PMC13470036/, Table 1.

Regenerate this table: python utils/kidney_metadata_verification.py
(writes data/kidney_metadata_verified.csv)

  Sample          | Specimen                | Spacing (um) | Label slices | Split        | Real GT?
  ----------------|--------------------------|--------------|--------------|--------------|----------
  kidney_1_dense  | LADAF-2021-17-Right      | 50.00        | 2279         | train        | YES
  kidney_1_voi    | LADAF-2021-17-Right      | 5.20         | 1397         | train        | YES
  kidney_2        | S-20-28-Kidney           | 50.00        | 2217         | train        | YES
  kidney_3_dense  | LADAF-2020-27-Kidney-2   | 50.16        | 501          | train        | YES
  kidney_3_sparse | LADAF-2020-27-Kidney-2   | 50.16        | 1035         | train        | YES
  kidney_5        | LADAF-2021-17-Left       | 50.28        | 0            | public_test  | NO (withheld)
  kidney_6        | LADAF-2022-13-Bottom     | 63.08        | 0            | private_test | NO (withheld)

Note: kidney_1_dense and kidney_1_voi are the SAME specimen at two
resolutions ~10x apart (50.00um vs 5.20um), both with dense expert labels,
so they form a same-organ multi-resolution pair (not synthetic). kidney_1_voi
is kept out of training by default (exclude_for_ood parameter in
train_real_gt.py) for the resolution test.

Label format: binary TIFF, values only {0, 255} (0 = background,
255 = vessel), checked on several samples from kidney_1_dense, kidney_2,
kidney_3_sparse. Foreground fraction on whole slices: 0.294% - 1.539%
(three samples checked). On random 64^3 patches, 71-80 out of 100 had NO
annotated foreground (vessels are sparse compared to the patch size); how
this is handled is described in section 3.

================================================================
2. MODEL / ARCHITECTURE
================================================================

Backbone: SAM-Med3D (https://github.com/openmedlab/SAM-Med3D), official
pretrained checkpoint (external/SAM-Med3D/ckpt/sam_med3d.pth, 1.2GB,
from the repo's Google Drive link). Loaded with model_type="vit_b_ori"
(768-dim encoder, 128^3 pretraining resolution), which is the right variant
for this checkpoint: it loads with 0 missing and 0 unexpected keys. (The
registry default "vit_b" is a different 384-dim/256^3 variant and would
not match.)

  Total parameters (HiPVesselAdaptModel): 100,622,065
  Trainable parameters (adapter + mask_decoder + collapse_head): 7,697,137 (7.65%)

Modules:
  - models/physics_adapter.py: PhysicsAwareAdapter -- spacing-conditioned
    bottleneck adapter (spacing -> Linear/GELU/Linear -> elementwise
    modulation -> low-rank down/up projection residual).
  - models/collapse_head.py: CollapseDetectionHead -- Conv3d/BatchNorm/ReLU/
    Conv3d/Sigmoid multi-task head. It runs, but is NOT TRAINED/VALIDATED;
    there is no supervision signal for it yet.
  - models/lora.py + models/plain_lora_model.py: PlainLoRAModel -- frozen
    backbone, LoRA (r=4, alpha=1.0) in all 24 attention qkv/proj layers
    (12 transformer blocks x 2 layers, counted), NO spacing conditioning.
    Trained and evaluated, see section 3.2.
  - models/spacing_lora.py + models/spacing_lora_model.py:
    SpacingConditionedLoRAModel -- the main proposed method. Same LoRA
    injection points as PlainLoRAModel, but each LoRA branch output is also
    FiLM-modulated (gamma, beta) by a small per-layer spacing encoder that
    gets the voxel spacing. The spacing is passed in by setting a plain
    (non-parameter) `current_spacing` attribute on each injected layer just
    before the encoder forward call, so SAM-Med3D's Attention.forward()
    does not need to be changed. Fixed a design problem BEFORE training:
    the FiLM hidden size first grew with the attention out_features, which
    added ~31M trainable params for the qkv layers alone (31.79% of the
    model). Changed to a fixed small hidden size (16), giving 8.88%
    trainable, comparable to Plain LoRA's 7.75%. Trained and evaluated,
    see section 3.2.
  - SAM-Med3D's mask_decoder is a PROMPTABLE head. Without a prompt it
    ended up predicting exactly 1.0 everywhere after training (checked by
    looking at the output probabilities). Fixed with an auto-prompt: two
    corner points covering the whole volume, acting as a whole-volume box
    (SAM-Med3D's box-embedding code has a hard-coded 2D reshape that
    crashes in this 3D port).

Loss (losses/combined_loss.py, HiPVesselLoss):
  total = dice_loss + alpha*cldice_loss + beta*anisotropic_penalty + gamma*weighted_BCE
  alpha=0.5, beta=0.05, gamma=1.0 (defaults)
  - dice_loss: standard soft Dice on probabilities.
  - cldice_loss: external/clDice's soft_cldice (topology-preserving).
  - anisotropic_penalty: mean squared spatial gradient of the prediction,
    scaled by the per-axis voxel spacing.
  - weighted_BCE: BCEWithLogitsLoss on logits, pos_weight computed per
    batch from the foreground ratio (clamped to max 50). Added because with
    only Dice+clDice, training collapsed to predicting 1.0 everywhere.
    (That solution has ~0.96 total loss vs ~0.02 for a near-perfect
    prediction under dice+clDice+aniso ALONE, but training still ended up
    there, because sigmoid gradients vanish at saturation before the
    optimizer can find the low-loss region. BCE on logits does not have
    this problem.)

================================================================
3. RESULTS (all numbers are script output)
================================================================

--------------------------------------------------------------
3.1 Track A (pseudo-labeled organs) -- python evaluate.py / baseline_zeroshot.py / evaluate_topology.py
--------------------------------------------------------------
Zero-shot vs 20-epoch fine-tuned (24 patches/epoch):

  Organ  | Zero-shot DSC | Fine-tuned DSC | Zero-shot HD95 (vox) | Fine-tuned HD95
  -------|---------------|----------------|-----------------------|------------------
  Heart  | 0.021         | 0.060          | 40.3                  | 27.4
  Lung   | 0.037         | 0.077          | 17.5                  | 8.5
  Kidney | 0.051         | 0.082          | 19.4                  | 24.8

Betti numbers (topology), 20-epoch model, 5 patches/organ:

  Organ  | Pred b0 | Target b0 | |delta b0| | Pred b1 | Target b1 | |delta b1|
  -------|---------|-----------|-------------|---------|-----------|-------------
  Heart  | 32.6    | 34.4      | 1.8         | 0.2     | 2.6       | 2.4
  Lung   | 21.8    | 49.8      | 28.0        | 4.6     | 3.2       | 1.4
  Kidney | 41.0    | 89.6      | 48.6        | 1.8     | 12.8      | 11.0

Noise between epochs (Heart DSC): 0.077 at 10 epochs -> 0.043 at 18
epochs -> 0.060 at 20 epochs. Not monotonic; with only 5 eval patches per
organ the variance is high. The zero-shot vs fine-tuned COMPARISON is more
reliable than any single number.

--------------------------------------------------------------
3.2 Track B (SenNet+HOA expert GT) -- python evaluate_real_gt.py
--------------------------------------------------------------
100 random 64^3 patches per run; patches with no annotated foreground are
left out of the metrics (count shown in the last column).

TERMINOLOGY: none of these rows is "full" fine-tuning of the backbone --
the SAM-Med3D image encoder is frozen in every row. The rows only differ in
(a) whether there is a trainable adapter after the frozen encoder
(PhysicsAwareAdapter), trainable low-rank updates inside the encoder's
attention layers (Plain LoRA, r=4), or low-rank updates that are also
modulated by the voxel spacing (Spacing-Conditioned LoRA, r=4), and (b) the
mask_decoder, which is always trained.

  Model                                   | Trainable | Dice  | IoU   | clDice | HD95 (vox) | ASSD (vox) | Empty-target excluded
  -----------------------------------------|-----------|-------|-------|--------|------------|------------|------------------------
  Zero-shot SAM-Med3D (no training)       | 0%        | 0.088 | 0.052 | 0.076  | 48.1       | 25.2       | 79/100
  PhysicsAwareAdapter (10 epochs)         | 7.65%     | 0.157 | 0.094 | 0.139  | 36.2       | inf*       | 71/100
  Plain LoRA, r=4 (10 epochs)             | 7.75%     | 0.067 | 0.038 | 0.057  | 45.7       | 24.8       | 76/100
  Spacing-Conditioned LoRA, r=4 (10 ep)   | 8.88%     | 0.206 | 0.131 | 0.261  | 23.4       | inf*       | 80/100

MAIN IN-DISTRIBUTION RESULT: Spacing-Conditioned LoRA (the proposed method)
is best on every metric, including against Plain LoRA, its closest ablation
(same architecture and training; the ONLY difference is the spacing
modulation of the LoRA branch). The extra trainable cost is small (8.88% vs
7.65-7.75%), so it is not just more capacity.

PLAIN LORA RESULT: after 10 epochs Plain LoRA is worse than even the
untrained zero-shot baseline, and well below both adapter-based methods.
Possible reasons, none confirmed: LoRA changes attention INSIDE the frozen
encoder while the adapter changes the encoder OUTPUT right before the
decoder (a shorter optimization path for the same ~170-220 gradient
steps); all three trained runs used the same loss weights/learning rate
(tuned for PhysicsAwareAdapter), never re-tuned per architecture -- a
confound, but Spacing-Conditioned LoRA still won despite it; 10 epochs may
be too few for Plain LoRA in particular -- but Spacing-Conditioned LoRA
(same LoRA + spacing modulation) reached a much lower training loss with
the same setup (1.56 vs Plain LoRA's 2.05), which suggests the spacing
conditioning itself made the difference, not just "more LoRA training".
Needs more work (LR sweep per architecture, several seeds) before this
ranking is final.

* ASSD=inf for PhysicsAwareAdapter AND Spacing-Conditioned LoRA: MONAI
warned "the prediction of class 0 is all 0", i.e. at least one patch had a
completely empty prediction. Plain LoRA's ASSD with the same evaluation was
finite (24.8). Needs checking before trusting ASSD for the two better
models.

RESOLVED 2026-09-15 (diagnose_assd_inf.py): I had earlier tested this
"completely empty prediction" idea on 15 patches and found 0 cases, so the
cause stayed open. A new, bigger run (125 non-empty patches for
PhysicsAwareAdapter, 92 for Spacing-Conditioned LoRA, calling MONAI's
SurfaceDistanceMetric per patch instead of only on the batch average) DID
reproduce one ASSD=inf case for Spacing-Conditioned LoRA, and it was a
completely empty prediction (pred_sum=0) on a patch with a tiny annotated
target (22 voxels, 0.01% of the 64^3 patch, kidney_3_dense).
PhysicsAwareAdapter had 0/125 such cases in the same run. CONCLUSION: the
empty-prediction idea was right -- it is a rare event that depends on which
patches are drawn (about 1% of non-empty patches in this draw), caused by
the model completely missing tiny/sparse vessel pieces after only 10
epochs, not a fixed property of either checkpoint. The earlier "0 of 15"
most likely just did not draw one of these rare patches; 15 samples are too
few to catch a ~1% event reliably. I also tested a second possibility (a
FULLY SATURATED prediction, the opposite case, also has no surface and
would also give inf): 0 cases of either kind for PhysicsAwareAdapter and
0 saturated cases for Spacing-Conditioned LoRA, so it is not a factor here.

Bug fixed while building this evaluation: the clDice library's
soft_cldice.forward(y_true, y_pred) returns 1-clDice (a LOSS, same as
soft_dice() in the same file), not the score. The first run printed
clDice=0.978 next to Dice=0.025, which cannot be right. The 0.978 was the
loss. Changed to `1.0 - cldice_fn(targets, binary_pred).item()`; clDice is
now in the same range as Dice, as expected. The training loss in
losses/combined_loss.py was already correct (there we do want the loss).

Model size:
  PhysicsAwareAdapter:      Total 100,622,065 / Trainable 7,697,137 (7.65%)
  Plain LoRA:               Total 100,730,416 / Trainable 7,805,488 (7.75%)
  Spacing-Conditioned LoRA: Total 101,985,328 / Trainable 9,060,400 (8.88%)
  (The spacing FiLM networks add ~1.25M params over Plain LoRA; the
  trainable budgets are of the same order, so the performance gap is not
  explained by model size.)

--------------------------------------------------------------
3.3 RESOLUTION-OOD TEST (key experiment) -- REVERSES the ranking of 3.2
--------------------------------------------------------------
All three trained models evaluated on kidney_1_voi (5.2um), which was not
used in ANY training (a ~10x resolution shift; all training data was
~50um). 40 patches sampled, 34-35 skipped as empty-target.

  Model                     | In-dist Dice (50um) | OOD Dice (5.2um) | OOD IoU | OOD clDice | OOD HD95 | OOD ASSD
  ---------------------------|----------------------|-------------------|---------|------------|----------|----------
  PhysicsAwareAdapter        | 0.157                | 0.064             | 0.035   | 0.059      | 53.1     | 25.3
  Plain LoRA                 | 0.067                | 0.067 (best)      | 0.036   | 0.210      | 50.7     | 25.9
  Spacing-Conditioned LoRA   | 0.206 (best in-dist) | 0.002 (worst)     | 0.001   | 0.008      | 41.6     | 18.7

THE IN-DISTRIBUTION RANKING REVERSES. Spacing-Conditioned LoRA, the clear
in-distribution winner, drops to almost zero Dice (0.002) at the unseen
resolution, worse than both other models on Dice/IoU/clDice.

HYPOTHESIS 1, TESTED AND REJECTED: I expected the spacing-FiLM network
(trained on basically one spacing, ~50um) to extrapolate badly ~10x out of
range and give large/erratic gamma/beta at 5.2um. Checked:
spacing_film(spacing) at 50um gives gamma in [0.70, 1.54] (mean 1.05); at
5.2um gamma is in [0.97, 1.06] (mean 1.01) -- SMALLER and closer to
identity than in-distribution, the opposite of what I expected. The FiLM
network is NOT confirmed as the cause.

HYPOTHESIS 2 (field-of-view shift), PARTLY SUPPORTED: a 64^3 patch at
5.2um covers a ~10x smaller physical field of view (~333um vs ~3200um
across) than any training patch. Evidence: the target foreground fraction
is ~20x higher on average at 5.2um (mean 0.67%, up to 6.7%, n=10) than at
50um (mean 0.032%, up to 0.67%, n=10), which fits the same vessel filling
more of a much smaller window. z-score normalization is not the problem
(both resolutions normalize to mean~=0, std~=0.90-0.98). However, the
frozen image_encoder output statistics are almost the same for both
resolutions (50um: mean=0.0013 std=0.218; 5.2um: mean=0.0014 std=0.221),
so there is no activation blow-up in the backbone. The open question is
whether the encoder/LoRA weights (trained only on 50um) pick up the WRONG
content at 5.2um even though the overall statistics are similar -- not
confirmed, would need a per-channel/spatial analysis.

CONCLUSION FOR THE PAPER: the method is NOT shown to generalize across
resolutions -- right now it generalizes WORSE than Plain LoRA on real OOD
data. This is more important than the in-distribution win in 3.2 and has
to be the main finding in the paper. Cause still open.

--------------------------------------------------------------
3.4 Controlled 100-epoch GPU retrain -- CONFIRMS 3.3, not under-training
--------------------------------------------------------------
GPU retrain (100 epochs, both architectures, same data split/protocol as
3.2/3.3) to answer the question left by the 10-epoch CPU numbers: is the
OOD collapse real, or just too little training? Output of
evaluate_real_gt.py from Kaggle kernel version 7
(rabeya100x/hip-vesseladapt-100ep-gpu-train), taken from kaggle
kernels_logs after the kernel reached COMPLETE:

  Model                                  | Trainable | In-dist Dice        | OOD Dice
  ----------------------------------------|-----------|----------------------|----------------------
  Plain LoRA (100ep, GPU)                | 7.75%     | 0.1838 (n=26/100)    | 0.1672 (n=5/40)
  Spacing-Conditioned LoRA (100ep, GPU)  | 8.88%     | 0.3145 (n=18/100)    | 0.0058 (n=5/40)

CONFIRMED, AND MORE TRAINING MAKES IT CLEARER:
- In-distribution lead grew: 0.206 vs 0.067 (10ep) -> 0.3145 vs 0.1838
  (100ep), Spacing-Conditioned LoRA ahead of Plain LoRA both times.
- OOD: Spacing-Conditioned LoRA still worst by >10x (0.0058 vs Plain LoRA's
  0.1672), and still ~98% below its OWN in-distribution score (0.0058 vs
  0.3145). Plain LoRA degrades gracefully (0.1672 OOD vs 0.1838 in-dist);
  Spacing-Conditioned LoRA collapses.
- So "10 epochs was too few" is not the explanation. The collapse is a
  property of the trained model. The OOD scripts (evaluate_fov_confound.py,
  analyze_film_layers.py, run_ablation.py -- all written, section 4) should
  now be run on these 100-epoch checkpoints, not the old 10-epoch ones.

CAVEAT: only 5 of the 40 OOD patches had any annotated vessel, for BOTH
models, so each OOD number above is an average over just 5 patches. The
gap (0.1672 vs 0.0058) is big enough to mean something even at this n,
but a larger OOD sample is still on the to-do list (section 4).

TWO OTHER THINGS IN THIS RUN'S OUTPUT:
- Final training loss is almost the same for both models (Plain LoRA
  1.4737, Spacing-Conditioned LoRA 1.4900, epoch 100/100). In the 10-epoch
  run Spacing-Conditioned LoRA's loss was clearly lower (1.56 vs 2.05). At
  100 epochs the training loss no longer reflects the big Dice gap on
  real GT.
- HD95 and ASSD were nan (n=0) in ALL FOUR evaluations of this run (both
  models, in-dist and OOD). The CPU run had real HD95 values and only
  ASSD=inf for two of the three models. _safe_metric() in
  evaluate_real_gt.py was silently catching whatever exception MONAI's
  HausdorffDistanceMetric/SurfaceDistanceMetric threw on every non-empty
  patch, while Dice/IoU/clDice worked on the same GPU tensors in the same
  loop -- so it is not a problem with moving the tensors. Changed
  _safe_metric() to print the exception the first time each metric fails
  instead of just returning None, so the next run will show the cause.
  Not diagnosed yet at the time of writing.

--------------------------------------------------------------
3.5 Field-of-view confound test -- strong support
--------------------------------------------------------------
Downloaded the 100-epoch checkpoints from the Kaggle kernel output (signed
URLs for just the two best_model_*.pth files, not the whole working
directory) and ran evaluate_fov_confound.py locally on both, samples=40.
Method: instead of a native 64^3 crop of kidney_1_voi (5.2um), take a much
bigger cube whose PHYSICAL size matches a 64^3 patch at the training
spacing (~50um), downsample it to 64^3 and evaluate on that -- same 5.2um
data, matched field of view.

  Model                     | In-dist (3.2) | OOD native-scale (3.3/3.4) | OOD FOV-matched (this test)
  ---------------------------|---------------|------------------------------|------------------------------
  Plain LoRA                | 0.1838 (n=26) | 0.1672 (n=5/40)             | 0.2895 (n=40/40)
  Spacing-Conditioned LoRA  | 0.3145 (n=18) | 0.0058 (n=5/40)             | 0.2215 (n=40/40)

CLEAR SUPPORT for the field-of-view hypothesis:
- All 40 FOV-matched patches had vessel (0 empty) vs 35/40 EMPTY in the
  native-scale OOD crop. At 5.2um a 64^3 window covers too little tissue to
  reliably contain a vessel, which is a big part of why that evaluation had
  only 5 usable patches.
- Spacing-Conditioned LoRA's OOD Dice goes from 0.0058 (~98% below its
  in-dist score) to 0.2215 (~70% of in-dist) just by matching the field of
  view on the SAME OOD data.
- Plain LoRA also improves a lot (0.1672 -> 0.2895), so FOV is not the
  WHOLE story (Plain LoRA never collapsed), but it is a major factor for
  both, and the main one for Spacing-Conditioned LoRA.
- With matched FOV the ranking flips again: Plain LoRA is slightly better
  than Spacing-Conditioned LoRA. SIGNIFICANT -- a separate scoring run
  (significance_test_fov.py, 40 new patches per model; the means differ a
  bit from the table because evaluate_fov_confound.py draws a new random
  cube each call: Plain LoRA 0.3681 vs Spacing-Conditioned LoRA 0.2618)
  gives Mann-Whitney U p=0.0216, significant at alpha=0.05.

INTERPRETATION (hypothesis, not proven): most of the "OOD collapse" looks
like an ordinary shift to an unseen visual scale, not something specific
to how FiLM uses the spacing value. Section 3.6 gives a second reason why
Spacing-Conditioned LoRA suffers more than Plain LoRA from the FOV shift
that is left after this correction.

--------------------------------------------------------------
3.6 Per-layer FiLM breakdown -- a second lead
--------------------------------------------------------------
analyze_film_layers.py on the 100-epoch Spacing-Conditioned LoRA
checkpoint, all 24 injected layers, in-dist (50um) vs OOD (5.2um) spacing.
No image encoder pass needed, since spacing_film is a small MLP that only
takes the 3-value spacing.

The earlier aggregate check (3.3, 10-epoch checkpoint) rejected "FiLM
gives wild values at an unseen spacing" -- OOD gamma was, if anything,
closer to identity. At 100 epochs the per-layer results go in the SAME
direction but MUCH MORE STRONGLY: gamma at the IN-DISTRIBUTION spacing has
become large (e.g. layer 19: gamma_mean=6.52, gamma_max=21.6 at 50um;
layer 2: gamma_max=19.7 at 50um), while gamma in the SAME layers at the
OOD spacing stays near identity (layer 19: gamma_mean=1.59,
gamma_max=3.23 at 5.2um; layer 2: gamma_max=3.06 at 5.2um). Same pattern
across all 24 layers, not one outlier.

UPDATED HYPOTHESIS (to test further): FiLM does not produce DANGEROUS
values OOD. Instead, training pushed it toward large modulation TUNED TO
the training spacing, and after 100 epochs the model may depend on it to
segment 50um data well. At an unseen spacing the FiLM output goes back
toward identity (gamma~1, beta~0), roughly Plain LoRA's behaviour for that
layer -- but not exactly, since the LoRA A/B weights were trained together
with the strong modulation. Together with 3.5: most of the OOD failure
looks like an ordinary field-of-view shift, and FiLM makes it worse by
falling back toward (but not quite reaching) plain-LoRA-like behaviour
that the model was never trained for. Still a hypothesis -- next check is
training on a wider range of real spacings so "unseen spacing" is no
longer mixed up with "unseen FOV".

--------------------------------------------------------------
3.7 Ablation: clDice loss term, 100 epochs GPU
--------------------------------------------------------------
Stage 4 of run_ablation.py (Spacing-Conditioned LoRA with the clDice loss
off, using the new use_cldice=False flag in HiPVesselLoss), trained on
Kaggle GPU for 100 epochs, otherwise the same as 3.4.

  Model                                    | In-dist Dice     | OOD Dice
  -------------------------------------------|-------------------|------------------
  Spacing-Conditioned LoRA, WITH clDice (3.4)| 0.3145 (n=18/100) | 0.0058 (n=5/40)
  Spacing-Conditioned LoRA, NO clDice        | 0.1898 (n=19/100) | 0.0015 (n=3/40)

The clDice loss clearly helps in-distribution -- without it in-dist Dice
drops ~40% relative (0.3145 -> 0.1898), not only the clDice metric. It
does NOT explain the OOD collapse -- OOD Dice is very low either way
(0.0058 vs 0.0015, both close to noise at n=3-5). So clDice and the
FOV/FiLM findings (3.5, 3.6) are separate: clDice improves in-distribution
quality in general; the OOD collapse seems to come mostly from FOV
mismatch and FiLM's spacing-specific modulation, not from clDice.

--------------------------------------------------------------
3.8 Cause found: HD95/ASSD nan on Kaggle GPU (sections 3.4, 3.7)
--------------------------------------------------------------
With the better _safe_metric() logging (3.4), the ablation run showed the
actual exception. It is an environment problem, not a bug in this code:

  HausdorffDistanceMetric raised: .../cupy/_core/include/cupy/_cccl/thrust/
  thrust/detail/config/cpp_dialect.h(74): catastrophic error: #error
  directive: Thrust requires at least C++17.

For GPU tensors, MONAI's HausdorffDistanceMetric/SurfaceDistanceMetric use
a CUDA kernel compiled on the fly by CuPy for the distance transform; on
Kaggle's image this compilation fails (CuPy/Thrust/CUDA toolkit version
mismatch in their Docker image, nothing I can change). Dice/IoU/clDice do
not use that code, which is why only HD95/ASSD were nan. Fixed in
evaluate_real_gt.py by moving binary_pred/targets to CPU only for the
HD95/ASSD calls (.cpu() right before _safe_metric()), so the broken
GPU/CuPy path is not used. Checked locally (HD95/ASSD still correct). Not
re-checked on Kaggle GPU, since re-running a 100-epoch kernel only for this
would use a lot of quota for a fix I'm fairly sure of.

--------------------------------------------------------------
3.9 Kaggle GPU setup problems (getting the 3.4 run to work)
--------------------------------------------------------------
The numbers in 3.2/3.3 come from a 10-epoch run on this machine's CPU (no
local GPU). To do a proper 100-epoch retrain (both architectures,
in-distribution + OOD eval) in a reasonable time, I moved the same code to
a Kaggle kernel with a free GPU. The project code is uploaded as a private
Kaggle dataset (rabeya100x/hip-vesseladapt-code); the SenNet+HOA
competition data is attached by Kaggle directly (no need to re-upload the
21GB zip); the kernel script (kaggle_train_kernel.py) clones
external/SAM-Med3D and external/clDice fresh, downloads the SAM-Med3D
checkpoint with gdown, then runs train_plain_lora / train_spacing_lora
(100 epochs each) and evaluate_real_gt in-distribution and on the held-out
kidney_1_voi OOD set, for both architectures.

This hit several Kaggle-specific bugs that did not show up locally, each
found from the kernel logs (kaggle kernels_logs):

  1. Input paths: Kaggle mounts datasets/competitions under
     /kaggle/input/datasets/<owner>/<name>/ and
     /kaggle/input/competitions/<slug>/, not directly under /kaggle/input.
     Fixed with a bounded recursive search for a folder containing a known
     marker file.
  2. Disk quota: making np.memmap copies of kidney_1_dense + kidney_1_voi
     needs ~23GB, more than the /kaggle/working quota, and it filled up
     half way (Bus error, no space left on device). Fixed by adding a
     raw_tiff_root mode to RealGTKidneyDataset that reads TIFF slices
     directly from the read-only competition mount in __getitem__, with no
     memmap conversion.
  3. clDice import: external/clDice's cldice.py (cloned fresh every run)
     does `from .soft_skeleton import SoftSkeletonize`, a relative import
     that only works inside a package -- I load it as a standalone file,
     so it always fails. It only worked locally because I had earlier
     patched my local copy (not committed) with a try/except ImportError
     fallback, which the fresh Kaggle clone does not have. Fixed in
     utils/load_cldice.py: soft_skeleton.py is registered in sys.modules
     first, then cldice.py's source is read and the relative import is
     rewritten to an absolute one before running it.
  4. SAM-Med3D "not importable": segment_anything's __init__.py does
     `from utils import *` (absolute, meant for SAM-Med3D's own utils/
     package), which clashed with this project's top-level utils/ package
     -- whichever comes first on sys.path wins, and the bare
     `except ImportError: HAS_SAM_MED3D = False` in the model files hid the
     real error. Fixed in utils/load_sam_med3d.py: sam_med3d_path is put at
     the front of sys.path only for that one import, and any utils/utils.*
     entries added to sys.modules during it are removed afterwards so this
     project's utils imports still work elsewhere. Also removed the silent
     except so a future failure shows the real traceback.
  5. CUDA "no kernel image is available for execution on the device":
     after 1-4, the kernel got to training and failed right away --
     Kaggle had given it a Tesla P100 (compute capability sm_60), which the
     PyTorch build in the kernel image does not support (sm_70 minimum).
     kernel-metadata.json did not ask for a specific accelerator. Fixed by
     setting "machine_shape": "NvidiaTeslaT4" in kernel-metadata.json
     (sm_75, supported).
  6. "RuntimeError: The size of tensor a (4) must match the size of tensor
     b (8)" in SAM-Med3D's image encoder: with a working GPU, training got
     to the first forward pass and crashed adding a positional embedding
     sized for 128^3 pretraining (8 tokens/axis at patch 16) to the 64^3
     input (4 tokens/axis). The interpolation fix (pos_embed in
     image_encoder3D.py; dense and image prompt embeddings in
     mask_decoder3D.py) had only been an uncommitted local edit in
     external/SAM-Med3D, so it was lost on every fresh clone (same problem
     as the clDice import in 3). Fixed properly now with
     utils/patch_external_repos.py -- an idempotent text patch that checks
     itself (fails loudly if upstream no longer matches, instead of doing
     nothing) and is applied after every fresh clone, locally or on Kaggle.

Status: kernel version 7 (all six fixes) reached COMPLETE on Kaggle's free
GPU -- results in section 3.4 above.

--------------------------------------------------------------
3.10 Vascular graph validation -- third separate piece of evidence for
     the OOD collapse
--------------------------------------------------------------
evaluate_vascular_graph.py run locally (CPU) on both 100-epoch GPU
checkpoints, in-distribution (kidney_1_dense/kidney_2/kidney_3_dense/
kidney_3_sparse, 20 samples/id, same protocol as 3.2-3.4) and OOD
(kidney_1_voi, 40 samples). It skeletonizes (skimage skeletonize,
26-connected) the predicted mask and the GT mask of each patch, builds a
graph from each skeleton (utils/vascular_graph_metrics.py) and compares
connectivity/branching -- this checks the vessel TREE STRUCTURE, which
Dice/clDice/Betti numbers do not show directly. The median is used next to
the mean because one in-distribution patch (kidney_2, Spacing-Conditioned
LoRA) is an extreme outlier (430 predicted branch points, 123226 um
predicted length vs GT's 1 branch point/3467 um on that patch) that would
dominate the mean; all per-patch numbers are in
vascular_graph_results.json.

  Model / condition          | n (non-empty) | Pred comp (med) | GT comp (med) | Pred branch (med) | GT branch (med) | Pred endp (med) | GT endp (med)
  ----------------------------|----------------|------------------|-----------------|---------------------|--------------------|--------------------|-----------------
  Plain LoRA, in-dist        | 23/80          | 3.0              | 3.0            | 0.0                 | 1.0                | 6.0                | 8.0
  Plain LoRA, OOD            | 5/40           | 6.0              | 1.0            | 1.0                 | 1.0                | 10.0               | 3.0
  Spacing LoRA, in-dist      | 22/80          | 3.0              | 1.0            | 0.5                 | 0.5                | 7.0                | 3.0
  Spacing LoRA, OOD          | 8/40           | 11.0             | 1.0            | 15.5                | 0.5                | 20.5               | 2.5

THIRD, STRUCTURAL PIECE OF EVIDENCE for the OOD collapse (next to 3.5's
FOV confound and 3.6's FiLM result -- all three separate, not yet combined
into one explanation):
- In-distribution, both models produce LESS vessel structure than the GT
  (fewer branch points, shorter total length) -- normal
  under-segmentation, not fragmentation. For Spacing-Conditioned LoRA the
  in-distribution GT patches are mostly one component (median 1) while its
  predictions already have 3 components (median) -- a small
  over-fragmentation even in-distribution.
- OOD, the two architectures fail in DIFFERENT WAYS, not just by different
  amounts: Plain LoRA's OOD predictions fragment somewhat (6 components vs
  GT's 1) but the branch-point count stays close to GT (median 1 vs 1),
  which fits the "graceful degradation" in 3.5. Spacing-Conditioned LoRA's
  OOD predictions fragment much more (11 components vs GT's 1, 11x) AND
  over-branch (median 15.5 spurious branch points vs GT's 0.5, 31x) AND
  have MORE total skeleton length than the real vessel (median 1566 vs
  GT's 213 um) -- not a smaller or weaker version of the vessel tree, but
  scattered small disconnected over-branched pieces instead of it. This is
  a separate structural signature from 3.5/3.6, and all three point the
  same way: Spacing-Conditioned LoRA's OOD failure is a qualitative
  breakdown, not only a lower score.
- CAVEAT: OOD patch counts are small (5 and 8 non-empty patches, same data
  problem as in 3.4), so this supports 3.5/3.6 but is not a large-n result
  by itself.

--------------------------------------------------------------
3.11 Multi-seed results -- n=3
--------------------------------------------------------------
Every Dice number in 3.2-3.10 is from ONE training run per architecture
(the 3.4 100-epoch checkpoints, default seed). Two more runs with fixed
seeds (seed=1, seed=2; same 100 epochs, protocol and data split) were
trained on the GPU quota of a second Kaggle account
(kaggle_kernels/kaggle_multiseed_kernel.py and
kaggle_multiseed_kernel_seed2.py; kernels nargisakterr/
hip-vesseladapt-multiseed-seed1 and -seed2, finished 2026-09-16), because
the weekly quota of my main account was used up (see 3.9 and section 2 of
baki.sh). seed=1 used ~9788s of that account's 21600s weekly quota, so
seed=2 could run on the SAME account the same day and reach n=3 without
waiting for a reset. Per-seed numbers in multiseed_seed1_results.json /
multiseed_seed2_results.json; n=3 mean+/-std (sample stdev) computed by
generate_summary_tables.py:

  Model          | In-dist Dice (n=3)      | OOD Dice (n=3)
  ----------------|--------------------------|--------------------------
  Plain LoRA      | 0.1838 +/- 0.0644       | 0.0990 +/- 0.0591
  Spacing LoRA    | 0.2718 +/- 0.0464       | 0.0102 +/- 0.0079

  Per-seed values:
    Plain LoRA:   in-dist = [0.1838, 0.2482, 0.1194], OOD = [0.1672, 0.0620, 0.0679]
    Spacing LoRA: in-dist = [0.3145, 0.2224, 0.2786], OOD = [0.0058, 0.0193, 0.0055]
  (first value = the original 3.4 checkpoint, default seed; second =
  seed=1; third = seed=2)

WHAT n=3 SHOWS AND WHAT IT DOESN'T:
- OOD collapse: CONFIRMED. Spacing-Conditioned LoRA's OOD Dice was very
  low in ALL THREE runs (0.0058, 0.0193, 0.0055 -- all under 0.02), and
  Plain LoRA beat it OOD in ALL THREE (0.1672, 0.0620, 0.0679 -- every one
  higher, by ~3x to ~29x). Three out of three on both points is as clear
  as a 3-seed result can be. This is now the best-supported finding of the
  project.
- In-distribution ranking: now LEANS TOWARD Spacing-Conditioned LoRA.
  It wins in 2 of 3 seeds, by a large margin both times (0.3145 vs 0.1838;
  0.2786 vs 0.1194); Plain LoRA wins the other seed by a much smaller
  margin (0.2482 vs 0.2224). The n=3 means (0.2718 vs 0.1838, a 0.088 gap)
  are further apart than either model's std (0.046, 0.064), unlike at n=2,
  where gap and std were similar and the ranking was unclear. Still only
  n=3 -- not enough for a formal significance test and not reported as
  one -- but the "unclear at n=2" result no longer holds.
- CAVEAT: n=3 is still small -- enough to say "the OOD collapse is not a
  one-off" and "the in-distribution ranking leans one way", not enough for
  a proper confidence interval. No more seeds planned for now (n=3 was the
  target in baki.sh section 2); more would only help.

================================================================
4. STATUS / NOT DONE YET
================================================================
- Plain LoRA: DONE -- trained at 10 epochs (CPU) AND 100 epochs (GPU),
  evaluated in-distribution AND OOD both times (sections 3.2, 3.3, 3.4).
- Spacing-Conditioned LoRA: DONE -- trained at 10 epochs (CPU) AND 100
  epochs (GPU), evaluated in-distribution AND OOD both times (sections
  3.2, 3.3, 3.4). Best in-distribution at both epoch counts, COLLAPSES
  out-of-distribution at both -- not a validated method yet.
- Resolution-OOD test: DONE at 10 and 100 epochs (sections 3.3, 3.4) --
  the key finding, and it goes AGAINST the proposed method as it is. The
  100-epoch GPU retrain (3.4) shows it is NOT from too little training --
  the collapse stays (and the gap grows) with 10x more training.
- Controlled 100-epoch retrain on Kaggle GPU: DONE (3.4). This was the
  most important check before treating the OOD collapse as real rather
  than a training-length effect -- it is real.
- Cause of the OOD collapse: NOT FOUND. One idea (FiLM extrapolation)
  tested and rejected (3.3). Still open -- the most important remaining
  question, and the reason evaluate_fov_confound.py /
  analyze_film_layers.py / run_ablation.py exist (see below).
- Hyperparameter tuning per architecture: NOT DONE. All three trained
  models used the same loss weights/LR, tuned for PhysicsAwareAdapter.
  SCRIPT WRITTEN (colab_run.py, run_hyperparameter_sweep()), NOT RUN YET.
  Moved from Kaggle (free GPU quota, 6h/week, mostly used up by the
  100-epoch retrain + OOD experiments) to Google Colab -- uses the same
  train_plain_lora/train_spacing_lora/evaluate_real_gt functions and gets
  code + competition data through the Kaggle API. See the docstring in the
  file for how to run it.
- ASSD=inf/nan: DONE for the 100-epoch GPU case (3.4, 3.7, 3.8) -- cause
  found (Kaggle's CuPy/Thrust toolchain cannot compile the GPU
  distance-transform kernel used by MONAI's HD95/ASSD) and fixed (tensors
  moved to CPU for those two metrics). The ORIGINAL 10-epoch CPU case
  (3.2, ASSD=inf, not nan) is a separate issue, only partly checked --
  empty-prediction idea tested on 15 patches, 0 confirmed, inconclusive.
- More baselines (nnU-Net, VesselFM, VesselSDF): not implemented.
- Ablation study (zero-shot -> +LoRA -> +spacing -> +clDice loss term):
  DONE for stage 4 (3.7) -- Spacing-Conditioned LoRA with clDice off, 100
  epochs GPU. Stages 1-3 already have 100-epoch numbers from 3.2/3.4
  (zero-shot, Plain LoRA, Spacing-Conditioned LoRA with clDice), so
  re-running stages 1-3 of run_ablation.py would just repeat them. The
  script is still there for a full re-run if needed, but the ablation is
  complete with the existing checkpoints.
- Field-of-view confound test: DONE (3.5), on the 100-epoch checkpoints
  (downloaded from the Kaggle kernel output). Strong support for the FOV
  hypothesis -- Spacing-Conditioned LoRA's OOD Dice goes from 0.0058 to
  0.2215 once the field of view matches training. Not the whole story
  (Plain LoRA improves too), but the main effect for Spacing-Conditioned
  LoRA.
- Per-layer FiLM breakdown: DONE (3.6), on the 100-epoch
  Spacing-Conditioned LoRA checkpoint. Updated hypothesis: training pushed
  FiLM toward strong modulation specific to the training spacing, which
  goes back toward identity at the OOD spacing -- a second lead next to
  the FOV effect. Not confirmed as THE cause; training on a wider range of
  real spacings (so unseen spacing and unseen FOV are no longer mixed) is
  the next check.
- Multi-seed results: DONE (3.11) -- n=3, as planned (ran as two Kaggle
  kernels on a second account instead of colab_run.py's run_multiseed(),
  both fitting in that account's weekly quota). Results: the OOD collapse
  happens in all 3 seeds (both architectures, same direction every time);
  in-distribution, Spacing-Conditioned LoRA wins 2 of 3 seeds by a wide
  margin, but n=3 is not treated as a formal significance result.
- Few-shot experiments, cross-organ generalization, final figures: not
  done.
- Vascular graph validation: DONE (3.10) -- run on both 100-epoch GPU
  checkpoints, in-dist and OOD. Third separate piece of evidence for the
  OOD collapse, structurally different from 3.5/3.6: Spacing-Conditioned
  LoRA's OOD predictions fragment (11x GT's component count) and
  over-branch (31x GT's branch-point count) instead of degrading
  gracefully like Plain LoRA.
- Significance test for the FOV-matched Dice gap in 3.5: DONE (see 3.5)
  -- Mann-Whitney U p=0.0216, significant at alpha=0.05.

================================================================
5. HOW TO REPRODUCE THE NUMBERS
================================================================
python utils/dataset_verification.py           # Track A dataset audit
python utils/kidney_metadata_verification.py   # Track B dataset audit
python diagnose_dataloader.py                  # Track A pipeline sanity check (numeric output only)
python diagnose_real_gt_dataset.py             # Track B pipeline sanity check (numeric output only)
python diagnose_assd_inf.py                    # section 3.2 ASSD=inf root cause (numeric output only)
python train.py                                # Track A training (resumable)
python train_real_gt.py                        # Track B training (resumable)
python evaluate.py                              # Track A evaluation
python baseline_zeroshot.py                     # Track A zero-shot baseline
python evaluate_topology.py                     # Track A Betti numbers
python evaluate_real_gt.py                      # Track B evaluation (pass checkpoint_path=None for zero-shot)
python train_plain_lora.py                      # Track B training, Plain LoRA baseline (resumable)
python train_spacing_lora.py                    # Track B training, Spacing-Conditioned LoRA (resumable)
python run_ablation.py --epochs 10              # zero-shot -> +LoRA -> +spacing -> +clDice, all in one run
python evaluate_fov_confound.py --checkpoint <path> --model plain_lora|spacing_lora   # FOV-confound OOD test
python analyze_film_layers.py --checkpoint <path>   # per-layer gamma/beta breakdown, Spacing-Conditioned LoRA
python -c "from evaluate_vascular_graph import evaluate_vascular_graph; from models.plain_lora_model import PlainLoRAModel; evaluate_vascular_graph(checkpoint_path='checkpoints_plain_lora_100ep_gpu/best_model_plain_lora.pth', sample_ids=['kidney_1_dense','kidney_2','kidney_3_dense','kidney_3_sparse'], samples_per_sample=20, model_class=PlainLoRAModel)"
                                                 # vascular graph validation (section 3.10) -- change model_class/checkpoint_path/
                                                 # sample_ids=['kidney_1_voi'] for the other three rows of the table in 3.10

More detail: README.md in the same folder.

--------------------------------------------------------------
3.12 Held-out benchmark, first results: Kaggle top-5 on kidney_6 (2026-09-23)
--------------------------------------------------------------
Test labels + team predictions from Zenodo 19685381 (Jain et al. 2026).
8 ROIs of 160^3 voxels (63.08um, ~10mm) drawn with seed 0 from labelled voxels
(benchmark/rois.py). Team RLE predictions decoded and cropped to the ROIs
(benchmark/import_team_rle.py), scored with benchmark/score.py. Mean over ROIs,
95% bootstrap CI for Dice. JSON: benchmark_results/kaggle_team*.json.

  Team | Dice [95% CI]        | clDice | NSD t=0 | NSD t=1 | HD95 | ASSD | comp (med, GT 28) | branch pts (med, GT 190.5)
  -----|----------------------|--------|---------|---------|------|------|-------------------|---------------------------
  1    | 0.832 [0.778, 0.878] | 0.811  | 0.633   | 0.923   | 4.16 | 0.74 | 267               | 149.5
  2    | 0.705 [0.552, 0.830] | 0.798  | 0.629   | 0.819   | 17.8 | 3.54 | 25                | 222.5
  3    | 0.819 [0.768, 0.868] | 0.774  | 0.601   | 0.894   | 5.94 | 0.93 | 196               | 127.5
  4    | 0.818 [0.765, 0.867] | 0.768  | 0.591   | 0.884   | 10.4 | 1.29 | 55                | 133.0
  5    | 0.810 [0.758, 0.860] | 0.753  | 0.574   | 0.880   | 7.89 | 1.10 | 149               | 120.0

Check of the scorer: NSD t=0 ranks the teams 1 > 2 > 3 > 4 > 5, the same
order as the private leaderboard in Jain et al. (0.774, 0.755, 0.727, 0.712,
0.691). Absolute values differ (per-ROI voxel-based MONAI NSD vs. the
challenge metric on the whole labelled volume). Team 2's connected-component
post-processing shows up as the only near-correct component count, which also
matches their paper.

Note: a first version of these ROIs was built while two kidney_6 label slices
(0531, 0539) were missing after an interrupted download, which shifted the
slices below them. Those ROIs and scores were deleted; RealGTKidneyDataset now
refuses stacks with gaps in the slice numbering.

Our models are not on these ROIs yet (needs GPU, baki.sh section 7). For
comparison: the Kaggle winners reach Dice ~0.81-0.83 here, while our first-round
models had Dice ~0.18-0.31 on training-volume patches.
