# Does Knowing the Voxel Size Help? Spacing-Aware Low-Rank Adaptation of a 3D Foundation Model for Multi-Resolution HiP-CT Vessel Segmentation

**Short title:** Voxel size and LoRA for HiP-CT vessels

**Target journal:** Medical Image Analysis (Elsevier), full-length article.
**Status:** draft. Anything marked `[TODO]` depends on experiments that are
set up (see `docs/todo_log.txt`, section 7) but not run yet. Measured numbers give their
source file. Do not submit while any `[TODO]` is left. Reading notes for every
cited paper: `related_papers_notes.md`.

Author: Rabeya Noor `[TODO: affiliation or "Independent Researcher", city, country; ORCID; e-mail]`

---

## Highlights

(3-5 bullets, at most 85 characters each, spaces included.)

- Kidney vessel benchmark at four voxel sizes, from 63 µm down to 5.2 µm
- Giving LoRA the voxel size helped at 50 µm but failed at 10x finer voxels
- A 2x2 test separates the spacing input from the field of view
- `[TODO: one highlight on spacing augmentation / the refinement head]`

---

## Abstract

`[TODO: update once the second-round results are in; keep it under 250 words.]`

The same organ can be scanned at very different voxel sizes. In hierarchical phase-contrast tomography (HiP-CT), a whole kidney is imaged at about 25-50 µm per voxel and small regions of it at 6.5 µm or less. Foundation models for 3D medical images are usually adapted to new tasks with parameter-efficient fine-tuning (PEFT), and the adapter is never told the voxel size. We ask what happens when it is. We adapt a frozen SAM-Med3D encoder with low-rank adaptation (LoRA), with and without feature-wise linear modulation (FiLM) of the LoRA branch by the voxel spacing. We test on held-out, expert-labelled kidney volumes at four voxel sizes: 63 µm (another donor and beamline), 50 µm, 13 µm and 5.2 µm. In a first set of experiments, spacing conditioning helped at the training scale (Dice 0.272 vs. 0.184 over three seeds) but failed at 5.2 µm (Dice below 0.02 for every seed), while plain LoRA got worse more gently. A 2 × 2 test that changes the field of view and the spacing input separately shows that `[TODO]`. Training with random rescaling, so that the spacing input actually varies, `[TODO]`. We also show that SAM-Med3D returns a 16³ mask for a 64³ input, which is too coarse for vessels a few voxels wide, and that a small full-resolution head `[TODO]`. All models are compared with nnU-Net, vesselFM and the five winning solutions of the SenNet + HOA Kaggle challenge on exactly the same regions. Code and benchmark regions are public.

**Keywords:** vessel segmentation; foundation models; parameter-efficient fine-tuning; voxel spacing; domain shift; synchrotron tomography

---

## 1. Introduction

HiP-CT can image a whole human organ and then zoom into any part of it without cutting the organ (Walsh et al., 2021). A kidney is first scanned at about 25 µm per voxel; regions of interest are then scanned at 6.5 µm and down to about 1 µm. The Human Organ Atlas makes these scans public (Walsh et al., 2026). For the kidney, experts have also drawn the blood vessels: on whole kidneys at 50 µm (25 µm scans binned by 2) and on a zoomed region of the same kidney at 5.2 µm (Yagis et al., 2024; Jain et al., 2026).

This creates an unusual problem for segmentation models. The same vessel tree looks very different in voxels at the two scales. A small artery that is 1-2 voxels wide at 50 µm is about 10-20 voxels wide at 5.2 µm. A model trained at one scale sees something close to a new kind of image at the other.

The standard answer is to resample every image to one voxel size. nnU-Net does this because, in its authors' words, "CNNs do not natively understand voxel spacings" (Isensee et al., 2021). For HiP-CT this is wasteful. Resampling the zoomed scans to 50 µm throws away the detail they were taken for, and resampling whole organs to 5 µm makes them enormous. Another answer is to tell the network the voxel size. HyperSpace builds the weights of a U-Net from the spacing with a hypernetwork (Joutard et al., 2024). Garrido-Martínez et al. (2026) feed scan metadata to U-Nets through FiLM layers (Perez et al., 2018).

Meanwhile, 3D foundation models such as SAM-Med3D (Wang et al., 2023) are now the usual starting point for new segmentation tasks. They are adapted with PEFT, most often LoRA (Hu et al., 2021; Zhang and Liu, 2023; Chen et al., 2024; Dutt et al., 2024). None of these adapters knows the voxel size. Adding it looks like an easy improvement. This paper tests whether it is.

The question matters in practice. In the SenNet + HOA challenge on these data (Jain et al., 2026), the top teams dealt with different voxel sizes by rescaling the test data to the training resolution, or by augmentation. None of them made use of the fact that the training set contained the same kidney at two resolutions. Their models also transferred poorly to new donors: the ranking of the top 50 teams on the public test set had almost no relation to their ranking on the private one (Kendall's tau 0.025).

Our first experiments gave a clear but surprising result. Spacing-conditioned LoRA was better than plain LoRA at the training resolution and almost completely failed on the 5.2 µm volume. This paper turns that observation into a controlled study. Our contributions are:

1. **A held-out benchmark at four voxel sizes** (63.08, 50.28, 13 and 5.2 µm), built from the public data of Jain et al. (2026). It uses fixed test regions, surface- and topology-aware metrics and bootstrap confidence intervals. The five winning challenge solutions, nnU-Net and vesselFM are scored on exactly the same regions.
2. **A controlled comparison** of plain and spacing-conditioned LoRA on a frozen SAM-Med3D encoder, with several seeds, a 2 × 2 test that changes the field of view and the spacing input separately, and an analysis of the FiLM parameters in all 24 adapted layers.
3. **Two fixes suggested by this analysis.** First, random rescaling during training so that the spacing input actually changes, as in HyperSpace. Second, a small full-resolution head that removes the 4× coarser output of SAM-Med3D's decoder. `[TODO: one sentence on the effect.]`
4. **Practical advice** for adapting foundation models to images taken at several resolutions, with code and data.

---

## 2. Related work

### 2.1 Foundation models and how they are adapted

The Segment Anything Model (SAM) started a line of promptable segmentation models for medical images. MedSAM fine-tunes SAM on 1.57 million 2D image-mask pairs (Ma et al., 2024). Its authors note that it has trouble with branching vessels, because one bounding box can contain both an artery and a vein. MedSAM2 moves to 3D images and videos on top of SAM 2 and fine-tunes the whole model (Ma et al., 2025). It has the same limitation: its box prompts and the lack of explicit 3D continuity make thin, branching vessels hard. SAM-Med3D is fully 3D and was trained on 22,000 volumes and 143,000 masks (Wang et al., 2023). All its training data were resampled to 1.5 mm and cropped to 128³, and masks smaller than 1 cm³ were removed. Our voxels (50 µm and 5.2 µm) are 30 to 300 times smaller than the voxels it was trained on.

Several works adapt SAM with a small number of trainable parameters. SAMed adds LoRA to the image encoder and trains the prompt encoder and decoder (Zhang and Liu, 2023). Medical SAM Adapter inserts adapters and handles 3D data with a space-depth transpose (Wu et al., 2025). MA-SAM combines low-rank updates with 3D adapters and reaches or exceeds nnU-Net without any prompt (Chen et al., 2024). Asokan et al. (2024) find that training the whole decoder on a small dataset can damage what the model already knows, and that LoRA plus the last decoder layers works just as well. Dutt et al. (2024) compare 17 PEFT methods on medical image classification and text-to-image generation (not segmentation) and find LoRA a strong choice for vision transformers, especially with little data. Conv-LoRA adds small convolutional experts to LoRA to give SAM's encoder a local, multi-scale prior (Zhong et al., 2024), and VesselSAM combines LoRA with atrous attention to segment the aorta on 2D slices with box prompts (Iltaf et al., 2025).

Three of these papers work around the same weakness we found: SAM's output is coarse. SAMed upsamples 224² slices to 512² before the model to get a usable output resolution. MA-SAM and Asokan et al. add transposed convolutions to the decoder. DINO-Med3D adds a separate high-resolution stream to recover detail lost when the image is cut into patches (Hu et al., 2026). None of these methods uses the voxel size.

### 2.2 Vessel segmentation and HiP-CT

Yagis et al. (2024) set the first nnU-Net baseline on three HiP-CT kidneys at 50 µm. Tested on a kidney not seen in training, it reached Dice 0.86-0.95 and clDice 0.82-0.89. The main errors were collapsed large vessels and breaks in small ones. The SenNet + HOA challenge (Jain et al., 2026) used three kidneys at 50 µm and the 5.2 µm region for training. It tested on part of another kidney of the same donor (50.28 µm) and on part of a kidney from a new donor scanned on a different beamline (63.08 µm). The best solutions were 2D, 2.5D and 3D U-Nets with pseudo-labels, surface- and topology-aware losses and test-time rescaling. They reached a normalised surface Dice (NSD, tolerance 0) of up to 0.774 on the private test set, and all of them broke small vessels into pieces. After the challenge, the test labels, an extra region scanned at 6.5 µm (binned to 13 µm), and the teams' predictions were made public. Our benchmark is built on this release.

vesselFM is a foundation model made only for 3D vessels (Wittmann et al., 2025). It is trained on real, randomly generated and synthesised images and works well on new kinds of images without further training. Its training data include HiP-CT kidney at 2.5 µm. vesselFM-CT extends the idea to all vessels in CT (Wittmann et al., 2026). clDice is the standard loss and metric that checks whether vessel centrelines are kept (Shit et al., 2021).

### 2.3 Voxel size and resolution

There are three broad ways to cope with changing voxel sizes. The first is to resample everything to one size, as nnU-Net does (Isensee et al., 2021). The second is to make the model robust by showing it many resolutions during training. SynthSeg does this for brain MRI by training on synthetic images with random contrast and resolution (Billot et al., 2023). It works on scans of any resolution, but its test images are still resampled to 1 mm, and its random resolutions are all coarser than the target, not finer. A scale-equivariant U-Net builds robustness to scale into the architecture itself (Sangalli et al., 2022).

The third way is to give the network the voxel size. HyperSpace does this with a hypernetwork and trains it on batches resampled to random spacings (Joutard et al., 2024). Inside the training range it matches models trained for one fixed spacing. At the finest resolutions it tested, its accuracy also drops sharply. The authors blame the limited receptive field of the generated U-Net, and they fix it by resampling the test image to the nearest training resolution. VoxelPrompt keeps track of the voxel size inside the network so that it can work at native resolution (Hoopes et al., 2024). Consispace learns how to resample slices before segmentation (You et al., 2026). Garrido-Martínez et al. (2026) use FiLM to condition 3D abdominal segmentation on scan metadata. It helped a plain U-Net a little and did not help nnU-Net consistently.

### 2.4 Distribution shift

Changes in scanner and protocol are the best-known source of distribution shift in medical imaging (Li and Fan, 2023; Su et al., 2024). Karani et al. (2021) handle them by adapting a small normalisation network to each test image. A tenfold change in voxel size on the same specimen, with the same labelling protocol, is an extreme but clean example of such a shift.

---

## 3. Methods

### 3.1 Backbone

We use the SAM-Med3D image encoder ("vit_b_ori": ViT-B, 768 channels, 12 transformer blocks, 16³ patches, pretrained at 128³) and its mask decoder, with the released weights (Wang et al., 2023). The encoder is frozen. The decoder always needs a prompt, so we give it two opposite corners of the input cube, which marks the whole cube as the target. A 64³ input gives 4³ tokens and a 16³ mask. A 128³ input gives 8³ tokens and a 32³ mask. The mask is enlarged to the input size by trilinear interpolation.

### 3.2 Adapters

**Plain LoRA.** We add low-rank updates of rank r = 4 (α = 1) to the `qkv` and `proj` layers of all 12 blocks, 24 layers in total: h = W x + (α/r) B A x. B starts at zero, so training starts from the pretrained model.

**Spacing-conditioned LoRA.** The same, but the low-rank branch is scaled and shifted by FiLM parameters that depend on the voxel spacing s (one value per axis): h = W x + (α/r) (γ(s) ⊙ B A x + β(s)). Each layer has its own small network (3 → 16 → 2·d_out, GELU) that outputs γ − 1 and β. Its last layer starts at zero, so γ = 1 and β = 0 at the start. This adds 1.25 M parameters (1,254,912). The mask decoder is trained in both models.

The FiLM input is the spacing encoding e(s). In the first round it was the raw spacing in µm. With that input the first layer of each FiLM network is close to linear in s, so the modulation learned at 50 µm shrinks almost in proportion to the spacing: in the first-round model, the mean |γ − 1| over all 24 layers was 1.75 at 50 µm and 0.19 at 5.2 µm (results/film/first_round_spacing_seed0.json). In the second round we use e(s) = log2(s / 50 µm), which maps the training spacing to 0 and each halving of the voxel size to −1.

**Full-resolution head.** A small two-level 3D CNN (16 and 32 channels, instance normalisation) takes the input image and the enlarged coarse logits and adds a correction at full resolution. Its last layer starts at zero, so it changes nothing at the start. It has 50,721 parameters. With the head, 7.80 % (plain, 7,856,209 of 100,781,137) and 8.93 % (spacing-conditioned, 9,111,121 of 102,036,049) of all parameters are trained. It serves the same purpose as the extra upsampling layers of MA-SAM and Asokan et al. (2024) and the high-resolution stream of DINO-Med3D.

### 3.3 Training

The loss is soft Dice + 0.5 × clDice + 0.05 × a gradient penalty scaled by the inverse voxel spacing + 1.0 × weighted binary cross-entropy on the logits. The weight of the vessel class comes from the fraction of vessel voxels in each batch, capped at 50.

**First round** (Section 5.6). 64³ patches on a regular grid; patches without any vessel skipped; 20 patches per epoch; 10 epochs on a CPU and 100 epochs on a T4 GPU; AdamW, learning rate 1e-4, batch size 1; three seeds for the 100-epoch runs.

**Second round** (main results). 128³ patches, half of them centred (with random jitter) on a vessel voxel, similar to nnU-Net's foreground sampling. 12 epochs of 250 optimiser steps; each step accumulates the gradients of two patches (batch size 1, so 6,000 patches per run). AdamW (learning rate 1e-4, weight decay 1e-4), 200 linear warm-up steps then cosine decay, float16 autocast for the forward pass, loss in float32, gradient clipping at norm 1. To keep reading the TIFF stacks fast, each data-loader worker reads a block of consecutive slices (160 slices, or 288 with spacing augmentation) and cuts 32 patches from it before reading the next block; blocks are drawn from the volumes in proportion to their size.

**Validation and model selection.** The last 20 % of the slices of kidney_1_dense are never used for training. Six fixed validation ROIs are cut from them (benchmark/rois.py, --validation). Every second epoch the model is run on them with the same sliding-window inference as on the test ROIs, and the weights with the best mean validation Dice are kept. No test ROI is used for any choice. Each configuration is trained with seeds 0, 1 and 2.

**Spacing augmentation.** For each patch we draw a scale k between 0.25 and 2 (uniform on a log scale), cut a cube of round(128·k) voxels, resize it to 128³ (trilinear for the image; trilinear then threshold 0.5 for the label) and give the model a spacing of k times the true one. For the 50 µm training data this covers 12.5 to 100 µm, so 13 µm is inside the augmented range and 5.2 µm is still outside it. This follows HyperSpace. Without it, the spacing input only ever takes the values 50.00 and 50.16 µm during training, so the FiLM network never sees it change.

### 3.4 Separating the spacing input from the field of view

When the voxel size shrinks tenfold and the input size stays the same, two things change together: the spacing value given to the model, and the amount of tissue in each input window. A 64³ window covers 3.2 mm at 50 µm and only 0.33 mm at 5.2 µm. We separate the two with a 2 × 2 design at test time:

| | spacing input = true value | spacing input = 50 µm |
|---|---|---|
| **native voxel grid** (small field of view) | native | tell50 |
| **resampled to 50 µm** (training field of view) | fov50 | (identical to fov50) |

In the fov50 setting, each region is resampled to 50 µm, segmented, and the logits are resampled back to the original grid. Every setting is therefore scored against the same original labels. Plain LoRA ignores the spacing input, so for it only native and fov50 differ.

---

## 4. Data and evaluation

### 4.1 Data

All data come from the SenNet + HOA challenge (Jain et al., 2026) and the later release on Zenodo (doi:10.5281/zenodo.19685381, CC-BY 4.0).

| Volume | Donor | Voxel (µm) | Beamline | Use |
|---|---|---|---|---|
| kidney_1_dense | LADAF-2021-17, right | 50.00 | BM05 | training |
| kidney_2 | S-20-28 | 50.00 | BM05 | training (about 65 % of vessels labelled) |
| kidney_3_dense / sparse | LADAF-2020-27 | 50.16 | BM05 | training (partly labelled) |
| kidney_5 | LADAF-2021-17, left | 50.28 | BM05 | test: other kidney, same donor |
| kidney_6 | LADAF-2022-13 | 63.08 | BM18 | test: new donor and beamline |
| kidney_additional_voi_13um | LADAF-2021-17, right | 13 (6.5, binned) | BM05 | test: 4× finer |
| kidney_1_voi | LADAF-2021-17, right | 5.2 | BM05 | test: 10× finer |

The 50 µm volumes are 25 µm scans binned by 2 (Yagis et al., 2024). The effective resolution of HiP-CT is coarser than its voxel size (about 72 µm at 25 µm per voxel and 18 µm at 6.5 µm per voxel; Walsh et al., 2021). In the Zenodo release, only slices 0-2225 of the 13 µm volume contain image data; the other 3,729 image files are empty, so we use those 2,226 slices. Jain et al. (2026) point out that the 13 µm region was scanned with a different imaging configuration (energy, filters, detector, propagation distance), so its contrast and intensities differ from both the 50 µm and the 5.2 µm data; it is not simply a binned or upsampled copy of either. To avoid leakage, no test volume is used for training or for choosing settings. We also do not train on the extra Human Organ Atlas scans of donor LADAF-2021-17: the 6.5 µm scan is the source of the 13 µm test volume, and the 2.6 µm scan is very likely the source of the 5.2 µm one.

### 4.2 Test regions

Every method is scored on the same fixed test regions (ROIs), not on random patches. In each test volume we pick ROI centres among vessel voxels with a fixed random seed, without overlap. Each ROI is a cube of 9.6 mm per side, limited to 160-640 voxels: 160³ at 63.08 µm, 191³ at 50.28 µm and 640³ at 13 µm and 5.2 µm (8.3 mm and 3.3 mm). We use 8 ROIs in kidney_5 and kidney_6 and 6 in each zoomed volume, 28 test ROIs in all. The six validation ROIs (Section 3.3) are 192³ cubes from the held-out slices of kidney_1_dense. The ROIs and their positions are released.

### 4.3 Metrics and statistics

For each ROI we compute Dice; clDice on the skeletons of the binary masks (Shit et al., 2021); NSD with a tolerance of 0 and 1 voxel (the challenge used 0); the 95th percentile Hausdorff distance and the average symmetric surface distance (MONAI); and the number of connected components and branch points of the skeletons.

Our NSD is computed per ROI with MONAI's voxel-based implementation, so its values differ from the challenge leaderboard. As a check, on the kidney_6 ROIs it ranks the five winning teams in exactly the leaderboard order (0.633, 0.629, 0.601, 0.591 and 0.574 here, against 0.774, 0.755, 0.727, 0.712 and 0.691 on the private leaderboard; Jain et al., 2026). Team 2, the only team that removed small disconnected pieces, is also the only one whose number of connected components is close to the labels (median 25 per ROI against 28 in the labels; 55-267 for the other teams). This matches the observations of Jain et al.

We report the mean over ROIs with a 95 % bootstrap confidence interval (10,000 resamples). To compare two methods on the same ROIs we give the bootstrap interval of the mean difference and a Wilcoxon signed-rank test. Our models are trained with `[TODO: 3]` seeds; `[TODO: decide whether to average seeds per ROI or treat ROI × seed as the unit]`.

### 4.4 Baselines

- **Kaggle top 5.** The released test predictions of the five winning teams for kidney_5, kidney_6 and the 13 µm volume, cut to our ROIs. These teams trained on the 5.2 µm volume, so they are not tested on it.
- **nnU-Net** (3d_fullres, `nnUNetTrainer_100epochs`, all training data in one fold), trained on 12 vessel-centred 256³ crops per training volume, because nnU-Net's preprocessed copies of the full volumes do not fit on the available disk. We run it twice: with the true voxel size, so that nnU-Net resamples each ROI to its ~50 µm training spacing, and with every ROI labelled as 50 µm, so that it works on the original grid.
- **vesselFM**, used as released, with and without its option to invert the image intensities (HiP-CT vessels are darker than the tissue around them). Its training data include HiP-CT kidney, so this is not a strict zero-shot test.
- **SAM-Med3D** without any fine-tuning, with the same whole-cube prompt.

### 4.5 Implementation

PyTorch and MONAI, one NVIDIA T4 (16 GB) on Kaggle. Sliding-window inference with 50 % overlap and Gaussian weighting; each window is normalised on its own, as in training. `[TODO: training time per run, inference time per ROI, total GPU hours.]`

---

## 5. Results

### 5.1 Held-out benchmark

Table 1. Dice [95 % CI] / clDice / NSD (t = 0) / NSD (t = 1), mean over ROIs.

| Method | kidney_5 (50.28 µm) | kidney_6 (63.08 µm) | 13 µm | kidney_1_voi (5.2 µm) |
|---|---|---|---|---|
| Kaggle team 1 | 0.940 [0.93, 0.95] / 0.875 / 0.807 / 0.971 | 0.832 [0.778, 0.878] / 0.811 / 0.633 / 0.923 | 0.838 [0.72, 0.93] / 0.677 / 0.549 / 0.800 | n/a (trained on it) |
| Kaggle team 2 | 0.012 [0.01, 0.02] / 0.014 / 0.012 / 0.053 | 0.705 [0.552, 0.830] / 0.798 / 0.629 / 0.819 | 0.003 [0.00, 0.01] / 0.006 / 0.004 / 0.008 | n/a |
| Kaggle team 3 | 0.936 [0.92, 0.95] / 0.854 / 0.797 / 0.962 | 0.819 [0.768, 0.868] / 0.774 / 0.601 / 0.894 | 0.844 [0.73, 0.94] / 0.703 / 0.550 / 0.799 | n/a |
| Kaggle team 4 | 0.941 [0.93, 0.95] / 0.885 / 0.797 / 0.967 | 0.818 [0.765, 0.867] / 0.768 / 0.591 / 0.884 | 0.812 [0.70, 0.90] / 0.714 / 0.490 / 0.746 | n/a |
| Kaggle team 5 | 0.937 [0.93, 0.95] / 0.874 / 0.780 / 0.969 | 0.810 [0.758, 0.860] / 0.753 / 0.574 / 0.880 | 0.747 [0.60, 0.88] / 0.513 / 0.259 / 0.551 | n/a |
| nnU-Net (resampled) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| nnU-Net (native grid) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| vesselFM | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SAM-Med3D, no fine-tuning | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Plain LoRA, first round (64³) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Spacing LoRA, first round (64³) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Plain LoRA, second round | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Spacing LoRA, second round | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Spacing LoRA + spacing augmentation | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Spacing LoRA + augmentation + full-resolution head | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

(Kaggle team rows: results/scores/kaggle_team*.json, built with `python -m benchmark.paper_tables`.)

Team 2's near-zero scores on kidney_5 and the 13 µm region are not an import error. Its masks are aligned with the labels, but they mark large parts of the tissue as vessel. Jain et al. (2026) report the same failure: Team 2 scored 0.04 NSD on the public test set (kidney 5) and won second place only through the private test set (kidney 6), and they leave Team 2 and Team 3 out of their qualitative analysis for this reason.

### 5.2 Spacing input vs. field of view

`[TODO: native / tell50 / fov50 for each LoRA model on the two zoomed volumes, with paired differences and CIs. Write the interpretation only after seeing the numbers. First-round result to compare with (docs/results_log.txt 3.5): on 40 field-of-view-matched patches of kidney_1_voi, spacing-conditioned LoRA went from Dice 0.0058 (native) to 0.2215, and plain LoRA from 0.1672 to 0.2895; on a separate draw the matched gap (0.3681 vs. 0.2618) gave Mann-Whitney U p = 0.0216.]`

### 5.3 Effect of spacing augmentation

`[TODO: with vs. without --fov-scale 0.25 2.0 for both LoRA variants at the four test spacings; FiLM γ per layer at 5.2, 13, 50 and 63 µm after training with augmentation, compared with Section 5.5.]`

### 5.4 Ablations

`[TODO: 64³ vs. 128³ patches; with vs. without the full-resolution head; foreground sampling on/off; clDice on/off. First-round clDice result (docs/results_log.txt 3.7): without clDice, training-volume Dice fell from 0.3145 to 0.1898 and the 5.2 µm Dice stayed very low (0.0058 vs. 0.0015).]`

### 5.5 What the FiLM layers learn

In the first-round spacing-conditioned model (100 epochs), γ moved far from 1 at the training spacing in all 24 adapted layers. For example, in layer 19 its mean was 6.52 and its maximum 21.6 at 50 µm, and in layer 2 its maximum was 19.7. At 5.2 µm, γ stayed close to 1 in the same layers (layer 19: mean 1.59, maximum 3.23; layer 2: maximum 3.06) (first_round/results/film_layer_analysis.json; docs/results_log.txt 3.6). So the model came to depend on a strong modulation that only exists at the training spacing. At 5.2 µm the adapter falls back to something like plain LoRA, but not quite, because its LoRA weights were trained together with that strong modulation. With the log-spacing input of the second round, the picture changes (results/film/*.json, analyze_film_sweep.py). Without spacing augmentation (s_base, seed 0), the mean |γ − 1| over the 24 layers is small and almost flat in the spacing: 0.067 at 50 µm, 0.047 at 13 µm, 0.061 at 5.2 µm and between 0.047 and 0.108 anywhere from 1 to 100 µm. The FiLM networks have learned little more than a constant, which is expected because their input was 0 or 0.005 for every training patch. With spacing augmentation (s_aug, seeds 0 and 1), the modulation is smallest around the training spacing (0.046-0.051 between 50 and 63 µm) and grows steadily as the voxels get finer: 0.051 and 0.048 at 50 µm, 0.159 and 0.155 at 13 µm, 0.245 and 0.238 at 5.2 µm. The two seeds agree to within 0.01 at every spacing, so this is a stable, learned dependence on the spacing, not noise. It is the opposite of the first round, where the modulation was large at 50 µm (1.75) and vanished at 5.2 µm (0.19). Whether the extra modulation at fine spacings actually helps segmentation there is tested by the tell50 setting in Section 5.2. `[TODO: add s_base seeds 1-2 and s_aug seed 2 when trained.]`

### 5.6 First-round study

The first experiments were scored on random 64³ patches from the training volumes and from kidney_1_voi, leaving out patches without vessels. The training-volume numbers are therefore not held-out results. We report them only to show how the question arose; Table 1 replaces them.

Over three seeds at 100 epochs, spacing-conditioned LoRA reached Dice 0.2718 ± 0.0464 on training-volume patches against 0.1838 ± 0.0644 for plain LoRA. On kidney_1_voi it reached 0.0102 ± 0.0079 against 0.0990 ± 0.0591 (per seed: 0.0058, 0.0193, 0.0055 vs. 0.1672, 0.0620, 0.0679; docs/results_log.txt 3.11). Only 5 of the 40 kidney_1_voi patches contained any vessel in the main run. On kidney_1_voi, the skeletons of spacing-conditioned LoRA had a median of 11 connected pieces and 15.5 branch points, against 1 and 0.5 in the labels and 6 and 1.0 for plain LoRA (docs/results_log.txt 3.10).

### 5.7 Structure of the errors

`[TODO: connected components and branch points on the benchmark ROIs for all methods; one example ROI per test volume.]`

---

## 6. Discussion

`[TODO: finish after the results. Each point below must be backed by a number from Section 5.]`

1. **The spacing input did not vary during training.** In the first round the model saw essentially one spacing. A FiLM network trained like this has no reason to do anything sensible at other spacings, and Section 5.5 shows it does not. Random rescaling is the obvious fix and is what HyperSpace does; SynthSeg gets its robustness the same way, by randomising resolution in training. Section 5.3 tests whether this is enough.
2. **Field of view.** At 5.2 µm a 64³ window covers 0.33 mm, less than the width of many of the vessels it should find. HyperSpace reports the same kind of failure at its finest resolutions and fixes it by resampling to the nearest training resolution. The Kaggle winners did the same by rescaling the test data, and nnU-Net does it by design. Our fov50 setting applies this idea to adapted foundation models.
3. **Scale of the pretrained encoder.** SAM-Med3D was trained on images resampled to 1.5 mm and never saw structures smaller than 1 cm³. A rank-4 update may be too small to shift its features to vessels at 50 µm, let alone 5.2 µm. MedSAM2's authors argue that fine-tuning the whole model works better than partial fine-tuning; we did not test this.
4. **Output resolution.** A 16³ output for a 64³ input limits how well thin vessels can be segmented, whatever the adapter. SAMed, MA-SAM and Asokan et al. all work around the same limit. This affects both LoRA variants equally and explains part of the gap to nnU-Net and the Kaggle solutions.
5. **Training the whole decoder.** We train the whole mask decoder. Asokan et al. (2024) found that this can hurt on small datasets. `[TODO: test LoRA + last decoder layers only, if GPU time allows.]`
6. **Advice.** `[TODO: e.g. resample to the training spacing when the test spacing is outside the training range; if you condition on spacing, train over the whole range you care about.]`

### Limitations

One organ and one imaging technique. Few labelled volumes: four for training and four for testing, and two of the test volumes come from a donor also used in training. Two training volumes are only partly labelled. Our nnU-Net is trained on crops rather than whole volumes and for `[TODO]` epochs, which is below its usual budget. The test ROIs cover only part of each test volume. `[TODO: number of seeds.]` HiP-CT is an ex-vivo research technique; we make no clinical claim.

---

## 7. Conclusion

`[TODO: one paragraph after the results.]`

---

## CRediT authorship contribution statement

**Rabeya Noor:** Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation, Visualization, Writing – original draft, Writing – review & editing.

## Declaration of competing interest

`[TODO: e.g. "The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."]`

## Declaration of generative AI and AI-assisted technologies in the writing process

`[TODO: Elsevier requires this section. State truthfully which tools were used and for what, and that you checked the content and take full responsibility for it. Generative AI may not be used to create figures or the graphical abstract.]`

## Funding

`[TODO: e.g. "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."]`

## Data availability

Training data: SenNet + HOA Kaggle challenge page. Test labels, the 13 µm volume and the challenge predictions: Zenodo, doi:10.5281/zenodo.19685381. Raw scans: Human Organ Atlas portal (https://human-organ-atlas.esrf.eu). Benchmark ROIs and positions: `[TODO: DOI]`.

## Code availability

`[TODO: public repository and DOI.]`

## Acknowledgements

`[TODO: thank the Human Organ Atlas and SenNet + HOA teams for releasing the data and labels; Kaggle for GPU time.]`

---

## References

Author-year style (elsarticle-harv). Details checked on PubMed or on the paper itself; `[check]` marks what is still missing. Convert with a reference manager before submission.

- Asokan, M., Benjamin, J.G., Yaqub, M., Nandakumar, K., 2024. A federated learning-friendly approach for parameter-efficient fine-tuning of SAM in 3D segmentation. arXiv:2407.21739. `[check published venue]`
- Billot, B., Greve, D.N., Puonti, O., Thielscher, A., Van Leemput, K., Fischl, B., Dalca, A.V., Iglesias, J.E., 2023. SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. Medical Image Analysis 86, 102789.
- Chen, C., Miao, J., Wu, D., Zhong, A., Yan, Z., Kim, S., Hu, J., Liu, Z., Sun, L., Li, X., Liu, T., Heng, P.A., Li, Q., 2024. MA-SAM: Modality-agnostic SAM adaptation for 3D medical image segmentation. Medical Image Analysis 98, 103310.
- Dutt, R., Ericsson, L., Sanchez, P., Tsaftaris, S.A., Hospedales, T., 2024. Parameter-efficient fine-tuning for medical image analysis: the missed opportunity. Medical Imaging with Deep Learning (MIDL), PMLR 250.
- Garrido-Martínez, J.F., Garrido-López, J.M., Zapata-Pérez, J.F., Martínez-Alajarín, J., 2026. Metadata conditioning via feature-wise linear modulation for multi-domain 3D abdominal segmentation: a comparative study between U-Net and nnU-Net. Sensors. `[check volume/article number]`
- Hoopes, A., Dey, N., Butoi, V.I., Guttag, J.V., Dalca, A.V., 2024. VoxelPrompt: a vision agent for end-to-end medical image analysis. arXiv:2410.08397.
- Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2021. LoRA: Low-rank adaptation of large language models. arXiv:2106.09685. `[check: ICLR 2022 version]`
- Hu, H., Ma, X., Liu, S., Zhang, L., Xie, X., Zhou, X., Hou, Z.-G., 2026. DINO-Med3D: bridging dimension and domain gaps in volumetric segmentation via progressive adaptation. MICCAI 2026. arXiv:2606.18886.
- Iltaf, A., Ahmed, R.M., Zhang, Z., Li, B., Zhou, S., 2025. VesselSAM: leveraging SAM for aortic vessel segmentation with AtrousLoRA. arXiv:2502.18185.
- Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H., 2021. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203-211.
- Jain, Y., Walsh, C.L., Yagis, E., Aslani, S., Nandanwar, S., Zhou, Y., Ha, J., Gustilo, K.S., Brunet, J., Rahmani, S., Tafforeau, P., Bellier, A., Weber, G.M., Lee, P.D., Börner, K., 2026. Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys. Nature Communications 17. `[check article number]`
- Joutard, S., Pietsch, M., Prevost, R., 2024. HyperSpace: hypernetworks for spacing-adaptive image segmentation. MICCAI 2024. arXiv:2407.03681.
- Karani, N., Erdil, E., Chaitanya, K., Konukoglu, E., 2021. Test-time adaptable neural networks for robust medical image segmentation. Medical Image Analysis 68, 101907.
- Li, Y., Fan, Y., 2023. Medical image segmentation with domain adaptation: a survey. arXiv:2311.01702.
- Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B., 2024. Segment anything in medical images. Nature Communications 15, 654.
- Ma, J., Yang, Z., Kim, S., Chen, B., Baharoon, M., Fallahpour, A., Asakereh, R., Lyu, H., Wang, B., 2025. MedSAM2: segment anything in 3D medical images and videos. arXiv:2504.03600.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A., 2018. FiLM: visual reasoning with a general conditioning layer. AAAI 2018.
- Sangalli, M., Blusseau, S., Velasco-Forero, S., Angulo, J., 2022. Scale equivariant U-Net. arXiv:2210.04508. `[check published venue]`
- Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J.P.W., Bauer, U., Menze, B.H., 2021. clDice - a novel topology-preserving loss function for tubular structure segmentation. CVPR 2021.
- Su, Z., Guo, J., Yang, X., Wang, Q., Coenen, F., Hussain, A., Huang, K., 2024. Navigating distribution shifts in medical image analysis: a survey. arXiv:2411.05824.
- Walsh, C.L., Tafforeau, P., Wagner, W.L., Jafree, D.J., Bellier, A., Werlein, C., Kühnel, M.P., Boller, E., Walker-Samuel, S., Robertus, J.L., Long, D.A., Jacob, J., Marussi, S., Brown, E., Holroyd, N., Jonigk, D.D., Ackermann, M., Lee, P.D., 2021. Imaging intact human organs with local resolution of cellular structures using hierarchical phase-contrast tomography. Nature Methods 18, 1532-1541.
- Walsh, C.L., Brunet, J., Stansby, D., et al., 2026. The Human Organ Atlas. Science Advances 12, eadz2240. `[read before citing; full author list on PubMed]`
- Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J., Qiao, Y., 2023. SAM-Med3D: towards general-purpose segmentation models for volumetric medical images. arXiv:2310.15161.
- Wittmann, B., Wattenberg, Y., Amiranashvili, T., Shit, S., Menze, B., 2025. vesselFM: a foundation model for universal 3D blood vessel segmentation. CVPR 2025, 20874-20884.
- Wittmann, B., Prabhakar, C., Shit, S., Menze, B., 2026. vesselFM-CT: segmenting all blood vessels in CT images for system-level cardiovascular analysis. arXiv:2606.09400.
- Wu, J., Wang, Z., Hong, M., Ji, W., Fu, H., Xu, Y., Xu, M., Jin, Y., 2025. Medical SAM adapter: adapting segment anything model for medical image segmentation. Medical Image Analysis 102, 103547.
- Yagis, E., Aslani, S., Jain, Y., Zhou, Y., Rahmani, S., Brunet, J., Bellier, A., Werlein, C., Ackermann, M., Jonigk, D., Tafforeau, P., Lee, P.D., Walsh, C.L., 2024. Deep learning for 3D vascular segmentation in hierarchical phase contrast tomography: a case study on kidney. Scientific Reports 14, 27258.
- Yan, N., Tang, L., Tao, Y., Yao, J., Guo, Z., Liu, M., Wang, J., 2025. Multi-view hybrid encoder U-Net for 3D renal vascular medical image segmentation. Scientific Reports 15, 44887.
- You, X., Yang, R., Zhang, M., Zhang, H., Li, H., Yu, Y., Yang, J., Navab, N., Gu, Y., 2026. Towards voxel spacing consistency for medical image segmentation. arXiv:2606.31839.
- Zhang, K., Liu, D., 2023. Customized segment anything model for medical image segmentation. arXiv:2304.13785.
- Zhong, Z., Tang, Z., He, T., Fang, H., Yuan, C., 2024. Convolution meets LoRA: parameter efficient finetuning for segment anything model. ICLR 2024. `[read before citing]`

---

## Figures (plan)

1. Graphical abstract (required by MedIA; must not be made with generative AI): the same kidney at 50 µm and 5.2 µm, the adapter, and the four test voxel sizes. `[TODO]`
2. Method: frozen SAM-Med3D, plain LoRA vs. FiLM-modulated LoRA, full-resolution head. `[TODO]`
3. Data: one ROI per test volume with its labels, drawn at the same physical scale. `[TODO]`
4. Main results: Dice and NSD per test voxel size for all methods. `[TODO]`
5. The 2 × 2 spacing / field-of-view test. `[TODO]`
6. FiLM γ per layer with and without spacing augmentation. `[TODO]`
7. Example segmentations and skeletons. `[TODO]`
