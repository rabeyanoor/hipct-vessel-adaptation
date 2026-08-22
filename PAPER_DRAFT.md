# Spacing-Conditioned LoRA for 3D Vessel Segmentation: An In-Distribution Gain and an Out-of-Distribution Collapse

**Status: draft.** All numbers here come from `test.sh` (the running experiment log for this project) and `baki.sh`. Places marked `[TODO]` still need to be filled in.

---

## Abstract

Parameter-efficient fine-tuning (PEFT) is the usual way to adapt frozen 3D medical foundation models to new anatomy. Voxel spacing, however, can differ by orders of magnitude between acquisition protocols, and it is rarely passed to the adapter as an input. We propose Spacing-Conditioned LoRA, a low-rank adaptation of SAM-Med3D's frozen image encoder where the LoRA branch is modulated by a FiLM layer conditioned on physical voxel spacing. We compare it with a Plain LoRA baseline (same architecture, no spacing input) on expert-annotated kidney vasculature (SenNet+HOA). Spacing-Conditioned LoRA is better in-distribution in 2 of 3 independent 100-epoch training runs (mean Dice 0.272 vs. 0.184, n=3 seeds), but it collapses when evaluated at an unseen voxel spacing (5.2 μm vs. ~50 μm training data): OOD Dice is below 0.02 in all 3 runs, while Plain LoRA degrades gracefully and beats it OOD in all 3 runs. We look at this collapse in three separate ways (a field-of-view confound test, a per-layer FiLM activation analysis and a vascular-graph comparison). We find that it is not caused by too little training (it appears at both 10 and 100 epochs), that it is largely explained by field-of-view mismatch rather than the spacing signal itself, and that structurally it shows up as fragmented, over-branched predictions rather than a smaller or weaker version of the true vessel tree. We present this as a reproducible warning about a design that seems natural (giving an adapter the physical metadata), not as a fix, and we include the full protocol to reproduce it.

---

## 1. Introduction

Three-dimensional vessel segmentation in high-resolution computed tomography (HiP-CT) is important for quantitative analysis of vascular morphology and tissue structure. However, vessels can look quite different depending on the physical voxel spacing and imaging resolution. A model trained at one scale may therefore face a large distribution shift on data acquired at another resolution. This is especially relevant for high-resolution ex-vivo imaging, where datasets can have very different physical sampling while showing similar anatomy.

Recent 3D medical foundation models are a promising starting point, since they can transfer representations learned from large medical imaging datasets to downstream segmentation tasks. Fully fine-tuning such models, though, is computationally expensive and may need a lot of labelled data. Parameter-efficient fine-tuning (PEFT) adapts only a small part of the model and keeps most of the pretrained backbone frozen. Low-rank adaptation (LoRA) is especially attractive here because it adds task-specific adaptation with a small number of trainable parameters.

A limitation of standard PEFT methods is that voxel spacing is usually not used as a conditioning variable. As a result, two volumes with similar anatomy but very different physical resolutions go through the same adaptation, even though their image scale is different. Adding spacing to the adaptation step is one way to give the model this acquisition information.

In this work we study spacing-aware parameter-efficient adaptation of a frozen 3D medical foundation model for vessel segmentation in multi-resolution HiP-CT data. We introduce a spacing-conditioned LoRA that modulates the low-rank pathway using voxel spacing. We evaluate it against zero-shot and standard PEFT baselines on expert-annotated kidney vessel data. Besides in-distribution evaluation, we run a resolution-shift experiment: models trained at about 50 μm spacing are tested on a held-out 5.2 μm scan of the same specimen.

We do not assume that better in-distribution performance means robustness to resolution changes; we test it. Our experiments show that spacing-conditioned adaptation can improve in-distribution segmentation but degrade severely under the unseen resolution. We therefore add topology-based analysis and targeted experiments on possible causes of the out-of-distribution behaviour, which gives a fuller picture of both the benefits and the limits of spacing-aware PEFT for multi-resolution 3D vessel segmentation.

So the main contribution is not a method we recommend using as it is. It is a controlled comparison that exposes a reproducible failure in a design that looks reasonable on paper: giving a PEFT adapter the physical spacing as a conditioning input. We report the failure in the same detail we would report a positive result, with three seeds and three different analyses of *why* it happens.

---

## 2. Related Work

All references below were checked against the original source (title, authors and venue). Two method names that came up while searching ("TopoLoRA-SAM", "MorVess") could not be matched to citable publications and are left out. vesselFM (Wittmann et al., CVPR 2025) and HyperSpace (see below) are real papers; vesselFM is discussed in `PAPER_MEDIA.md`.

**Segmentation backbone and baseline.**
Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203-211. The standard self-configuring segmentation baseline. We do not compare against it here (see Limitations), but any PEFT-based approach should eventually be measured against it.

Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J., & Qiao, Y. (2023). SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images. arXiv:2310.15161. The frozen 3D backbone we adapt.

**Parameter-efficient fine-tuning of segmentation foundation models.**
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. The low-rank mechanism that both Plain LoRA and Spacing-Conditioned LoRA are based on.

Zhong, Z., Tang, Z., He, T., Fang, H., & Yuan, C. (2024). Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model. *ICLR 2024*. Adds a convolutional inductive bias to LoRA in SAM's encoder. It starts from the same LoRA-on-SAM setup as we do but addresses a different problem (image-domain bias, not acquisition metadata).

Iltaf, A., Ahmed, R. M., Zhang, Z., Li, B., & Zhou, S. (2025). VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with AtrousLoRA. arXiv:2502.18185. The closest prior work in both domain and mechanism: SAM adapted with a LoRA variant for vessel segmentation. It targets aortic (not renal/HiP-CT) vessels and does not use voxel spacing, which is the gap our comparison looks at.

**Spacing- and metadata-conditioned segmentation.**
Joutard, S., Pietsch, M., & Prevost, R. (2024). HyperSpace: Hypernetworks for spacing-adaptive image segmentation. *MICCAI 2024*. Uses a hypernetwork conditioned on voxel spacing so one segmentation model can work natively at different resolutions. The idea is the same as ours (spacing as a conditioning input), but it generates the full model weights instead of modulating a LoRA branch with FiLM, and it is trained with every batch resampled to a random spacing. Its accuracy also drops sharply at the highest resolutions it was tested on, which the authors attribute to the limited receptive field of the generated U-Net, and resampling to the closest training resolution at test time fixes it. This matches our field-of-view result (Section 4.5). A direct comparison with HyperSpace on our SenNet+HOA data would be a natural next step; we have not done it yet.

Garrido-Martínez, J. F., Garrido-López, J. M., Zapata-Pérez, J. F., & Martínez-Alajarín, J. (2026). Metadata Conditioning via Feature-Wise Linear Modulation for Multi-Domain 3D Abdominal Segmentation: A Comparative Study Between U-Net and nnU-Net. *Sensors*. The mechanism most similar to our Spacing-Conditioned LoRA: FiLM conditioning on case-level metadata (acquisition sequence and intensity descriptors, not physical spacing) for 3D abdominal segmentation. They find that FiLM conditioning helps a plain U-Net a little but does *not* consistently help the already-optimized nnU-Net. This is a related warning about FiLM conditioning and is worth discussing together with our OOD result.

Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*. The general conditioning mechanism that both our work and Garrido-Martínez et al. (2026) apply to medical segmentation.

**Topological/structural loss.**
Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J. P. W., Bauer, U., & Menze, B. H. (2021). clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR 2021*.

**Data.**
Jain, Y., Walsh, C. L., Yagis, E., Aslani, S., Nandanwar, S., Zhou, Y., Ha, J., Gustilo, K. S., Brunet, J., Rahmani, S., Tafforeau, P., Bellier, A., Weber, G. M., Lee, P. D., & Börner, K. (2026). Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys. *Nature Communications*, 17, 8273. https://doi.org/10.1038/s41467-026-74050-8

`[TODO: this covers the mechanism (LoRA on SAM, FiLM conditioning, spacing-adaptive segmentation) and the reference baseline (nnU-Net). Still to add: broader domain-shift / domain-generalization work in medical segmentation (the 2024 survey "Navigating Distribution Shifts in Medical Image Analysis", arXiv:2411.05824, looks like a good starting point but I have only seen the abstract so far, check it before citing); other SAM-for-3D-medical variants; and, if the wider-spacing experiment (baki.sh section 3) goes ahead, papers on training across several native resolutions.]`

---

## 3. Methods

### 3.1 Data

We use expert-annotated kidney vasculature from the SenNet+HOA "blood-vessel-segmentation" dataset. Voxel spacing is not stored in the TIFF files or the dataset metadata, so we took it from Table 1 of the dataset paper (Jain et al., 2026). Every spacing value used in the project is documented in `utils/kidney_metadata_verification.py`. Training used `kidney_1_dense`, `kidney_2`, `kidney_3_dense` and `kidney_3_sparse` (all ~50 μm). `kidney_1_voi`, the same specimen as `kidney_1_dense` (LADAF-2021-17-Right) scanned at 5.2 μm, was left out of training completely and used as the out-of-distribution resolution test set. This way the shift is only in resolution, with the same organ and the same annotation protocol, not a shift between datasets.

### 3.2 Architectures

All three models freeze the SAM-Med3D image encoder and differ only in the trainable adaptation on top of it (the `mask_decoder` is trainable in all three):

- **PhysicsAwareAdapter**: a trainable bottleneck adapter (down-project / GELU / up-project, reduction=4) on the frozen encoder's output *tokens*, gated multiplicatively by a spacing embedding (`spacing_fc`, a 2-layer MLP on the 3-value spacing vector) before the bottleneck. So this model is **also spacing-conditioned**, just through gating after the encoder instead of FiLM inside it (7.65% of parameters trainable).
- **Plain LoRA** (r=4): low-rank updates in the encoder's attention `qkv`/`proj` layers, zero-initialized so LoRA starts as a no-op. It is the only one of the three with **no spacing input anywhere in the forward pass** (7.75% trainable).
- **Spacing-Conditioned LoRA** (r=4): the same LoRA injection points as Plain LoRA, plus a FiLM layer (Perez et al., 2018) conditioned on the patch's voxel spacing. In `models/spacing_lora.py`, `spacing_film` is a small MLP that produces per-layer (γ, β) from the 3-value spacing vector; these are applied to the LoRA branch output (not the frozen base path) in all 24 injected attention layers. It is zero-initialized so γ starts at 1 and β at 0, i.e. identity modulation (8.88% trainable). The FiLM networks add ~1.25M parameters over Plain LoRA. All three trainable budgets are of the same order of magnitude, so none of the results below can be explained by one model just having more capacity.

All three use the same automatic prompt. SAM-Med3D's mask decoder always needs a prompt and has no prompt-free mode, so every forward pass uses two corner points that act as a bounding box over the whole patch. There is no interactive prompting during training or evaluation.

### 3.3 Training protocol

We train on 64³ voxel patches with AdamW. The loss (`losses/combined_loss.py`, `HiPVesselLoss`, default α=0.5/β=0.05/γ=1.0) has four terms: soft Dice loss; the clDice topology term (Shit et al., 2021, weight α); an **anisotropic gradient penalty** weighted by inverse voxel spacing per axis (weight β), specifically `mean((∂pred/∂z / spacing_z)² + (∂pred/∂y / spacing_y)² + (∂pred/∂x / spacing_x)²)`, which uses the same spacing metadata as the spacing-conditioned models; and class-weighted BCE on the logits (weight γ), where the positive-class weight is computed per batch from the foreground fraction and clamped to 50 (foreground is only ~1-5% of voxels). Because of the anisotropic penalty, **the training loss uses spacing for all three architectures**, including Plain LoRA. The comparison is therefore about whether spacing enters the *model's forward pass*, not whether it appears anywhere in training. We report two training setups: a first 10-epoch run on CPU (the only local hardware available), and a controlled 100-epoch retrain on a Kaggle T4 GPU, done to check whether the 10-epoch OOD result was just under-training.

### 3.4 Evaluation protocol

We use 100 random 64³ patches per in-distribution run and 40 per OOD run (from `kidney_1_voi`). Note that the in-distribution patches are drawn from the same four kidneys used for training, so the in-distribution numbers are not held-out test results; the held-out benchmark in `PAPER_MEDIA.md` replaces them. Patches with no annotated foreground are left out of Dice/IoU/clDice/HD95/ASSD, since these metrics are undefined or misleadingly high on an empty target; we report how many were excluded. Metrics: Dice, IoU, clDice, HD95, ASSD (MONAI implementations).

---

## 4. Results

### 4.1 In-distribution comparison, single run (10 epochs, CPU)

| Model | Trainable | Dice | IoU | clDice | HD95 (vox) | ASSD (vox) |
|---|---|---|---|---|---|---|
| Zero-shot SAM-Med3D | 0% | 0.088 | 0.052 | 0.076 | 48.1 | 25.2 |
| PhysicsAwareAdapter | 7.65% | 0.157 | 0.094 | 0.139 | 36.2 | inf* |
| Plain LoRA | 7.75% | 0.067 | 0.038 | 0.057 | 45.7 | 24.8 |
| Spacing-Conditioned LoRA | 8.88% | **0.206** | **0.131** | **0.261** | **23.4** | inf* |

At 10 epochs Spacing-Conditioned LoRA is best on every metric, including against its most direct ablation, Plain LoRA (same architecture, only the spacing modulation is missing). Plain LoRA is even worse than the untrained zero-shot baseline at 10 epochs. All three trained runs used the same loss weights and learning rate without re-tuning, which is a confound; the hyperparameter sweep (Section 4.10) looks at this.

*ASSD=inf appeared for PhysicsAwareAdapter and Spacing-Conditioned LoRA. We traced it (Section 4.9) to a rare event (~1% of non-empty patches, depending on which patches are drawn) where the model predicts nothing at all on a patch with a very small annotated target. It is not a general property of either checkpoint.

### 4.2 Resolution-OOD test (main finding)

All three models evaluated at 5.2 μm (`kidney_1_voi`), a ~10x resolution shift that was never seen in training:

| Model | In-dist Dice (50 μm) | OOD Dice (5.2 μm) |
|---|---|---|
| PhysicsAwareAdapter | 0.157 | 0.064 |
| Plain LoRA | 0.067 | 0.067 (best) |
| Spacing-Conditioned LoRA | 0.206 (best in-dist) | 0.002 (worst) |

The best in-distribution model drops to almost zero Dice out-of-distribution, the opposite of what a "spacing-aware" design should give.

### 4.3 Controlled 100-epoch GPU retrain

| Model | Trainable | In-dist Dice | OOD Dice |
|---|---|---|---|
| Plain LoRA (100ep) | 7.75% | 0.1838 (n=26/100) | 0.1672 (n=5/40) |
| Spacing-Conditioned LoRA (100ep) | 8.88% | 0.3145 (n=18/100) | 0.0058 (n=5/40) |

With 10x more training, both the in-distribution lead and the OOD collapse become clearer. So "10 epochs was too few" is not the explanation.

### 4.4 Multi-seed results (n=3)

Three independent 100-epoch training runs per architecture (the original run plus two reruns with fixed seeds), each evaluated in-distribution and OOD:

| Model | In-dist Dice (n=3, mean±std) | OOD Dice (n=3, mean±std) |
|---|---|---|
| Plain LoRA | 0.1838 ± 0.0644 | 0.0990 ± 0.0591 |
| Spacing-Conditioned LoRA | **0.2718 ± 0.0464** | **0.0102 ± 0.0079** |

Per-seed values: Plain LoRA in-dist [0.1838, 0.2482, 0.1194], OOD [0.1672, 0.0620, 0.0679]. Spacing-Conditioned LoRA in-dist [0.3145, 0.2224, 0.2786], OOD [0.0058, 0.0193, 0.0055].

The OOD collapse happens in all 3 seeds (Dice below 0.02 every time), and Plain LoRA beats it OOD in all 3. In-distribution, the ranking was unclear with n=2; with n=3 Spacing-Conditioned LoRA comes out ahead (2 of 3 seeds, both by a wide margin). n=3 is still small, so we do not make a formal significance claim.

### 4.5 Field-of-view confound test

Is the OOD failure caused by the spacing *value* itself, or by the fact that a native 64³ crop at 5.2 μm covers much less tissue (a field-of-view confound)? We compared the native-scale crop with a field-of-view-matched crop (the same physical size as a 64³ patch at ~50 μm, downsampled to 64³), both from the same 5.2 μm data:

| Model | In-dist (50 μm) | OOD native-scale | OOD FOV-matched |
|---|---|---|---|
| Plain LoRA | 0.1838 | 0.1672 (n=5/40) | 0.2895 (n=40/40) |
| Spacing-Conditioned LoRA | 0.3145 | 0.0058 (n=5/40) | **0.2215 (n=40/40)** |

All 40 FOV-matched patches contained annotated vessel, while 35/40 native-scale patches were empty; at 5.2 μm a 64³ window is often just too small to contain a vessel. Spacing-Conditioned LoRA's OOD Dice goes from 0.0058 to 0.2215, about 70% of its in-distribution score, only by matching the field of view on the *same* OOD data. On a new draw of 40 patches per model (Plain LoRA 0.3681, Spacing-Conditioned LoRA 0.2618), the FOV-matched gap gives Mann-Whitney U p=0.0216 (α=0.05). Most of the "OOD collapse" therefore looks like an ordinary shift in visual scale, not something specific to how the spacing value is used.

### 4.6 Per-layer FiLM activation breakdown

At the 100-epoch checkpoint, in all 24 injected layers, γ at the training spacing has become large (e.g. layer 19: γ_mean=6.52, γ_max=21.6 at 50 μm; layer 2: γ_max=19.7 at 50 μm), while γ in the *same layers* at the OOD spacing goes back toward identity (layer 19: γ_mean=1.59, γ_max=3.23 at 5.2 μm; layer 2: γ_max=3.06 at 5.2 μm). The pattern is the same across all 24 layers. Our interpretation: training pushes the FiLM network toward strong modulation tied to the training spacing, and the model comes to depend on it. At an unseen spacing FiLM falls back toward (but does not fully reach) Plain-LoRA-like behaviour that the model was never trained for. This is a second explanation that works alongside the FOV confound.

### 4.7 clDice loss-term ablation

| Model | In-dist Dice | OOD Dice |
|---|---|---|
| Spacing-Conditioned LoRA, with clDice | 0.3145 (n=18/100) | 0.0058 (n=5/40) |
| Spacing-Conditioned LoRA, no clDice | 0.1898 (n=19/100) | 0.0015 (n=3/40) |

The clDice term clearly helps in-distribution (without it Dice drops by ~40% relative), but it does not explain the OOD collapse, which happens either way.

### 4.8 Vascular graph structural validation

Connectivity and branching statistics from skeleton graphs (skimage `skeletonize` + a 26-connected `networkx` graph) of predicted vs. ground-truth masks, independent of Dice/clDice:

| Condition | n (non-empty) | Pred components (med) | GT components (med) | Pred branch pts (med) | GT branch pts (med) |
|---|---|---|---|---|---|
| Plain LoRA, in-dist | 23/80 | 3.0 | 3.0 | 0.0 | 1.0 |
| Plain LoRA, OOD | 5/40 | 6.0 | 1.0 | 1.0 | 1.0 |
| Spacing LoRA, in-dist | 22/80 | 3.0 | 1.0 | 0.5 | 0.5 |
| Spacing LoRA, OOD | 8/40 | 11.0 | 1.0 | 15.5 | 0.5 |

This gives a third, structural view. Plain LoRA's OOD predictions break up somewhat but keep a branch-point count close to the ground truth (graceful degradation). Spacing-Conditioned LoRA's OOD predictions are 11x more fragmented and have 31x more branch points than the ground truth: a scattered set of small, disconnected, over-branched pieces, not a smaller or weaker version of the real vessel tree.

### 4.9 Cause of ASSD=inf

We found the cause of the ASSD=inf values in Section 4.1: the model sometimes predicts a completely empty mask. A larger diagnostic draw confirmed this and showed it happens when the model misses very small or sparse annotated targets (~1% of non-empty patches). It is not a general property of either checkpoint. (The 100-epoch GPU run had a *different* problem: HD95/ASSD came out as nan because CuPy/Thrust JIT compilation failed on GPU tensors in the Kaggle environment. We fixed it by computing those two metrics on CPU.)

### 4.10 Hyperparameter sensitivity

A full grid (3 learning rates × 3 loss-α values × 2 architectures, 15 epochs each) shows that the best configuration changes with longer training. At 15 epochs all 18 configurations are close (Dice 0.126-0.172) and Plain LoRA is best overall (lr=2e-4, α=0.5, Dice=0.1724), which reverses the conclusion of an earlier, smaller sweep. All main results above (Sections 4.2-4.8) use the loss weights that were originally tuned for PhysicsAwareAdapter, not re-tuned per architecture. We list this as a limitation.

---

## 5. Discussion

### 5.1 Why does Spacing-Conditioned LoRA collapse OOD?

There are two explanations that complement each other, and neither is enough alone. (1) Field-of-view mismatch (Section 4.5) accounts for most of the size of the collapse and also affects Plain LoRA, only less. (2) The spacing-specific FiLM modulation falling back toward (but not reaching) Plain-LoRA-like behaviour (Section 4.6) may explain why Spacing-Conditioned LoRA suffers *more* than Plain LoRA from the FOV shift that remains after correction. The vascular-graph analysis (Section 4.8) adds a third, structural signature that fits both: fragmentation and spurious branching instead of graceful degradation.

### 5.2 Implications

Giving a PEFT adapter the physical spacing as input seems like a natural design choice, and it does give an in-distribution gain (over 3 seeds). But the gain seems to come from the adapter becoming *more* dependent on the specific training spacing, not more robust to spacing changes, which is the opposite of what the design is supposed to do. We present this as a specific, reproducible warning for anyone conditioning an adapter on acquisition metadata, not as a general argument against spacing conditioning.

### 5.3 Limitations

- One organ (kidney) and one dataset (SenNet+HOA); no cross-organ or multi-institution validation.
- Only two native spacings (50 μm for training, 5.2 μm held out for the OOD test), so "unseen spacing" and "unseen field-of-view" are not yet separated by a third, intermediate training spacing. Data for this is available and already prepared (three more native spacings of the same specimen, 25 μm, 6.5 μm and 2.6 μm, from the Human Organ Atlas via the `eternalaudrey/hipct_3d` HF mirror, converted to memmap). However, it only has Frangi pseudo-labels, not expert labels, and two of the three spacings (6.5 μm, 2.6 μm) are close enough to the 5.2 μm test point that training on them could make the OOD test less "unseen". This is still an open decision (`[TODO: see baki.sh section 3]`) because it affects the main claim of the paper.
- No direct comparison against nnU-Net (Isensee et al., 2020) or other published foundation-model baselines. nnU-Net is cited in Related Work as the reference point but not run here.
- n=3 seeds is enough to say the OOD collapse is not a one-off and that the in-distribution ranking leans toward Spacing-Conditioned LoRA, but it is not a properly powered estimate.
- No clinical validation, and no clinical claims are made.
- Hyperparameters were not re-tuned per architecture for the main 100-epoch results (Section 4.10 shows this matters).

---

## 6. Reproducibility Statement

The code, the commands to reproduce every number in this paper, and the dated experiment history are in `test.sh` and `baki.sh` in this repository. The repository also contains all checkpoints, the per-seed and per-patch JSON results (`sweep_results.json`, `full_sweep_results.json`, `vascular_graph_results.json`, `multiseed_seed1_results.json`, `multiseed_seed2_results.json`, `film_layer_analysis.json`), a script that rebuilds all summary tables from those files (`generate_summary_tables.py`) and a script that rebuilds all data figures (`generate_figures.py`). `[TODO: decide on public release, see Section 8.]`

**Settings used for the main 100-epoch results (Sections 4.2-4.4):** patch size 64×64×64 voxels; batch size 1; AdamW, learning rate 1e-4 (not tuned per architecture, Section 4.10); LoRA rank r=4, α=1.0, in 24 attention `qkv`/`proj` layers; loss = Dice + 0.5·clDice + 0.05·anisotropic gradient penalty + 1.0·class-weighted BCE (Section 3.3); 100 epochs, 20 samples/epoch; evaluation on 100 random 64³ patches in-distribution and 40 OOD, with empty-target patches excluded and counted; 3 seeds (Section 4.4); hardware: Kaggle T4 GPU for all 100-epoch training, local CPU for the original 10-epoch runs (Section 4.1) and all CPU-only analyses (vascular graph validation, ASSD diagnostic, FiLM layer breakdown).

---

## Declarations

**Data availability.** The SenNet+HOA "blood-vessel-segmentation" dataset (expert-annotated kidney vasculature, the main data for this paper) is publicly available on its Kaggle competition page. The three extra native-spacing kidney volumes, prepared only for the planned wider-spacing experiment (Section 5.3), come from the public Human Organ Atlas mirror `eternalaudrey/hipct_3d` on Hugging Face. No private or restricted data was used.

**Code availability.** `[TODO: decide on public release.]` The code is currently in a private local repository and has not been pushed anywhere. A DOI archive (e.g. Zenodo) is usually created from a GitHub release, so that comes after the release.

**Ethics statement.** `[TODO: probably "Not applicable": only a public, de-identified imaging dataset (SenNet+HOA) was used and no new human or animal data was collected. Check the wording the journal expects, and whether the ethics approval of the original SenNet+HOA / Human Organ Atlas data collection should be cited.]`

**Competing interests.** `[TODO.]`

**Funding.** `[TODO.]`

**Author contributions.** `[TODO.]`

**Acknowledgements.** `[TODO. Possibly: the SenNet+HOA / Human Organ Atlas dataset creators (cited in Section 2); the authors of SAM-Med3D, LoRA, clDice and FiLM; Kaggle's free GPU tier, used for all 100-epoch training.]`

---

## 7. Conclusion

We presented HiP-VesselAdapt, a spacing-aware parameter-efficient adaptation framework for 3D vessel segmentation with a frozen medical foundation model. It adds physical voxel spacing to a LoRA-based adaptation, so that acquisition scale can influence the trainable pathway while the foundation-model encoder stays frozen.

On expert-annotated kidney vessel data, spacing-conditioned LoRA gave strong in-distribution results across training seeds and beat a matched Plain LoRA baseline in 2 of 3 runs by a large margin. However, on a held-out 5.2 μm scan of the same specimen it degraded badly out-of-distribution, in all 3 runs, and ten times more training did not fix it. This shows that better performance inside the training resolution range does not guarantee robustness to very different resolutions, and we think this, rather than the in-distribution gain, is the main result of the paper.

Our analysis suggests that the OOD behaviour is linked to a combined shift in image scale and physical field of view (correcting the field of view recovers most of the lost performance). The spacing-conditioning itself, which falls back toward near-identity modulation at the unseen spacing in all 24 injected layers, is an additional factor. A graph-based comparison of predicted and ground-truth vessel topology supports both: the OOD failures of Spacing-Conditioned LoRA are not a smaller or weaker vessel tree but a different, fragmented and over-branched structure. Because field of view and spacing are coupled in our data, we cannot fully separate their effects here, and this remains a limitation.

Overall, providing spacing information to an adapter can help within a known resolution range, but it does not automatically give resolution-invariant generalization; in our case it made the OOD failure worse than a spacing-blind baseline. Future work should use wider, independently controlled spacing ranges to separate unseen spacing from unseen field of view, and test spacing-aware adaptation on more multi-resolution 3D vascular datasets before drawing general conclusions.

---

## References

1. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203-211.
2. Jain, Y., Walsh, C. L., Yagis, E., Aslani, S., Nandanwar, S., Zhou, Y., Ha, J., Gustilo, K. S., Brunet, J., Rahmani, S., Tafforeau, P., Bellier, A., Weber, G. M., Lee, P. D., & Börner, K. (2026). Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys. Nature Communications, 17, 8273.
3. Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J., & Qiao, Y. (2023). SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images. arXiv:2310.15161.
4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
5. Zhong, Z., Tang, Z., He, T., Fang, H., & Yuan, C. (2024). Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model. ICLR 2024.
6. Iltaf, A., Ahmed, R. M., Zhang, Z., Li, B., & Zhou, S. (2025). VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with AtrousLoRA. arXiv:2502.18185.
7. Joutard, S., Pietsch, M., & Prevost, R. (2024). HyperSpace: Hypernetworks for spacing-adaptive image segmentation. MICCAI 2024.
8. Garrido-Martínez, J. F., Garrido-López, J. M., Zapata-Pérez, J. F., & Martínez-Alajarín, J. (2026). Metadata Conditioning via Feature-Wise Linear Modulation for Multi-Domain 3D Abdominal Segmentation: A Comparative Study Between U-Net and nnU-Net. Sensors.
9. Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018.
10. Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J. P. W., Bauer, U., & Menze, B. H. (2021). clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR 2021.

`[TODO: see the note at the end of Section 2 for what is still missing (domain-shift surveys, other SAM-for-3D-medical variants, multi-spacing training if baki.sh section 3 goes ahead).]`

---

## 8. Remaining items

- **GitHub release / DOI archive**: not done yet. Git identity and GitHub login still need to be set up. The Zenodo DOI comes from a GitHub release, so it waits for that.
- **Supplementary material**: the JSON result files (Section 6) already cover this; they need to be put into the format the journal wants.
- **Journal formatting**: `[TODO: choose the venue. LaTeX template, word limits, abstract format and reference style all depend on it.]`
- **Cover letter**: write once the venue and the significance statement are decided.
- **Final check**: every number in this draft can be traced back to `test.sh`/`baki.sh`; go through it once against those files before submitting.

---

## Figures (6 of 6 done)

- **Figures 1-3** (method diagrams: shared backbone and injection points, the FiLM branch difference, the resolution/FOV setup): drawn separately with a diagram tool, not stored in this repo.
- **Figures 4-6** (results: n=3 Dice comparison with per-seed points, FiLM γ dumbbell chart over all 24 layers, vascular-graph comparison): made by `generate_figures.py`, saved in `figures/`.

For Figure 5 I needed per-layer data that did not exist yet, so I added an `--output-json` flag to `analyze_film_layers.py` and re-ran it on the 100-epoch checkpoint. The full 24-layer output is in `film_layer_analysis.json`.

`[TODO / not done yet]`:
- **Qualitative examples** (raw patch / GT / Plain LoRA / Spacing LoRA prediction side by side, one in-dist and one OOD): need to pick representative patches (median Dice per condition, not the best ones) and run inference to get the prediction masks.
- **FOV confound figure** (in-dist / OOD native / OOD FOV-matched, 3 bars per model, table in Section 4.5): not made yet, easy to add to `generate_figures.py`.
- **(Optional) 3D skeleton overlays**: not made; would need 3D rendering, more work than the 2D charts.
