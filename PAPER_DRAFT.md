# Spacing-Conditioned LoRA for 3D Vessel Segmentation: An Honest Account of an In-Distribution Gain and an Out-of-Distribution Collapse

**Status: DRAFT.** Every number below is transcribed from this repository's `test.sh` (the project's running, honestly-dated experimental log) and `baki.sh`. Nothing here is invented. Sections marked `[NEEDS AUTHOR INPUT]` are places a fabricated answer would be worse than no answer, and are left for you to fill in.

---

## Abstract

Adapting frozen 3D medical foundation models to new anatomy typically uses parameter-efficient fine-tuning (PEFT), but voxel spacing varies by orders of magnitude across acquisition protocols and is rarely given to the adapter as an explicit signal. We propose Spacing-Conditioned LoRA, a low-rank adaptation of SAM-Med3D's frozen image encoder in which the LoRA branch is modulated by a FiLM layer conditioned on physical voxel spacing, and compare it against a Plain LoRA baseline (identical architecture, no spacing signal) on real expert-annotated kidney vasculature (SenNet+HOA). Spacing-Conditioned LoRA wins in-distribution in 2 of 3 independent training runs at 100 epochs (mean Dice 0.272 vs. 0.184, n=3 seeds), but collapses catastrophically when evaluated at a real, unseen voxel spacing (5.2 μm vs. ~50 μm training data): OOD Dice under 0.02 in all 3 independent runs, while Plain LoRA degrades gracefully and outperforms it OOD in all 3 runs. We investigate this collapse with three independent lines of evidence -- a field-of-view confound test, a per-layer FiLM activation analysis, and a vascular-graph structural comparison -- and find it is not a training-budget artifact (confirmed at both 10 and 100 epochs), is substantially explained by field-of-view mismatch rather than the spacing signal itself, and manifests structurally as fragmented, over-branched predictions rather than a smaller/weaker version of the true vessel tree. We report this as a cautionary, reproducible finding about a natural-seeming design (giving an adapter explicit physical metadata) rather than a proposed fix, and provide the full reproduction protocol.

---

## 1. Introduction

Three-dimensional vessel segmentation in high-resolution computed tomography (HiP-CT) is important for quantitative analysis of vascular morphology and tissue structure. However, vessel appearance can change substantially with physical voxel spacing and imaging resolution. A model trained at one spatial scale may therefore experience a substantial distribution shift when applied to data acquired at a different resolution. This challenge is particularly relevant for high-resolution ex-vivo imaging, where datasets can contain large differences in physical sampling while representing similar anatomical structures.

Recent 3D medical foundation models provide a promising starting point for this problem because they can transfer representations learned from large-scale medical imaging data to downstream segmentation tasks. However, fully fine-tuning such models can be computationally expensive and may require substantial amounts of annotated data. Parameter-efficient fine-tuning (PEFT) offers an alternative by adapting only a small subset of model parameters while keeping the pretrained backbone largely frozen. Low-rank adaptation (LoRA) is particularly attractive in this setting because it can introduce task-specific adaptation with a relatively small trainable parameter budget.

An important limitation of conventional PEFT approaches is that physical voxel spacing is generally not treated as an explicit conditioning variable. Consequently, two volumes containing similar anatomical structures but acquired at substantially different physical resolutions may be processed through the same adaptation mechanism despite experiencing different image-scale characteristics. Incorporating physical spacing into the adaptation process provides a way to explicitly expose the model to this acquisition-level information.

In this work, we investigate spacing-aware parameter-efficient adaptation of a frozen 3D medical foundation model for vessel segmentation in multi-resolution HiP-CT data. We introduce a spacing-conditioned LoRA mechanism that modulates the low-rank adaptation pathway using physical voxel-spacing information. The proposed approach is evaluated against zero-shot and conventional parameter-efficient adaptation baselines using real expert-annotated kidney vessel data. In addition to in-distribution evaluation, we perform a resolution-shift experiment in which models trained at approximately 50 μm spacing are evaluated on a held-out 5.2 μm representation of the same physical specimen, creating a substantial resolution shift.

Rather than assuming that improved in-distribution performance necessarily translates to robustness under resolution changes, we explicitly investigate this assumption. Our experiments show that spacing-conditioned adaptation can improve in-distribution segmentation performance while exhibiting severe degradation under the unseen resolution shift. We therefore complement quantitative segmentation evaluation with topology-oriented analysis and targeted investigation of possible sources of the observed out-of-distribution behavior. This analysis provides a more complete characterization of both the potential and limitations of spacing-aware parameter-efficient adaptation for multi-resolution 3D vessel segmentation.

The central contribution of this work is therefore not a method we recommend adopting as-is. It is a controlled comparison that surfaces a specific, reproducible failure mode in a design that looks reasonable on paper: giving a PEFT adapter explicit physical spacing as a conditioning signal. We report the failure with the same rigor we would report a success, including three independent seeds and three independent investigative angles into *why* it fails.

---

## 2. Related Work

Every citation below was verified directly against its source (title, authors, venue checked live) before being added -- none is recalled from memory or accepted from an unverified secondary mention. A general caution for anyone extending this section: an earlier pass over candidate related work surfaced several plausible-sounding method names ("VesselFM," "TopoLoRA-SAM," "MorVess") that could not be confirmed as real, citable publications and were deliberately excluded rather than guessed at. One name from that same pass -- "HyperSpace" -- *did* turn out to be real once checked directly (see below), which is itself the reason every entry here was checked rather than trusted secondhand.

**Segmentation backbone and baseline.**
Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203-211. The field's standard, self-configuring segmentation baseline; not implemented as a comparison in this work (see Limitations) but cited as the reference point any PEFT-based approach should eventually be measured against.

Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J., & Qiao, Y. (2023). SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images. arXiv:2310.15161. The frozen 3D backbone this work adapts.

**Parameter-efficient fine-tuning of segmentation foundation models.**
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. The low-rank mechanism both Plain LoRA and Spacing-Conditioned LoRA build on.

Zhong, Z., Tang, Z., He, T., Fang, H., & Yuan, C. (2024). Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model. *ICLR 2024*. Injects convolutional inductive bias into LoRA for SAM's encoder -- a different modification to the same LoRA-on-SAM starting point this work uses, aimed at a different problem (image-domain bias, not physical acquisition metadata).

Iltaf, A., Ahmed, R. M., Zhang, Z., Li, B., & Zhou, S. (2025). VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with AtrousLoRA. arXiv:2502.18185. The closest prior work in both domain and mechanism: SAM adapted with a LoRA variant specifically for vessel segmentation. Targets aortic (not renal/HiP-CT) vasculature and does not condition on physical voxel spacing -- the gap this work's comparison is aimed at.

**Spacing- and metadata-conditioned segmentation (the direct comparison class for this work's core idea).**
Joutard, S., Pietsch, M., & Prevost, R. (2024). HyperSpace: Hypernetworks for spacing-adaptive image segmentation. *MICCAI 2024*. Uses a hypernetwork conditioned on voxel spacing to let one segmentation model process images natively at varying resolutions -- the same underlying premise as this work (explicit spacing as a conditioning signal), via a hypernetwork over full model weights rather than a FiLM-modulated LoRA branch. HyperSpace's own evaluation does not report the kind of severe held-out-resolution collapse this work finds for FiLM-modulated LoRA; whether that reflects the different conditioning mechanism, a different resolution range, or a difference in how "unseen resolution" was tested is an open question this work does not resolve, and a direct empirical comparison against HyperSpace on this project's own SenNet+HOA data is a natural next step, not yet done.

Garrido-Martínez, J. F., Garrido-López, J. M., Zapata-Pérez, J. F., & Martínez-Alajarín, J. (2026). Metadata Conditioning via Feature-Wise Linear Modulation for Multi-Domain 3D Abdominal Segmentation: A Comparative Study Between U-Net and nnU-Net. *Sensors*. The most directly analogous mechanism to this work's Spacing-Conditioned LoRA: FiLM-based conditioning on case-level metadata (acquisition sequence, intensity descriptors -- not physical spacing specifically) for 3D abdominal segmentation. Notably, that study finds FiLM conditioning helps a plain U-Net modestly but does *not* consistently help the already-optimized nnU-Net -- a different but related caution about FiLM-based conditioning not reliably paying off, worth discussing alongside this work's OOD collapse finding rather than treating the two as unrelated.

Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*. The general conditioning mechanism both this work and Garrido-Martínez et al. (2026) apply to a medical segmentation setting.

**Topological/structural loss.**
Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J. P. W., Bauer, U., & Menze, B. H. (2021). clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR 2021*.

**Data.**
Jain, Y., Walsh, C. L., Yagis, E., Aslani, S., Nandanwar, S., Zhou, Y., Ha, J., Gustilo, K. S., Brunet, J., Rahmani, S., Tafforeau, P., Bellier, A., Weber, G. M., Lee, P. D., & Börner, K. (2026). Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys. *Nature Communications*, 17, 8273. https://doi.org/10.1038/s41467-026-74050-8

`[NEEDS AUTHOR INPUT: this pass covers the mechanism (LoRA-on-SAM, FiLM conditioning, spacing-adaptive segmentation) and the reference baseline (nnU-Net) directly relevant to this work's core comparison. A thorough submission should still add: broader domain-shift/domain-generalization work in medical segmentation (a 2024 survey, "Navigating Distribution Shifts in Medical Image Analysis," arXiv:2411.05824, turned up in the same search pass and is a reasonable entry point but was not itself verified beyond its search snippet -- confirm directly before citing); any other SAM-for-3D-medical-imaging variants relevant to your specific claims; and, if you pursue the wider-spacing training experiment (baki.sh section 3), literature on training across multiple native physical resolutions specifically.]`

---

## 3. Methods

### 3.1 Data

Real, expert-annotated kidney vasculature from the SenNet+HOA "blood-vessel-segmentation" dataset. Voxel spacing is not embedded in the source TIFF files or dataset metadata and was taken directly from the companion paper's Table 1 (Jain et al., 2026) -- every spacing value used in this project is documented, not inferred (`utils/kidney_metadata_verification.py`). Training used `kidney_1_dense`, `kidney_2`, `kidney_3_dense`, `kidney_3_sparse` (all ~50 μm). `kidney_1_voi` -- the same physical specimen as `kidney_1_dense` (LADAF-2021-17-Right) imaged at 5.2 μm -- was held out entirely from training and used as the out-of-distribution resolution test set, giving a genuine same-organ, same-annotation-protocol resolution shift rather than a cross-dataset domain shift.

### 3.2 Architectures

All three compared models freeze the SAM-Med3D image encoder; they differ only in the trainable adaptation mechanism placed on top of it (the `mask_decoder` is always trainable in all three):

- **PhysicsAwareAdapter**: a trainable bottleneck adapter (down-project / GELU / up-project, reduction=4) applied to the frozen encoder's output *tokens*, gated multiplicatively by a spacing embedding (`spacing_fc`, a 2-layer MLP from the 3-number spacing vector) before the bottleneck -- i.e. this architecture is **also spacing-conditioned**, just via post-encoder multiplicative gating rather than in-encoder FiLM (7.65% of total parameters trainable).
- **Plain LoRA** (r=4): low-rank updates injected into the encoder's attention `qkv`/`proj` layers, zero-initialized so LoRA starts as a no-op. This is the only one of the three architectures with **no spacing signal anywhere in its forward pass** (7.75% trainable).
- **Spacing-Conditioned LoRA** (r=4): identical LoRA injection points as Plain LoRA, additionally modulated by a FiLM layer (Perez et al., 2018) conditioned on the patch's physical voxel spacing -- `models/spacing_lora.py`'s `spacing_film`, a small MLP producing per-layer (γ, β) from the 3-number spacing vector, applied to the LoRA branch's output (not the frozen base path) across 24 injected attention layers, zero-initialized so γ starts at 1 and β at 0 (identity modulation) (8.88% trainable). Spacing FiLM networks add ~1.25M parameters over Plain LoRA -- the three architectures' trainable budgets are the same order of magnitude, so no result below is explained by one model simply having more capacity.

All three use the same auto-prompting scheme: since SAM-Med3D's mask decoder is a promptable head with no automatic (prompt-free) mode, every forward pass prompts with two whole-volume corner points standing in for a bounding box covering the entire patch, rather than a real point/box prompt -- there is no user-in-the-loop prompting at train or eval time.

### 3.3 Training protocol

64³ voxel patches, AdamW, patch-level loss (`losses/combined_loss.py`, `HiPVesselLoss`, α=0.5/β=0.05/γ=1.0 default) combining four terms: soft Dice loss; the clDice topological term (Shit et al., 2021, weight α); an **anisotropic gradient penalty** weighted by inverse physical voxel spacing per axis (weight β) -- not a generic boundary term, specifically `mean((∂pred/∂z / spacing_z)² + (∂pred/∂y / spacing_y)² + (∂pred/∂x / spacing_x)²)`, computed from the same spacing metadata used by the spacing-conditioned architectures; and class-weighted BCE on raw logits (weight γ), with the positive-class weight computed per-batch from the actual foreground fraction (clamped to 50, since foreground is only ~1-5% of voxels). Note the anisotropic penalty means **all three architectures' training losses use spacing metadata**, even Plain LoRA's -- the architectural comparison is about whether spacing informs the *model's forward pass*, not whether it appears anywhere in the training objective. Two training regimes are reported: an initial 10-epoch run on CPU (this project's only local compute), and a controlled 100-epoch retrain on a Kaggle T4 GPU once the 10-epoch OOD finding needed to be checked against an under-training explanation.

### 3.4 Evaluation protocol

100 random 64³ patches per in-distribution run, 40 per OOD run (from `kidney_1_voi`); patches with zero annotated foreground are excluded from Dice/IoU/clDice/HD95/ASSD (reported as excluded counts, not hidden) since these metrics are mathematically undefined or misleadingly favorable on an empty target. Metrics: Dice, IoU, clDice, HD95, ASSD (MONAI implementations).

---

## 4. Results

### 4.1 In-distribution comparison, single run (10 epochs, CPU)

| Model | Trainable | Dice | IoU | clDice | HD95 (vox) | ASSD (vox) |
|---|---|---|---|---|---|---|
| Zero-shot SAM-Med3D | 0% | 0.088 | 0.052 | 0.076 | 48.1 | 25.2 |
| PhysicsAwareAdapter | 7.65% | 0.157 | 0.094 | 0.139 | 36.2 | inf* |
| Plain LoRA | 7.75% | 0.067 | 0.038 | 0.057 | 45.7 | 24.8 |
| Spacing-Conditioned LoRA | 8.88% | **0.206** | **0.131** | **0.261** | **23.4** | inf* |

Spacing-Conditioned LoRA beats every other model on every metric at 10 epochs, including its most direct ablation (Plain LoRA -- identical architecture, the only difference is spacing modulation). Plain LoRA notably underperforms even the untrained zero-shot baseline at 10 epochs; all three trained runs used identical, un-retuned loss weights/learning rate (a real confound, addressed by the hyperparameter sweep, Section 4.8).

*ASSD=inf initially appeared for PhysicsAwareAdapter and Spacing-Conditioned LoRA. Root-caused (Section 4.9): a rare (~1% of non-empty patches), patch-draw-dependent event where the model produces a completely empty prediction on a patch with a vanishingly small annotated target -- not a systemic property of either checkpoint.

### 4.2 Resolution-OOD test (the paper's central finding)

All three models evaluated at 5.2 μm (`kidney_1_voi`), a genuine ~10x unseen resolution shift, held out from all training:

| Model | In-dist Dice (50 μm) | OOD Dice (5.2 μm) |
|---|---|---|
| PhysicsAwareAdapter | 0.157 | 0.064 |
| Plain LoRA | 0.067 | 0.067 (best) |
| Spacing-Conditioned LoRA | 0.206 (best in-dist) | 0.002 (worst) |

The in-distribution winner collapses to near-zero Dice out-of-distribution -- the opposite of what a "spacing-aware" mechanism would be expected to buy.

### 4.3 Controlled 100-epoch GPU retrain -- confirms the collapse is not under-training

| Model | Trainable | In-dist Dice | OOD Dice |
|---|---|---|---|
| Plain LoRA (100ep) | 7.75% | 0.1838 (n=26/100) | 0.1672 (n=5/40) |
| Spacing-Conditioned LoRA (100ep) | 8.88% | 0.3145 (n=18/100) | 0.0058 (n=5/40) |

10x more training sharpens both the in-distribution lead and the OOD collapse. Rules out "10 epochs was too few" as the explanation.

### 4.4 Multi-seed statistical reporting (n=3)

Three independent 100-epoch training runs per architecture (original run + two explicitly-seeded reruns), in-distribution and OOD every time:

| Model | In-dist Dice (n=3, mean±std) | OOD Dice (n=3, mean±std) |
|---|---|---|
| Plain LoRA | 0.1838 ± 0.0644 | 0.0990 ± 0.0591 |
| Spacing-Conditioned LoRA | **0.2718 ± 0.0464** | **0.0102 ± 0.0079** |

Per-seed raw values -- Plain LoRA in-dist: [0.1838, 0.2482, 0.1194], OOD: [0.1672, 0.0620, 0.0679]. Spacing-Conditioned LoRA in-dist: [0.3145, 0.2224, 0.2786], OOD: [0.0058, 0.0193, 0.0055].

The OOD collapse is 3-for-3 across independent seeds (every run under Dice 0.02) and Plain LoRA beats it OOD in all 3 runs. The in-distribution ranking, ambiguous at n=2, resolves toward Spacing-Conditioned LoRA winning (2 of 3 seeds, by a wide margin both times). n=3 is still a small-n result and is not reported as a formal significance claim.

### 4.5 Field-of-view confound test

Root-cause investigation: is the OOD failure about the spacing *value* itself, or simply that a native-scale 64³ crop at 5.2 μm covers far less real tissue (a field-of-view confound)? A field-of-view-matched crop (physically the same extent as a 64³ patch at ~50 μm, downsampled to 64³) was compared against the native-scale crop, same real 5.2 μm data:

| Model | In-dist (50 μm) | OOD native-scale | OOD FOV-matched |
|---|---|---|---|
| Plain LoRA | 0.1838 | 0.1672 (n=5/40) | 0.2895 (n=40/40) |
| Spacing-Conditioned LoRA | 0.3145 | 0.0058 (n=5/40) | **0.2215 (n=40/40)** |

All 40 FOV-matched patches had real annotated vessel, vs. 35/40 empty at native scale -- the native 64³ window at 5.2 μm is often simply too small to contain a vessel. Spacing-Conditioned LoRA's OOD Dice recovers from catastrophic (0.0058) to 0.2215 -- roughly 70% of its own in-distribution score -- purely by matching field of view on the *same* real OOD data. A significance check on a fresh 40-patch draw per model (Plain LoRA 0.3681, Spacing-Conditioned LoRA 0.2618) gives Mann-Whitney U p=0.0216 (α=0.05) for the FOV-matched gap. Most of the "OOD collapse" looks like an ordinary unseen-visual-scale domain shift, not something specific to how the spacing value is used numerically.

### 4.6 Per-layer FiLM activation breakdown

At the 100-epoch checkpoint, across all 24 injected layers: γ at the in-distribution spacing has grown large during training (e.g., layer 19: γ_mean=6.52, γ_max=21.6 at 50 μm; layer 2: γ_max=19.7 at 50 μm), while the *same layers'* γ at the OOD spacing reverts toward identity (layer 19: γ_mean=1.59, γ_max=3.23 at 5.2 μm; layer 2: γ_max=3.06 at 5.2 μm) -- consistent across all 24 layers. Revised hypothesis: training pushes the FiLM network toward strong, training-spacing-specific modulation the model becomes reliant on; at an unseen spacing, FiLM degenerates toward (not cleanly reaching) Plain-LoRA-like behavior it was never optimized for -- a second, complementary lead alongside the FOV confound.

### 4.7 clDice loss-term ablation

| Model | In-dist Dice | OOD Dice |
|---|---|---|
| Spacing-Conditioned LoRA, with clDice | 0.3145 (n=18/100) | 0.0058 (n=5/40) |
| Spacing-Conditioned LoRA, no clDice | 0.1898 (n=19/100) | 0.0015 (n=3/40) |

The clDice loss term contributes real in-distribution performance (removing it drops Dice ~40% relative) but does not explain the OOD collapse -- catastrophic either way.

### 4.8 Vascular graph structural validation

Skeleton-graph connectivity/branching statistics (skimage `skeletonize` + a 26-connected `networkx` graph) on predicted vs. real GT masks, independent of Dice/clDice:

| Condition | n (non-empty) | Pred components (med) | GT components (med) | Pred branch pts (med) | GT branch pts (med) |
|---|---|---|---|---|---|
| Plain LoRA, in-dist | 23/80 | 3.0 | 3.0 | 0.0 | 1.0 |
| Plain LoRA, OOD | 5/40 | 6.0 | 1.0 | 1.0 | 1.0 |
| Spacing LoRA, in-dist | 22/80 | 3.0 | 1.0 | 0.5 | 0.5 |
| Spacing LoRA, OOD | 8/40 | 11.0 | 1.0 | 15.5 | 0.5 |

A third, structurally distinct piece of evidence: Plain LoRA's OOD predictions fragment moderately but keep branch-point counts close to real GT (graceful degradation). Spacing-Conditioned LoRA's OOD predictions fragment 11x worse and hyper-branch 31x worse than GT -- a scattered field of small disconnected over-branched fragments, not a smaller/weaker prediction of the true vessel tree.

### 4.9 Root cause found: ASSD=inf

The original ASSD=inf finding (Section 4.1) was root-caused: a genuine completely-empty-prediction event, confirmed via a larger diagnostic draw, correlated with the model missing tiny/sparse annotated targets (~1% of non-empty patches) -- not a systemic property of either checkpoint. (A *different*, already-explained anomaly affects the 100-epoch GPU run specifically: HD95/ASSD returned nan there due to a Kaggle-environment CuPy/Thrust JIT-compilation failure on GPU tensors, fixed by moving those two metrics' computation to CPU.)

### 4.10 Hyperparameter sensitivity

A full grid (3 learning rates × 3 loss-α values × 2 architectures, 15 epochs each) found the best configuration shifts with more training: at 15 epochs, Dice values across all 18 configs are close together (0.126-0.172) and Plain LoRA wins overall (lr=2e-4, α=0.5, Dice=0.1724) -- reversing an earlier reduced-grid sweep's conclusion. All headline results above (Sections 4.2-4.8) used the loss weights tuned originally against PhysicsAwareAdapter, not re-tuned per architecture -- a stated limitation, not a hidden one.

---

## 5. Discussion

### 5.1 Why does Spacing-Conditioned LoRA collapse OOD?

Two independent, complementary explanations, neither alone sufficient: (1) field-of-view mismatch (Section 4.5) accounts for most of the raw magnitude of the collapse and affects Plain LoRA too, just less severely; (2) FiLM's spacing-specific modulation degenerating toward (not reaching) plain-LoRA-like behavior (Section 4.6) plausibly explains why Spacing-Conditioned LoRA is hit *harder* than Plain LoRA by whatever FOV-shift remains even after correction. The vascular-graph evidence (Section 4.8) adds a third, independent, structurally distinct signature consistent with both: not graceful degradation, but fragmentation and spurious branching.

### 5.2 Implications

Explicitly conditioning a PEFT adapter on physical spacing metadata is a natural-seeming design choice, and it delivers a real in-distribution gain (confirmed across 3 seeds). But it appears to buy that gain by making the adapter *more* reliant on the specific training-distribution spacing value, not more robust to spacing variation generally -- the opposite of what the design intuitively promises. This is reported as a specific, reproducible caution for anyone conditioning an adapter on acquisition metadata, not as a universal claim against spacing-conditioning as a concept.

### 5.3 Limitations

- Single organ (kidney), single dataset (SenNet+HOA); no cross-organ or multi-institution validation.
- Only two native physical spacings were used for training (50 μm; 5.2 μm held out as the OOD test) -- "unseen spacing" and "unseen field-of-view" are not yet disentangled by a third, intermediate native training spacing. Real data for this exists and is now staged (three more native spacings -- 25 μm, 6.5 μm, 2.6 μm -- of the same physical specimen, Human Organ Atlas via the `eternalaudrey/hipct_3d` HF mirror, converted to memmap this session), but is Frangi-pseudo-labeled, not real expert GT, and two of the three spacings (6.5 μm, 2.6 μm) sit close enough to the 5.2 μm OOD test point that training on them risks weakening that test's "fully unseen spacing" validity -- an open design decision (`[NEEDS AUTHOR INPUT: see baki.sh section 3]`), not yet resolved, and not something to decide unilaterally given it affects this paper's central claim's validity.
- No direct comparison against nnU-Net (Isensee et al., 2020) or other published foundation-model baselines -- cited in Related Work as the field's reference point, not implemented here.
- n=3 seeds is enough to say the OOD collapse is not a single-run fluke and that the in-distribution ranking leans toward Spacing-Conditioned LoRA, but is not a properly powered estimate by normal statistical standards.
- No clinical validation; no claim of clinical usefulness is made.
- Hyperparameters were not independently re-tuned per architecture for the headline 100-epoch results (Section 4.10 shows this matters).

---

## 6. Reproducibility Statement

Full code, exact commands to reproduce every number in this paper, and dated experimental history are in this repository's `test.sh` and `baki.sh`. All checkpoints, raw per-seed/per-patch JSON results (`sweep_results.json`, `full_sweep_results.json`, `vascular_graph_results.json`, `multiseed_seed1_results.json`, `multiseed_seed2_results.json`, `film_layer_analysis.json`), a script to regenerate every summary table in this paper from those files (`generate_summary_tables.py`), and a script to regenerate every data-driven figure (`generate_figures.py`) are included. `[NEEDS AUTHOR INPUT: decide on public release -- see Section 8 below.]`

**Methods reproducibility, at a glance** (every value below is the real one used for the headline 100-epoch results, Section 4.2-4.4, not a default that was later changed): patch size 64×64×64 voxels; batch size 1; optimizer AdamW, learning rate 1e-4 (untuned per architecture, Section 4.10); LoRA rank r=4, α=1.0, injected into 24 attention `qkv`/`proj` layers; loss = Dice + 0.5·clDice + 0.05·anisotropic-gradient penalty + 1.0·class-weighted BCE (Section 3.3); 100 epochs, 20 samples/epoch; evaluation on 100 random 64³ patches in-distribution / 40 OOD, empty-target patches excluded and reported; 3 independent seeds (Section 4.4); hardware: Kaggle T4 GPU for all 100-epoch training, this project's own CPU for the original 10-epoch runs (Section 4.1) and all CPU-only analysis (vascular graph validation, ASSD diagnostic, FiLM layer breakdown).

---

## Declarations

**Data availability.** The SenNet+HOA "blood-vessel-segmentation" dataset (real expert-annotated kidney vasculature, this paper's primary data) is publicly available via its Kaggle competition page. The three additional native-spacing kidney volumes used only for the staged-but-not-yet-trained wider-spacing experiment (Section 5.3's Limitations) are from the public Human Organ Atlas mirror `eternalaudrey/hipct_3d` on Hugging Face. No private or restricted-access data was used.

**Code availability.** `[NEEDS AUTHOR INPUT: decide on public release.]` All code is currently in a local, private working repository, not yet pushed to a public host. If released: no git identity (`user.name`/`user.email`) is currently configured in this repository and no GitHub authentication is set up in the development environment used -- both would need to be done directly by the author(s), not automated. A DOI archive (e.g., via Zenodo) is normally created from a GitHub release, so is blocked on the same step.

**Ethics statement.** `[NEEDS AUTHOR INPUT -- likely "Not applicable": this work uses only a pre-existing, publicly released, de-identified imaging dataset (SenNet+HOA); no new human or animal data was collected. Confirm the exact wording your target journal expects, and whether the original SenNet+HOA/Human Organ Atlas data collection's own ethics approval needs to be cited here.]`

**Competing interests.** `[NEEDS AUTHOR INPUT.]`

**Funding.** `[NEEDS AUTHOR INPUT.]`

**Author contributions.** `[NEEDS AUTHOR INPUT -- state this in your own words; I have no basis to draft it.]`

**Acknowledgements.** `[NEEDS AUTHOR INPUT. Consider acknowledging: the SenNet+HOA/Human Organ Atlas dataset creators (Section 2's citation); the SAM-Med3D, LoRA, clDice, and FiLM authors whose code/methods this work builds on; Kaggle's free GPU tier, which all 100-epoch training in this paper ran on.]`

---

## 7. Conclusion

We presented HiP-VesselAdapt, a spacing-aware parameter-efficient adaptation framework for 3D vessel segmentation using a frozen medical foundation model. The proposed approach incorporates physical voxel spacing into a LoRA-based adaptation mechanism, allowing acquisition-scale information to influence the trainable adaptation pathway while keeping the main foundation-model encoder frozen.

Experiments on real expert-annotated kidney vessel data showed that spacing-conditioned LoRA achieved strong in-distribution segmentation performance across independent training seeds, outperforming a matched Plain LoRA baseline in 2 of 3 runs by a substantial margin. However, evaluation on a held-out 5.2 μm representation of the same physical specimen revealed a pronounced out-of-distribution degradation despite the improved in-distribution results -- confirmed in all 3 independent training runs, and not resolved by an order of magnitude more training. This finding demonstrates that improved performance within the training resolution range does not by itself guarantee robustness to substantially different physical resolutions, and is, we believe, the paper's actual headline result rather than the in-distribution gain.

Our subsequent analysis indicates that the observed OOD behavior is associated with a coupled shift in image scale and physical field-of-view -- correcting for field of view recovers most of the lost performance -- while the behavior of the spacing-conditioning mechanism itself, which reverts toward near-identity modulation at the unseen spacing across all 24 injected layers, provides an additional, complementary factor. A structural, graph-based comparison of predicted against real ground-truth vessel topology corroborates both: Spacing-Conditioned LoRA's out-of-distribution failures are not a smaller or weaker version of the true vessel tree, but a qualitatively different, fragmented and over-branched structure. Because field-of-view and spacing are coupled in the data available for this study, their individual causal contributions cannot be completely disentangled here, and this remains a limitation of the work rather than a resolved mechanism.

Overall, the results highlight an important consideration for spacing-aware adaptation of 3D medical foundation models: explicitly providing physical spacing information can improve adaptation within a known resolution regime, but it does not automatically provide resolution-invariant generalization, and in this instance made the out-of-distribution failure worse than a spacing-blind baseline rather than better. Future work should investigate broader, independently-controlled spacing ranges -- disentangling unseen spacing from unseen field-of-view, which the data used here could not fully separate -- and evaluate spacing-aware adaptation across additional multi-resolution 3D vascular datasets before drawing conclusions about the approach more generally.

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

`[NEEDS AUTHOR INPUT: this covers the mechanism/baseline literature directly relevant to this work's core comparison (see Section 2's own note for what's still missing -- broader domain-shift surveys, other SAM-for-3D-medical variants, multi-spacing-training literature if you pursue baki.sh section 3).]`

---

## 8. Everything else you asked for (status, not drafted here)

- **Final figures**: see the dedicated recommendation list below. **Not generated** -- this repository has a strict text-only, no-plotting rule (every result above is a table for exactly this reason). Actual chart/figure files should be produced in a separate tool/notebook outside this repo, from the JSON files already listed in Section 6.
- **Code cleanup**: reviewed -- the codebase is already in a clean, documented state (every script has a docstring explaining what and why, `baki.sh`/`test.sh` track status honestly). No dead code or stray debug files found. Nothing further needed unless you want a specific pass (e.g., type hints, docstring standardization).
- **GitHub release / DOI archive**: blocked on the same two things noted earlier this session -- git identity (`user.name`/`user.email`) is not configured in this repo, and there is no GitHub authentication available in this environment (no `gh` CLI login, no SSH key, no credential helper). Both need to be set up by you directly; I can push once they are, if you want to revisit the earlier "no GitHub push" decision. A DOI archive (e.g., Zenodo) is normally created *from* a GitHub release, so it's blocked on the same prerequisite.
- **Supplementary material**: the raw JSON result files (Section 6) already function as this. I can assemble them into a single supplementary PDF/document structure once you tell me the target journal's expected format.
- **Journal-specific formatting**: `[NEEDS AUTHOR INPUT: which journal/venue? Formatting (LaTeX template, word limits, structured abstract requirements, reference style) differs enough between venues that I don't want to guess and reformat twice.]`
- **Cover letter**: needs the target journal and a decided-on "significance statement" -- happy to draft once you pick a venue.
- **Final internal review**: everything in this draft is traceable to a real number in `test.sh`/`baki.sh` -- recommend you (or a co-author) read it once against those files before anything goes out, same as any other honest review pass.

---

## Figures -- status: 6 of 6 built, published as Artifacts (not in this repo)

Per this repo's standing no-plotting rule, no chart-generation code was added to the codebase. Instead, all six figures were built and published as two Claude Artifacts (real data, exportable as PNG/PDF for submission from the artifact page itself):

- **Figures 1-3** (mechanism diagrams -- shared backbone/injection points, the exact FiLM branch difference, the resolution+FOV setup): `https://claude.ai/artifact/KgJym8jrYWrfMiXR2pEXbU`
- **Figures 4-6** (results, real data -- n=3 Dice comparison with per-seed points, all-24-layer FiLM γ dumbbell chart, vascular-graph structural comparison): `https://claude.ai/artifact/EnfyZojxpQZvets7Qi8KN7`

Figure 5 needed real per-layer data that didn't exist yet -- `analyze_film_layers.py` gained an `--output-json` flag and was re-run against the 100-epoch checkpoint; full 24-layer output is in `film_layer_analysis.json`.

`[NEEDS AUTHOR INPUT / NOT YET DONE]`:
- **Qualitative renders** (raw patch / GT / Plain LoRA / Spacing LoRA prediction, side by side, one in-dist + one OOD): needs picking genuinely representative patches (by median Dice within condition, not best-case) and running inference to get the actual prediction masks -- not done yet, ask if you want this built.
- **FOV confound recovery figure** (In-dist / OOD-native / OOD-FOV-matched, 3 bars per model, Section 4.5's table): not yet built as a chart -- straightforward to add to the Figures 4-6 artifact if wanted.
- **(Optional) 3D skeleton overlays**: not built -- would need 3D rendering, a bigger lift than the 2D charts above.
