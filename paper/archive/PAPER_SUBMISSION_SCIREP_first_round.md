# Spacing-conditioned LoRA improves in-distribution vessel segmentation but collapses out-of-distribution

**Status: draft for *Scientific Reports*.** Format rules as of 2026-09-16: unstructured abstract of at most 200 words; main text at most 4,500 words (not counting Abstract, Methods, References and figure legends); at most 8 figures and tables combined in the main text; Nature numbered reference style. This is a shortened version of `PAPER_DRAFT.md`, which keeps all the tables and is the reference for every number. No new results are added here. `[TODO]` marks things I still need to fill in.

Author(s): Rabeya Noor `[TODO: affiliation]`

---

## Abstract *(199 words)*

Parameter-efficient fine-tuning (PEFT) is commonly used to adapt frozen 3D medical foundation models to new anatomy, but the adapter is rarely told the physical voxel spacing, even though spacing can vary by orders of magnitude. We introduce Spacing-Conditioned LoRA, a low-rank adaptation of a frozen 3D segmentation backbone in which the LoRA branch is modulated by a FiLM layer conditioned on voxel spacing. We compare it with a matched Plain LoRA baseline (no spacing input) on expert-annotated kidney vasculature. Over three independent runs, Spacing-Conditioned LoRA is better in-distribution in two (mean Dice 0.272 vs. 0.184), but it collapses at an unseen voxel spacing (5.2 μm, against ~50 μm training data): out-of-distribution Dice is below 0.02 in all three runs, while Plain LoRA degrades gracefully and wins out-of-distribution every time. Three further analyses (a field-of-view-matched re-test, a per-layer activation breakdown and a vascular-graph comparison) show that the collapse is not caused by too little training, is largely explained by field-of-view mismatch rather than the spacing signal itself, and appears as fragmented, over-branched predictions instead of a weaker copy of the true vessel tree. We present this as a reproducible caution about a reasonable-looking design choice, not as a proposed fix.

---

## Introduction

Three-dimensional vessel segmentation in high-resolution computed tomography is important for quantitative analysis of vascular morphology and tissue structure. However, vessels can look very different depending on the physical voxel spacing. A model trained at one scale may see a large distribution shift when applied to data acquired at another resolution. This matters especially for high-resolution ex-vivo imaging, where the same specimen is often scanned at several native resolutions.

Recent 3D medical foundation models are a useful starting point because they transfer representations learned from large imaging datasets, but fully fine-tuning them is expensive. Parameter-efficient fine-tuning (PEFT) trains only a small subset of parameters instead. Low-rank adaptation (LoRA)<sup>1</sup> is one of the most widely used PEFT methods, including for the Segment Anything Model (SAM) family in medical imaging<sup>2,3,4</sup>. A limitation of standard PEFT is that voxel spacing is almost never given to the model as an input, so two volumes of similar anatomy at very different resolutions are adapted in exactly the same way.

In this work we study spacing-aware PEFT of a frozen 3D medical foundation model (SAM-Med3D<sup>2</sup>) for kidney vessel segmentation. We add a spacing-conditioned LoRA in which the low-rank pathway is modulated by the physical voxel spacing through a FiLM layer<sup>5,6</sup>. We compare it with zero-shot and spacing-blind PEFT baselines on expert-annotated data. Instead of assuming that an in-distribution gain means the model is robust to resolution, we test this directly with a resolution shift within a single specimen: training data at 50 μm, and a held-out 5.2 μm volume of the same kidney. The closest prior mechanisms are spacing-adaptive segmentation<sup>7</sup> and metadata-conditioned FiLM segmentation<sup>5</sup>, and neither reports the kind of collapse at a held-out resolution that we observe (see Related Work). Our main contribution is therefore not a method to adopt as it is, but a controlled study (three seeds, three independent analyses) of a reproducible failure mode in a design that looks sensible on paper.

---

## Related Work

**Segmentation backbone and reference baseline.** nnU-Net<sup>8</sup> is still the standard self-configuring segmentation method. We do not compare against it directly here (see Limitations), but any PEFT-based approach should eventually be measured against it. SAM-Med3D<sup>2</sup> is the frozen 3D backbone we adapt.

**PEFT of segmentation foundation models.** LoRA<sup>1</sup> is the low-rank mechanism used by both our baseline and our proposed method. Conv-LoRA<sup>3</sup> adds a convolutional inductive bias to LoRA in SAM's encoder; it starts from the same place but targets image-domain bias rather than acquisition metadata. VesselSAM<sup>4</sup> is the closest prior work in both domain and mechanism. It adapts SAM with a LoRA variant for aortic vessel segmentation, but it targets a different vascular bed and does not use voxel spacing.

**Spacing- and metadata-conditioned segmentation.** HyperSpace<sup>7</sup> conditions a hypernetwork on voxel spacing so that a single model can process images at their native resolution. The idea is the same as ours, but it uses full-weight hypernetworks instead of a FiLM-modulated LoRA branch. HyperSpace is trained with every batch resampled to a random spacing. Its accuracy also drops sharply at the highest resolutions it was tested on, which the authors attribute to the limited receptive field of the network, and resampling to the closest training resolution fixes it; this agrees with our field-of-view result. Comparing against HyperSpace on our data is an obvious next step that we have not done yet. Garrido-Martínez et al.<sup>5</sup> use FiLM conditioning on case-level metadata (not physical spacing) for 3D abdominal segmentation. They find it gives only a modest improvement for a plain U-Net and does not consistently help an already-tuned nnU-Net. This is a related warning that FiLM conditioning does not always pay off, and we discuss it next to our own OOD result. FiLM itself<sup>6</sup> is the general conditioning mechanism used in both works.

**Topological loss.** clDice<sup>9</sup> is the topology-preserving loss term in our training objective.

`[TODO: add broader domain-shift / domain-generalization literature for medical segmentation and other SAM-for-3D-medical variants. See PAPER_DRAFT.md Section 2 for what is still missing, including one survey that still needs to be checked.]`

---

## Results

### In-distribution gain (Table 1; 10-epoch exploratory comparison)

After 10 epochs (exploratory run on CPU), Spacing-Conditioned LoRA was better than every other model on every in-distribution metric, including its most direct ablation, Plain LoRA, which has the same architecture but no spacing input (Table 1). Plain LoRA was even worse than the untrained zero-shot baseline at this point. All trained runs used the same loss weights without re-tuning; we return to this under Hyperparameter sensitivity.

### Out-of-distribution collapse (Table 1; main finding)

We then evaluated the same three models on a held-out volume at roughly 10× finer resolution (5.2 μm, same physical specimen as the 50 μm training data). The in-distribution ranking reversed. Spacing-Conditioned LoRA, the clear winner in-distribution, dropped to almost zero Dice out-of-distribution (0.002), worse than both other models (Table 1).

### The collapse is not caused by too little training

We retrained both LoRA variants for 100 epochs on a GPU under controlled conditions. This made both effects stronger rather than removing them. In-distribution Dice for Plain LoRA and Spacing-Conditioned LoRA reached 0.1838 and 0.3145, while OOD Dice reached 0.1672 and 0.0058. Ten times more training did not close the OOD gap between the two architectures; if anything, the gap grew.

### The collapse replicates across seeds (Figure 4)

We ran three independent 100-epoch trainings per architecture (the original run plus two reruns with different seeds), so the pattern is not a one-off (Figure 4). Spacing-Conditioned LoRA had OOD Dice below 0.02 in all three runs (mean 0.0102 ± 0.0079), and Plain LoRA beat it OOD in all three (mean 0.0990 ± 0.0591). With only two runs the in-distribution ranking was unclear; with three it leans toward Spacing-Conditioned LoRA (mean 0.2718 ± 0.0464 vs. 0.1838 ± 0.0644, winning 2 of 3 seeds by a wide margin). This is still a small sample and we do not claim formal significance.

### Field-of-view mismatch explains most of the collapse (Table 2)

A fixed 64³-voxel window covers much less tissue at 5.2 μm than at 50 μm. We therefore checked whether the OOD failure comes from the spacing value itself or from this accompanying shrinkage of the field of view (FOV). When we re-cropped the same OOD data to match the training FOV (Table 2), Spacing-Conditioned LoRA's Dice went from 0.0058 back up to 0.2215, about 70% of its in-distribution score. All 40 FOV-matched patches contained vessel, compared with 35 of 40 empty patches at native scale. On an independent redraw of 40 patches per model (Plain LoRA 0.3681, Spacing-Conditioned LoRA 0.2618), a Mann-Whitney U test gave p=0.0216. So most of what looked like a spacing collapse behaves like an ordinary shift in visual scale.

### FiLM reverts toward identity at the unseen spacing (Figure 5)

In all 24 injected layers, the FiLM γ parameter becomes large and layer-specific at the training spacing after training (for example, layer 19: γ_mean = 6.52 at 50 μm), but moves back toward its identity initialization at the unseen spacing (same layer: γ_mean = 1.59 at 5.2 μm). This happens across all layers, not in a single outlier (Figure 5). It suggests a second explanation: training makes the model depend on strong modulation specific to the training spacing, and that modulation is missing at the OOD spacing.

### The failure is structurally different, not only worse (Figure 6)

We also compared skeleton graphs of the predictions, which does not depend on Dice. Plain LoRA's OOD predictions break up moderately but keep a number of branch points close to the ground truth, i.e. they degrade gracefully. Spacing-Conditioned LoRA's OOD predictions are 11× more fragmented and have 31× more branch points than the ground truth (Figure 6). They look like a scattered set of disconnected, over-branched pieces rather than a smaller version of the real vessel tree.

### Ablation and hyperparameter sensitivity (Supplementary Tables S1-S2)

Removing the clDice term lowers Spacing-Conditioned LoRA's in-distribution Dice by about 40% (relative), but it neither fixes nor worsens the OOD collapse; both versions fail badly (Supplementary Table S1). A full grid over learning rate and loss weight showed that the best setting depends on training length, but the OOD result does not change qualitatively (Supplementary Table S2). The main results above use loss weights chosen before this sweep was run, which is a limitation.

---

## Discussion

Two explanations together account for most of the collapse; neither is enough alone. Field-of-view mismatch (Table 2) explains most of the size of the drop and affects both architectures, though Plain LoRA less. The FiLM modulation falling back toward, but not fully reaching, Plain-LoRA-like behavior (Figure 5) may explain why Spacing-Conditioned LoRA suffers more from the FOV shift that remains after correction. The vascular-graph results (Figure 6) add a third, structural signature that fits both explanations: fragmentation and spurious branching instead of graceful degradation.

Giving a PEFT adapter the physical spacing as an input seems like a natural idea, and it does give an in-distribution gain that holds over three seeds. But it seems to get that gain by making the adapter depend more on the specific training spacing, not by making it more robust to spacing changes, which is the opposite of what one would expect. We see this as a caution for anyone who conditions a segmentation adapter on acquisition metadata, not as an argument against spacing conditioning in general. HyperSpace<sup>7</sup> trains over a range of spacings with a different conditioning mechanism, so a direct comparison would be a useful next step.

**Limitations.** We use one organ (kidney) and one dataset, with no cross-organ or multi-institution validation. There are only two native spacings (50 μm for training, 5.2 μm held out), so the effects of unseen spacing and unseen FOV are not fully separated. A third, intermediate native spacing would help; we have prepared data at three more native spacings of the same specimen, but it has no expert labels, and two of the three spacings are close enough to the OOD test spacing that training on them could weaken the test. We have not decided how to handle this yet. We do not compare against nnU-Net or other published foundation-model baselines. Three seeds are enough to show the collapse is not a fluke and that the in-distribution ranking leans one way, but not for a properly powered estimate. There is no clinical validation and we make no clinical claims. Loss hyperparameters were not re-tuned per architecture for the main results.

---

## Methods

**Data.** We use expert-annotated kidney vasculature from the SenNet+HOA "blood-vessel-segmentation" dataset<sup>10</sup>. Voxel spacing is not stored in the image files, so we took it from Table 1 of the dataset paper. Training used four ~50 μm volumes (`kidney_1_dense`, `kidney_2`, `kidney_3_dense`, `kidney_3_sparse`). `kidney_1_voi`, which is the same specimen as `kidney_1_dense` scanned at 5.2 μm, was kept out of training completely and used as the OOD test set. This gives a resolution shift with the same organ and the same annotation protocol.

**Architectures.** All three models freeze the SAM-Med3D<sup>2</sup> image encoder and train the mask decoder; they differ only in how they adapt the encoder. PhysicsAwareAdapter is a bottleneck adapter after the encoder, gated multiplicatively by a spacing embedding (7.65% trainable). Plain LoRA (r=4) adds low-rank updates to the encoder's attention `qkv`/`proj` layers, zero-initialized so they start as a no-op, and has no spacing input anywhere in the forward pass (7.75% trainable). Spacing-Conditioned LoRA (r=4) uses the same LoRA injection, but the LoRA output is also modulated by a FiLM layer<sup>6</sup>, with γ and β produced by a small MLP from the 3-value spacing vector. It is zero-initialized so that γ=1 and β=0 at the start (8.88% trainable; 1.25M more parameters than Plain LoRA, which is the same order of magnitude, so model size does not explain the results). SAM-Med3D's decoder always needs a prompt, so all three models prompt the whole patch instead of using a real point or box.

**Training.** We use 64³ voxel patches, AdamW and a learning rate of 1e-4. The loss is Dice + 0.5·clDice<sup>9</sup> + 0.05·(anisotropic gradient penalty weighted per axis by inverse voxel spacing) + 1.0·(class-weighted binary cross-entropy, with the positive weight computed per batch from the foreground fraction and clamped to 50). Because of the anisotropic term, the training loss uses spacing for all three architectures, including Plain LoRA; the difference between them is only in the forward pass. There are two training setups: a first 10-epoch run on CPU, and a controlled 100-epoch retrain on a Kaggle T4 GPU, done to rule out under-training as the cause of the OOD result. For the multi-seed result we ran three 100-epoch trainings per architecture (the original run plus two reruns with fixed seeds).

**Evaluation.** We sample 100 random 64³ patches per in-distribution run and 40 per OOD run. The in-distribution patches come from the training kidneys, so those numbers are not held-out test results. Patches with no annotated foreground are left out of Dice/IoU/clDice/HD95/ASSD, since these metrics are undefined or misleadingly high on an empty target; the number of excluded patches is reported. Metrics are computed with MONAI. For the field-of-view-matched evaluation we took a much larger crop of the OOD volume, sized to cover the same physical area as a 64³ patch at ~50 μm, and downsampled it to 64³. For the structural comparison we skeletonized the predicted and ground-truth masks (`skimage.morphology.skeletonize`), built a 26-connected graph from each skeleton (`networkx`) and compared the number of connected components and branch points.

---

## Data availability

The SenNet+HOA "blood-vessel-segmentation" dataset used in this paper is publicly available on its Kaggle competition page. The extra native-spacing kidney volumes prepared for future work (see Limitations) come from the public Human Organ Atlas mirror `eternalaudrey/hipct_3d` on Hugging Face. No private or restricted data was used. All result files mentioned in the paper (per-seed and per-patch JSON) are in the project repository. `[TODO: add repository URL/DOI once the code release is decided.]`

## Code availability

`[TODO: decide on public release.]` The code, the commands to reproduce every number in the paper, and a dated experiment log are in the project repository, which is currently private.

## Acknowledgements

`[TODO. Possibly: the SenNet+HOA / Human Organ Atlas dataset creators; the authors of SAM-Med3D, LoRA, clDice and FiLM; Kaggle's free GPU tier, which was used for all the 100-epoch training.]`

## Author contributions

`[TODO: e.g. "R.N. designed the study, implemented all methods, ran all experiments, and wrote the manuscript."]`

## Competing interests

`[TODO: probably "The author declares no competing interests."]`

## Ethics statement

`[TODO: probably "Not applicable": only a public, de-identified imaging dataset was used and no new human or animal data was collected. Check the expected wording and whether the original dataset's ethics approval should be cited.]`

## Funding

`[TODO.]`

---

## References

*(Nature numbered style, in order of first appearance in the main text. Full details are also in `PAPER_DRAFT.md`.)*

1. Hu, E. J. *et al.* LoRA: Low-Rank Adaptation of Large Language Models. Preprint at https://arxiv.org/abs/2106.09685 (2021).
2. Wang, H. *et al.* SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images. Preprint at https://arxiv.org/abs/2310.15161 (2023).
3. Zhong, Z., Tang, Z., He, T., Fang, H. & Yuan, C. Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model. In *ICLR* (2024).
4. Iltaf, A., Ahmed, R. M., Zhang, Z., Li, B. & Zhou, S. VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with AtrousLoRA. Preprint at https://arxiv.org/abs/2502.18185 (2025).
5. Garrido-Martínez, J. F., Garrido-López, J. M., Zapata-Pérez, J. F. & Martínez-Alajarín, J. Metadata Conditioning via Feature-Wise Linear Modulation for Multi-Domain 3D Abdominal Segmentation: A Comparative Study Between U-Net and nnU-Net. *Sensors* (2026).
6. Perez, E., Strub, F., de Vries, H., Dumoulin, V. & Courville, A. FiLM: Visual Reasoning with a General Conditioning Layer. In *AAAI* (2018).
7. Joutard, S., Pietsch, M. & Prevost, R. HyperSpace: Hypernetworks for spacing-adaptive image segmentation. In *MICCAI* (2024).
8. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J. & Maier-Hein, K. H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nat. Methods* **18**, 203-211 (2020).
9. Shit, S. *et al.* clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. In *CVPR* (2021).
10. Jain, Y. *et al.* Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys. *Nat. Commun.* **17**, 8273 (2026).

---

## Figure legends

**Figure 1.** Shared frozen backbone and trainable mask decoder, with three interchangeable adaptation mechanisms attached at two points (inside the encoder's attention layers, or between encoder and decoder). Only two of the three receive the spacing signal.

**Figure 2.** The extra step Spacing-Conditioned LoRA adds to Plain LoRA: the low-rank update is modulated by γ and β from a small network conditioned on physical spacing, before scaling and adding to the frozen path. Both the LoRA and FiLM branches are zero-initialized, so training starts from a no-op.

**Figure 3.** The same kidney specimen at two native resolutions. A fixed 64³-voxel window covers ~9.6× less tissue at the OOD spacing (5.2 μm, ≈0.33 mm) than at the training spacing (50 μm, 3.2 mm).

**Figure 4.** In-distribution and out-of-distribution Dice over three training seeds, mean ± s.d. with the individual seeds shown. The panels are separate because the OOD values are about 15× smaller.

**Figure 5.** FiLM γ_max for each of the 24 injected layers at the training spacing and the OOD spacing, showing a uniform return toward the identity value γ=1.

**Figure 6.** Median connected-component and branch-point counts of predictions vs. ground truth in four conditions (two architectures × in-distribution/OOD), from the skeleton-graph analysis.

---

## Table 1. In-distribution and out-of-distribution Dice, 10-epoch exploratory comparison

| Model | Trainable | In-dist Dice | In-dist IoU | In-dist clDice | OOD Dice (5.2 μm) |
|---|---|---|---|---|---|
| Zero-shot SAM-Med3D | 0% | 0.088 | 0.052 | 0.076 | -- |
| PhysicsAwareAdapter | 7.65% | 0.157 | 0.094 | 0.139 | 0.064 |
| Plain LoRA | 7.75% | 0.067 | 0.038 | 0.057 | 0.067 |
| Spacing-Conditioned LoRA | 8.88% | 0.206 | 0.131 | 0.261 | 0.002 |

## Table 2. Field-of-view confound test (OOD data, native-scale vs. FOV-matched crop)

| Model | In-dist Dice (50 μm) | OOD native-scale | OOD FOV-matched |
|---|---|---|---|
| Plain LoRA | 0.1838 | 0.1672 (n=5/40) | 0.2895 (n=40/40) |
| Spacing-Conditioned LoRA | 0.3145 | 0.0058 (n=5/40) | 0.2215 (n=40/40) |

---

## Supplementary Information (moved out of the main text to stay within 8 display items)

**Supplementary Table S1.** clDice ablation (Spacing-Conditioned LoRA with and without clDice, in-distribution and OOD Dice).
**Supplementary Table S2.** Full hyperparameter grid (3 learning rates × 3 loss-α values × 2 architectures, 15 epochs each).
**Supplementary Table S3.** 100-epoch single-run results (replaced by the n=3 result in Figure 4, kept for reference).
**Supplementary Table S4.** Per-patch vascular-graph data behind Figure 6.
**Supplementary Note S1.** Investigation of ASSD=inf in the 10-epoch run, and the separate HD95/ASSD=nan bug in the Kaggle environment (100-epoch run). These are two different problems, both resolved; details in `PAPER_DRAFT.md` Section 4.9.

`[TODO: put S1-S4 together as a Supplementary Information PDF once the venue is confirmed. All numbers are already in PAPER_DRAFT.md Section 4 and the *_results.json files.]`
