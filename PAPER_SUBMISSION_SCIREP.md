# Spacing-conditioned LoRA improves in-distribution vessel segmentation but collapses out-of-distribution

**Status: DRAFT, formatted for *Scientific Reports* (Nature Portfolio) submission requirements** (checked live against nature.com's submission guidelines, cross-verified against an independent mirror, 2026-09-16): unstructured abstract ≤200 words; main text ≤4,500 words excluding Abstract/Methods/References/figure legends; ≤8 figures+tables combined in the main text; Nature numbered reference style. This is a derived, trimmed version of the full working draft (`PAPER_DRAFT.md`, which keeps every table and stays the source of truth for all numbers) -- nothing here is a new claim, only a reorganization for one specific venue's rules. `[NEEDS AUTHOR INPUT]` markers are unchanged from the working draft's meaning: a fabricated answer would be worse than a flagged gap.

Author(s): `[NEEDS AUTHOR INPUT -- confirm name(s), affiliation(s), and order. This session only has one identifiable person on the project (git/Kaggle account "rabeya100x"; email on file is rabeyanoor.dev@gmail.com) -- if that is you and this is solo work, "Rabeya Noor" as sole author is a reasonable placeholder; correct it if wrong.]`

---

## Abstract *(191 words, verified by direct count; no citations, per the format rule)*

Parameter-efficient fine-tuning (PEFT) adapts frozen 3D medical foundation models to new anatomy, but physical voxel spacing -- which varies by orders of magnitude -- is rarely an explicit signal to the adapter. We introduce Spacing-Conditioned LoRA, a low-rank adaptation of a frozen 3D segmentation backbone whose LoRA branch is modulated by a FiLM layer conditioned on voxel spacing, compared against a matched Plain LoRA baseline (no spacing signal) on real expert-annotated kidney vasculature. Across three independent runs, Spacing-Conditioned LoRA wins in-distribution twice (mean Dice 0.272 vs. 0.184), but collapses at a real, unseen voxel spacing (5.2 μm vs. ~50 μm training data): out-of-distribution Dice falls below 0.02 in all three runs, while Plain LoRA degrades gracefully and wins out-of-distribution every time. Three independent analyses -- a field-of-view-matched re-test, a per-layer activation breakdown, and a vascular-graph structural comparison -- show the collapse is not a training-budget artifact, is substantially explained by field-of-view mismatch rather than the spacing signal itself, and manifests as fragmented, over-branched predictions rather than a weaker version of the true vessel tree. We report this as a specific, reproducible caution about a natural-seeming design choice, not a proposed fix.

---

## Introduction

Three-dimensional vessel segmentation in high-resolution computed tomography is important for quantitative analysis of vascular morphology and tissue structure, but vessel appearance can change substantially with physical voxel spacing. A model trained at one spatial scale may experience a severe distribution shift when applied to data acquired at a different resolution -- a challenge particularly relevant to high-resolution ex-vivo imaging, where a single specimen may exist as multiple volumes at different native physical samplings.

Recent 3D medical foundation models offer a starting point for this problem by transferring representations learned from large-scale imaging data, but full fine-tuning of such models is often computationally expensive. Parameter-efficient fine-tuning (PEFT) adapts a small trainable parameter subset instead, and low-rank adaptation (LoRA)<sup>1</sup> is a widely used PEFT mechanism, including for the Segment Anything Model (SAM) family in medical settings<sup>2,3,4</sup>. A limitation of conventional PEFT is that physical voxel spacing is rarely an explicit conditioning input, so two volumes of similar anatomy at very different physical resolutions are adapted identically despite different image-scale characteristics.

We investigate spacing-aware PEFT of a frozen 3D medical foundation model (SAM-Med3D<sup>2</sup>) for kidney vessel segmentation, introducing a spacing-conditioned LoRA mechanism that modulates the low-rank pathway using physical voxel spacing via a FiLM layer<sup>5,6</sup>. We compare it against zero-shot and spacing-blind PEFT baselines on real expert-annotated data, and -- rather than assuming in-distribution gains imply resolution robustness -- explicitly test that assumption with a genuine, same-specimen resolution shift (50 μm training data; a held-out 5.2 μm volume of the same physical kidney). Related work on spacing-adaptive segmentation<sup>7</sup> and metadata-conditioned FiLM segmentation<sup>5</sup> provides the closest prior mechanisms; neither reports the severe held-out-resolution collapse found here (see Related Work). Our central contribution is therefore not a method we recommend adopting as-is, but a controlled, three-seed, three-angle investigation of a specific, reproducible failure mode in a design that looks reasonable on paper.

---

## Related Work

**Segmentation backbone and reference baseline.** nnU-Net<sup>8</sup> remains the field's self-configuring reference segmentation method; we do not implement it as a direct comparison here (see Limitations) but cite it as the standard any PEFT-based approach should eventually be measured against. SAM-Med3D<sup>2</sup> is the frozen 3D backbone this work adapts.

**PEFT of segmentation foundation models.** LoRA<sup>1</sup> is the low-rank mechanism both our baseline and proposed method build on. Conv-LoRA<sup>3</sup> injects convolutional inductive bias into LoRA for SAM's encoder -- a different modification to the same starting point, aimed at image-domain bias rather than acquisition metadata. VesselSAM<sup>4</sup> is the closest prior work in both domain and mechanism, adapting SAM with a LoRA variant for aortic vessel segmentation; it targets a different vascular bed and does not condition on physical voxel spacing.

**Spacing- and metadata-conditioned segmentation.** HyperSpace<sup>7</sup> conditions a hypernetwork on voxel spacing so one model can process images natively across resolutions -- the same underlying premise as this work, via full-weight hypernetworks rather than a FiLM-modulated LoRA branch. HyperSpace's own evaluation does not report the severe held-out-resolution collapse found here; whether that reflects the different conditioning mechanism, a different resolution range, or how "unseen resolution" was tested is an open question, and a direct comparison against HyperSpace on our own data is a natural next step, not yet performed. Garrido-Martínez et al.<sup>5</sup> apply FiLM-based conditioning on case-level metadata (not physical spacing) for 3D abdominal segmentation and find it helps a plain U-Net only modestly and does not consistently help an already-optimized nnU-Net -- a related caution about FiLM-based conditioning not reliably paying off, worth discussing alongside our own OOD collapse rather than as an unrelated result. FiLM itself<sup>6</sup> is the general conditioning mechanism both works apply to segmentation.

**Topological loss.** clDice<sup>9</sup> is the topology-preserving loss term used in our training objective.

`[NEEDS AUTHOR INPUT: a fully thorough submission should still add broader domain-shift/domain-generalization literature in medical segmentation and any further SAM-for-3D-medical variants -- see PAPER_DRAFT.md Section 2's fuller note for what specifically remains, including one survey found but not yet independently verified.]`

---

## Results

### In-distribution gain (Table 1; 10-epoch exploratory comparison)

At 10 epochs (CPU-only exploratory run), Spacing-Conditioned LoRA outperformed every comparator on every in-distribution metric, including its most direct ablation, Plain LoRA (identical architecture, no spacing signal) -- see Table 1. Plain LoRA notably underperformed the untrained zero-shot baseline at this epoch count; all trained runs shared un-retuned loss weights (addressed later, Hyperparameter sensitivity below).

### Out-of-distribution collapse (Table 1; the paper's central finding)

The same three models evaluated at a genuine, held-out ~10× unseen resolution (5.2 μm, same physical specimen as the 50 μm training data) show the in-distribution ranking reversed: Spacing-Conditioned LoRA, the clear in-distribution winner, collapsed to near-zero Dice out-of-distribution (0.002), worse than both comparators (Table 1).

### The collapse is not a training-budget artifact

A controlled 100-epoch GPU retrain of both LoRA variants sharpened both effects rather than resolving them: in-distribution Dice for Plain LoRA and Spacing-Conditioned LoRA reached 0.1838 and 0.3145 respectively, while OOD Dice reached 0.1672 and 0.0058 -- ten times more training did not shrink the gap between architectures OOD; if anything it widened.

### The collapse replicates across independent seeds (Figure 4)

Three independent 100-epoch training runs per architecture (the original run plus two additionally-seeded reruns) confirm the pattern is not a single-run fluke (Figure 4). Spacing-Conditioned LoRA's OOD Dice was below 0.02 in all three runs (mean 0.0102 ± 0.0079), and Plain LoRA beat it OOD in all three (mean 0.0990 ± 0.0591). The in-distribution ranking, ambiguous with only two runs, resolves toward Spacing-Conditioned LoRA at three runs (mean 0.2718 ± 0.0464 vs. 0.1838 ± 0.0644, winning 2 of 3 individual seeds by a wide margin) -- still a small-n result, not reported as a formal significance claim.

### Field-of-view mismatch explains most of the collapse (Table 2)

Because a fixed 64³-voxel window covers far less real tissue at 5.2 μm than at 50 μm, we tested whether the OOD failure reflects the spacing value itself or simply this coupled field-of-view (FOV) shrinkage. Re-cropping the same real OOD data to match the training FOV (Table 2) recovered Spacing-Conditioned LoRA's Dice from 0.0058 to 0.2215 -- about 70% of its own in-distribution score -- and every one of 40 FOV-matched patches contained real vessel, versus 35 of 40 empty at native scale. A significance check on an independent 40-patch redraw per model (Plain LoRA 0.3681, Spacing-Conditioned LoRA 0.2618) gave a Mann-Whitney U p=0.0216. Most of the apparent "spacing" collapse looks like an ordinary unseen-visual-scale shift.

### FiLM reverts toward identity at the unseen spacing (Figure 5)

Across all 24 injected layers, the FiLM modulation's γ parameter grows large and layer-specific at the training spacing after training (e.g. layer 19: γ_mean = 6.52 at 50 μm) but reverts toward its identity initialization at the unseen spacing (same layer: γ_mean = 1.59 at 5.2 μm) -- uniformly, not from one outlier layer (Figure 5). This is a second, complementary lead: training appears to make the model reliant on strong, training-spacing-specific modulation that is not available at test time OOD.

### The failure is structurally different, not just quantitatively worse (Figure 6)

A skeleton-graph structural comparison, independent of Dice, shows Plain LoRA's OOD predictions fragment moderately but keep branch-point counts close to real ground truth (graceful degradation), while Spacing-Conditioned LoRA's OOD predictions fragment 11× worse and hyper-branch 31× worse than ground truth (Figure 6) -- a scattered field of disconnected, over-branched fragments, not a smaller version of the true vessel tree.

### Ablation and hyperparameter sensitivity (Supplementary Tables S1-S2)

Removing the clDice loss term drops Spacing-Conditioned LoRA's in-distribution Dice ~40% relative but does not rescue or worsen the OOD collapse (both catastrophic; Supplementary Table S1). A full learning-rate × loss-weight grid found the best configuration shifts with training length and does not change the qualitative OOD finding (Supplementary Table S2); headline results above used loss weights tuned before this sweep existed, a stated limitation.

---

## Discussion

Two complementary explanations account for most of the observed collapse, neither alone sufficient. Field-of-view mismatch (Table 2) explains most of the raw magnitude and affects both architectures, Plain LoRA less severely. FiLM's spacing-specific modulation degenerating toward, but not cleanly reaching, Plain-LoRA-like behavior (Figure 5) plausibly explains why Spacing-Conditioned LoRA is hit harder by whatever FOV-shift remains after correction. The vascular-graph evidence (Figure 6) adds a third, structurally distinct signature consistent with both explanations: fragmentation and spurious branching, not graceful degradation.

Explicitly conditioning a PEFT adapter on physical spacing metadata is a natural-seeming design choice that delivers a real, three-seed-confirmed in-distribution gain. It appears to buy that gain by making the adapter more reliant on the specific training-distribution spacing value, not more robust to spacing variation generally -- the opposite of what the design intuitively promises. We report this as a specific, reproducible caution for anyone conditioning a segmentation adapter on acquisition metadata, not as a claim against spacing-conditioning as a concept; HyperSpace's different conditioning mechanism<sup>7</sup> and different reported robustness make a direct head-to-head comparison a natural next step.

**Limitations.** Single organ (kidney), single dataset; no cross-organ or multi-institution validation. Only two native training/test spacings (50 μm; 5.2 μm held out), so unseen-spacing and unseen-FOV are not yet fully disentangled by a third, intermediate native spacing -- real data for this exists and has been staged (three further native spacings of the same specimen) but is not expert-annotated, and two of the three sit close enough to the OOD test spacing that training on them risks weakening that test's validity; this design decision is not yet resolved. No direct comparison against nnU-Net or other published foundation-model baselines. n=3 seeds supports "the collapse is not a fluke" and "the in-distribution ranking leans one way," not a formally powered estimate. No clinical validation; no claim of clinical usefulness is made. Loss hyperparameters were not independently re-tuned per architecture for the headline results.

---

## Methods

**Data.** Real expert-annotated kidney vasculature from the SenNet+HOA "blood-vessel-segmentation" dataset<sup>10</sup>. Voxel spacing is not embedded in the source files and was taken from the dataset's companion publication's Table 1, not inferred. Training used four ~50 μm volumes (`kidney_1_dense`, `kidney_2`, `kidney_3_dense`, `kidney_3_sparse`); `kidney_1_voi` -- the same physical specimen as `kidney_1_dense` at 5.2 μm -- was held out entirely from training as the OOD test set, giving a genuine same-organ, same-annotation-protocol resolution shift.

**Architectures.** All three compared models freeze a SAM-Med3D<sup>2</sup> image encoder and always train the mask decoder; they differ only in adaptation mechanism. PhysicsAwareAdapter: a post-encoder bottleneck adapter gated multiplicatively by a spacing embedding (7.65% trainable). Plain LoRA (r=4): low-rank updates in the encoder's attention `qkv`/`proj` layers, zero-initialized to a no-op, no spacing signal anywhere in its forward pass (7.75% trainable). Spacing-Conditioned LoRA (r=4): identical LoRA injection, with the LoRA branch's output additionally modulated by a FiLM layer<sup>6</sup> (γ, β from a small MLP over the 3-number spacing vector), zero-initialized so γ=1, β=0 at the start of training (8.88% trainable; +1.25M parameters over Plain LoRA -- the same order of magnitude, so capacity does not explain the results). All three auto-prompt the entire patch (no real point/box prompt) since SAM-Med3D's decoder has no prompt-free mode.

**Training.** 64³ voxel patches, AdamW, learning rate 1e-4, loss = Dice + 0.5·clDice<sup>9</sup> + 0.05·(anisotropic gradient penalty weighted by inverse voxel spacing per axis) + 1.0·(class-weighted binary cross-entropy, per-batch positive weight from the actual foreground fraction, clamped to 50) -- note the anisotropic term means all three architectures' training losses use spacing metadata, even Plain LoRA's; the architectural distinction is about the forward pass, not the objective. Two regimes: an initial 10-epoch run (CPU), and a controlled 100-epoch retrain (Kaggle T4 GPU) once the OOD finding needed checking against an under-training explanation. Three independent 100-epoch runs per architecture for the multi-seed result (the original run plus two explicitly-seeded reruns).

**Evaluation.** 100 random 64³ patches per in-distribution run, 40 per OOD run; patches with zero annotated foreground excluded from Dice/IoU/clDice/HD95/ASSD (excluded counts reported, not hidden) since these metrics are undefined or misleadingly favorable on an empty target. Metrics via MONAI. Field-of-view-matched evaluation used a much larger native-scale crop of the OOD volume, physically sized to match a 64³ patch at ~50 μm, downsampled to 64³. Structural comparison skeletonized (`skimage.morphology.skeletonize`) predicted and real-GT masks and built a 26-connected graph over each skeleton (`networkx`) to compare connected-component and branch-point counts.

---

## Data availability

The SenNet+HOA "blood-vessel-segmentation" dataset (this paper's primary data) is publicly available via its Kaggle competition page. Additional native-spacing kidney volumes staged for future work (Limitations) are from the public Human Organ Atlas mirror `eternalaudrey/hipct_3d` on Hugging Face. No private or restricted-access data was used. All result files (raw per-seed and per-patch JSON) referenced in this paper are available in the project repository. `[NEEDS AUTHOR INPUT: insert final repository URL/DOI once code availability, below, is resolved.]`

## Code availability

`[NEEDS AUTHOR INPUT: decide on public release.]` Code, exact commands to reproduce every number in this paper, and a dated experimental log are maintained in the project repository (currently private/local). No git identity or GitHub authentication is yet configured in the development environment used -- releasing publicly requires the author to set both up directly.

## Acknowledgements

`[NEEDS AUTHOR INPUT. Consider: the SenNet+HOA/Human Organ Atlas dataset creators; the SAM-Med3D, LoRA, clDice, and FiLM authors whose code/methods this work builds on; Kaggle's free GPU tier, on which all 100-epoch training in this paper ran.]`

## Author contributions

`[NEEDS AUTHOR INPUT -- state in your own words; single-author placeholder if applicable: "R.N. designed the study, implemented all methods, ran all experiments, and wrote the manuscript."]`

## Competing interests

`[NEEDS AUTHOR INPUT -- likely: "The author declares no competing interests," if accurate. Confirm before using.]`

## Ethics statement

`[NEEDS AUTHOR INPUT -- likely "Not applicable": this work uses only a pre-existing, publicly released, de-identified imaging dataset; no new human or animal data was collected. Confirm exact wording expected and whether the original dataset's own ethics approval should be cited.]`

## Funding

`[NEEDS AUTHOR INPUT.]`

---

## References

*(Nature numbered style, ordered by first appearance in this document's main text; full details also in `PAPER_DRAFT.md`'s longer reference list.)*

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

**Figure 1.** Shared frozen backbone and trainable mask decoder, with three interchangeable adaptation mechanisms attaching at two points (inside the encoder's attention layers, or between encoder and decoder); only two of the three receive the physical spacing signal.

**Figure 2.** The one additional step Spacing-Conditioned LoRA adds over Plain LoRA: the low-rank update is modulated by γ,β from a small network conditioned on physical spacing, before scaling and summing with the frozen path. Both the LoRA and FiLM branches are zero-initialized to a true no-op start.

**Figure 3.** The same physical kidney specimen at two native resolutions; a fixed 64³-voxel window covers ~9.6× less real tissue at the OOD spacing (5.2 μm, ≈0.33 mm) than at the training spacing (50 μm, 3.2 mm).

**Figure 4.** In-distribution and out-of-distribution Dice across three independent training seeds, mean ± s.d. with individual seed values shown; separate panels because the OOD scale is roughly 15× smaller.

**Figure 5.** FiLM γ_max per injected layer (all 24), training spacing vs. OOD spacing, showing uniform reversion toward the γ=1 identity value.

**Figure 6.** Median predicted vs. real ground-truth connected-component and branch-point counts across four conditions (two architectures × in-distribution/OOD), from skeleton-graph structural analysis.

---

## Table 1. In-distribution and out-of-distribution Dice, 10-epoch exploratory comparison

| Model | Trainable | In-dist Dice | In-dist IoU | In-dist clDice | OOD Dice (5.2 μm) |
|---|---|---|---|---|---|
| Zero-shot SAM-Med3D | 0% | 0.088 | 0.052 | 0.076 | -- |
| PhysicsAwareAdapter | 7.65% | 0.157 | 0.094 | 0.139 | 0.064 |
| Plain LoRA | 7.75% | 0.067 | 0.038 | 0.057 | 0.067 |
| Spacing-Conditioned LoRA | 8.88% | 0.206 | 0.131 | 0.261 | 0.002 |

## Table 2. Field-of-view confound test (real OOD data, native-scale vs. FOV-matched crop)

| Model | In-dist Dice (50 μm) | OOD native-scale | OOD FOV-matched |
|---|---|---|---|
| Plain LoRA | 0.1838 | 0.1672 (n=5/40) | 0.2895 (n=40/40) |
| Spacing-Conditioned LoRA | 0.3145 | 0.0058 (n=5/40) | 0.2215 (n=40/40) |

---

## Supplementary Information (moved out of main text to stay within the 8-display-item cap)

**Supplementary Table S1.** clDice loss-term ablation (Spacing-Conditioned LoRA, with vs. without clDice, in-dist and OOD Dice).
**Supplementary Table S2.** Full hyperparameter grid (3 learning rates × 3 loss-α values × 2 architectures, 15 epochs each).
**Supplementary Table S3.** 100-epoch single-run breakdown (superseded by the n=3 result, Figure 4; kept for the original single-run numbers).
**Supplementary Table S4.** Full per-patch vascular-graph data underlying Figure 6.
**Supplementary Note S1.** ASSD=inf root-cause investigation (10-epoch run) and the separate HD95/ASSD=nan Kaggle-environment bug (100-epoch run) -- two different anomalies, both resolved, detailed in `PAPER_DRAFT.md` Section 4.9.

`[NEEDS AUTHOR INPUT: assemble these four items as an actual Supplementary Information PDF/document once a venue is confirmed -- the source numbers for all four already exist in PAPER_DRAFT.md Section 4 and the project's *_results.json files; this is compilation, not new analysis.]`
