# Notes on the related papers

One entry per paper: what they did, the numbers that matter for us, and where
it goes in manuscript.md. PDFs of most of them are in `new/`. Read each paper
yourself before citing it; these notes only tell you what to look for.

Venue/volume details were checked on PubMed or the paper itself on
2026-09-23 unless marked `[check]`.

---

## A. Data and HiP-CT (must read fully)

**Walsh et al. 2021, Nature Methods 18:1532-1541** (`new/s41592-021-01317-x.pdf`)
HiP-CT method. Whole organs at ~25 µm/voxel, then zoomed volumes at 6.5 µm and
2.5-1.3 µm, without cutting. Measured resolution (half-bit): 72 µm at
25 µm/voxel, 18.3 µm at 6.5 µm/voxel, 10.4 µm at 2.5 µm/voxel, so voxel size and
real resolution are not the same thing. Kidney whole-organ scan takes ~3.5 h.
Cite in: Introduction (what HiP-CT is), Data section.

**Walsh et al. 2026, Science Advances 12:eadz2240, "The Human Organ Atlas"**
Open data portal for HiP-CT scans. Cite when saying the data are public.
(I have not read the full text; it was blocked. Read it before citing.)

**Yagis et al. 2024, Scientific Reports 14:27258** (`new/s41598-024-77582-5.pdf`)
First nnU-Net baseline on HiP-CT kidneys (kidney 1, 2, 3 at 50 µm, binned from
25 µm). Leave-one-kidney-out: Dice 0.9410, 0.9523, 0.8585; clDice 0.8886,
0.8533, 0.8228; NSD(t=0) 0.7631, 0.7120, 0.7132. Main errors: collapsed large
vessels and breaks in small vessels. Uses NSD with tolerance 0 because small
vessels are 1-2 voxels wide. Cite in: Related work, Discussion (our numbers are
far below this).

**Jain et al. 2026, Nature Communications 17** (`new/s41467-026-74050-8.pdf`)
The Kaggle challenge paper and our data source. 1,401 participants. Training:
kidney 1 (50 µm + 5.2 µm VOI), 2, 3. Test: kidney 5 (50.28 µm, same donor,
BM05), kidney 6 (63.08 µm, other donor, BM18). Metric NSD tolerance 0. Winner
NSD 0.774 on private test. Public vs private leaderboard rank correlation of
top-50 teams: Kendall's tau 0.0253 (poor generalisation). Teams handled
resolution by rescaling test data (team 3, 5); nobody used the 50 µm / 5.2 µm
pair to their advantage. All top teams fragment small vessels. Extra VOI at
6.5 µm binned to 13 µm tested after the challenge. Data + labels + team
predictions on Zenodo 19685381; weights on Zenodo 19685078.
Cite in: everywhere (data, baselines, metric, motivation).

**Yan et al. 2025, Scientific Reports 15:44887** (not in `new/`, abstract only)
Multi-view hybrid encoder U-Net for HiP-CT renal vessels (three orthogonal 2D
views). Dice 0.939, mean surface Dice 0.852 on kidneys not used in training.
Another strong number on the same data type. Cite in: Related work. Read the
full paper first (which kidneys, which resolution).

---

## B. Foundation models and how they are adapted

**Wang et al. 2023, SAM-Med3D, arXiv:2310.15161** (`new/2310.15161v3.pdf`)
Our backbone. Fully 3D SAM. 22K volumes, 143K masks, 245 categories.
All data resampled to 1.5 mm and cropped to 128³; masks smaller than 1 cm³ were
removed. So it has never seen structures at our scale (50 µm to 5.2 µm).
Patch size 16 -> 8³ tokens at 128³. Cite in: Methods, Discussion.

**Ma et al. 2024, MedSAM, Nature Communications 15:654** (`new/s41467-024-44824-z.pdf`)
2D SAM fine-tuned on 1.57M image-mask pairs, box prompt. Authors say it has
trouble with "vessel-like branching structures because the bounding box
prompt can be ambiguous" (arteries and veins share a box). Cite in: Related
work, and to explain why we use a whole-volume prompt.

**Ma et al. 2025, MedSAM2, arXiv:2504.03600** (`new/2504.03600v1.pdf`)
SAM2.1-Tiny, full fine-tuning on 455K 3D pairs + videos, box on the middle
slice then propagation. Limitation section: box prompts make "vessels with
thin and branching structures" hard, and the model does not explicitly model
3D continuity. They also say full fine-tuning beats partial fine-tuning, which
is a point against PEFT we should mention. Cite in: Related work, Limitations.

**Zhang and Liu 2023, SAMed, arXiv:2304.13785** (`new/2304.13785v2.pdf`)
LoRA (rank 4) on SAM's image encoder + fine-tune prompt encoder and decoder;
Synapse DSC 81.88. Important detail: they upsample 224² CT slices to 512²
before the model "to maintain the decent image resolution of the predicted
segmentation logits". Same problem as our 16³ output grid. Cite in: Related
work, Methods (refinement head motivation).

**Wu et al. 2025, Medical SAM Adapter, Medical Image Analysis 102:103547**
(`new/2304.12620v7.pdf`; note the published author list differs from arXiv:
Wu, Wang, Hong, Ji, Fu, Xu, Xu, Jin)
Adapters in 2D SAM, space-depth transpose for 3D, hyper-prompting adapter,
2% of parameters, 17 tasks. Cite in: Related work.

**Chen et al. 2024, MA-SAM, Medical Image Analysis 98:103310** (`new/2309.08842v1.pdf`)
FacT low-rank updates + 3D adapters in SAM, full decoder fine-tuning, extra
up-sampling layers to recover output resolution, no prompts. Beats nnU-Net by
0.9%, 2.6% and 9.9% Dice on CT multi-organ, MRI prostate and surgical video.
So PEFT of SAM *can* match nnU-Net. Cite in: Related work, Discussion.

**Asokan et al. 2024, FLAP-SAM, arXiv:2407.21739** (`new/2407.21739v1.pdf`) `[check venue]`
Federated PEFT of SAM for 3D (Fed-KiTS). Findings: fine-tuning the whole
decoder on small data "tends to distort the inherent capabilities" of the
model; LoRA + only the decoder output layers works as well with far fewer
parameters. Also adds two transposed-conv layers to SAM's decoder to upsample.
Cite in: Related work, Discussion (we fine-tune the whole decoder).

**Hu et al. 2026, DINO-Med3D, MICCAI 2026, arXiv:2606.18886** (`new/2606.18886v1.pdf`)
Adapts DINOv3 to 3D with 3D adapters and a "detail-recovery" high-resolution
stream to put back the high-frequency detail lost in patch embedding. Same idea
as our refinement head. Cite in: Methods / Related work.

**Zhong et al. 2024, Conv-LoRA, ICLR 2024** (not in `new/`) `[read it]`
Adds convolution to LoRA in SAM's encoder. Cite in Related work only after
reading.

**Iltaf et al. 2025, VesselSAM, arXiv:2502.18185** (`new/2502.18185v4.pdf`)
LoRA + atrous attention in SAM for the aorta. 2D slices resampled to 1 mm,
bounding-box prompts, Dice ~93%. Large vessel, 2D, box prompt, so not
comparable to our thin 3D vessels. Cite in: Related work.

**Dutt et al. 2024, MIDL (PMLR 250), arXiv:2305.08252** (not in `new/`)
Benchmark of 17 PEFT methods in medical imaging (classification and
generation, not segmentation). LoRA best for ViTs; PEFT beats full
fine-tuning with little data. Cite in: Related work (one sentence).

**Hu et al. 2021, LoRA, arXiv:2106.09685** `[read method section]`
**Perez et al. 2018, FiLM, AAAI** `[read method section]`

---

## C. Voxel spacing and resolution

**Joutard et al. 2024, HyperSpace, MICCAI 2024, arXiv:2407.03681** (`new/2407.03681v1.pdf`)
Hypernetwork (MLP, 3 hidden layers) makes all U-Net weights from the spacing.
Every training batch is resampled to a random spacing from the dataset range.
BRATS, SPIDER, MM-WHS. As good as fixed-spacing models inside the range; also
drops sharply at the highest resolutions (they blame the U-Net's depth /
receptive field) and extrapolation beyond the training range drops on 2 of 3
datasets. Resampling to the closest training resolution at test time fixes
it. Our results agree with this. Cite in: Introduction, Related work,
Methods (spacing augmentation), Discussion.

**Isensee et al. 2021, nnU-Net, Nature Methods 18:203-211** (`new/1809.10486v1.pdf` is the 2018 arXiv version; cite the journal version)
"CNNs do not natively understand voxel spacings", so all images are resampled
to the median spacing. Epoch = 250 batches; at least 1/3 of each batch contains
foreground; augmentation includes random scaling. Cite in: Introduction,
Methods (our training choices follow it), baseline.

**Billot et al. 2023, SynthSeg, Medical Image Analysis 86:102789** (not in `new/`)
Brain MRI segmentation robust to any contrast and resolution, by training on
synthetic images with randomised contrast and resolution (domain
randomisation); at test time images are resampled to 1 mm. Robustness comes
from randomising the resolution in training, not from telling the network the
resolution. Their resolution changes are towards coarser data; ours is the
opposite (finer). Cite in: Related work, Discussion.

**Sangalli et al. 2022, Scale Equivariant U-Net, arXiv:2210.04508** (not in `new/`) `[check venue]`
U-Net made approximately equivariant to scale; generalises to unseen scales
much better than a U-Net trained with scale jittering (2D pets and cells).
Cite in: Related work (architectural route to scale robustness).

**Hoopes et al. 2024, VoxelPrompt, arXiv:2410.08397** (`new/2410.08397v2.pdf`)
Vision-language agent; its vision network works at native resolution by
tracking voxel spacing through the network levels instead of resampling.
Only loosely related. Optional one-sentence citation.

**You et al. 2026, Consispace, arXiv:2606.31839** (`new/2606.31839v1.pdf`)
Learned resampling to make axial slice spacing consistent before
segmentation. Loosely related (preprocessing). Optional.

---

## D. Vessels and topology

**Wittmann et al. 2025, vesselFM, CVPR 2025:20874-20884** (`new/2411.17386v2.pdf`)
Vessel foundation model (real + domain-randomised + generated data, MONAI
UNet). Training data include HiP-CT kidney at 2.5 µm. Zero-shot Dice 29.7-74.7
on four unseen sets; SAM-Med3D zero-shot only 2.1-7.9 Dice there. Domain
randomisation includes random zoom (scale). Baseline for us (not strictly
zero-shot on HiP-CT).

**Wittmann et al. 2026, vesselFM-CT, arXiv:2606.09400** (`new/2606.09400v1.pdf`)
All vessels in CT; TubeLoss/TubeDice for vessels of very different radii;
multi-scale test-time augmentation (1x, 1.33x, 2x). Optional citation.

**Shit et al. 2021, clDice, CVPR** `[read]`
Our loss and metric.

---

## E. Distribution shift

**Karani et al. 2021, Medical Image Analysis 68:101907**
Test-time adaptation of a small normalisation network per test image, guided
by a denoising autoencoder prior; brain, heart, prostate MRI. They also
resample to fixed voxel sizes. Cite in: Related work (other ways to handle
shift).

**Su et al. 2024, arXiv:2411.05824** and **Li and Fan 2023, arXiv:2311.01702**
Surveys of distribution shift / domain adaptation in medical imaging. One
citation each is enough.

---

## Papers I would NOT cite
- TopoLoRA-SAM, MorVess: could not be found.
- Anything you have not opened and read yourself.
