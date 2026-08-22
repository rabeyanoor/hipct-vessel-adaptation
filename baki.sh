HiP-VesselAdapt -- remaining work (todo list, not a script)
================================================================
Things that are still not done. For what is done (with numbers) see
README.md / test.sh. Keeping everything here so nothing gets lost when I
come back to it.

================================================================
1. DONE (2026-09-15)
================================================================
- Reduced hyperparameter sweep (kaggle_kernels/kaggle_sweep_kernel.py,
  kernel rabeya100x/hip-vesseladapt-hp-sweep on Kaggle GPU) finished and
  results downloaded: sweep_results.json has all 12 configs (3 LR x
  2 loss_alpha x {plain_lora, spacing_lora}, 5 epochs each). Best:
  spacing_lora, lr=1e-4, loss_alpha=0.25, dice=0.1364, clearly best of
  the 12, and every spacing_lora config beat every plain_lora config.
  lr=1e-4 is already the TRAIN_CONFIG default, but loss_alpha=0.25 is
  NOT the default in train_spacing_lora.py (loss_alpha=0.5), so the full
  3x3 sweep is worth running.
- Kaggle API/CLI note for next time: the NEWER methods of the `kaggle` pip
  package (v2.2.4) -- `KaggleApi.kernels_status()`, `.kernels_list()`,
  and the CLI `kaggle datasets version` / `kaggle kernels push` -- go
  through the internal kagglesdk client, which uses a BEARER token from the
  `KAGGLE_API_TOKEN` env var or `~/.kaggle/access_token`, NOT the
  (username, key) basic auth in `~/.kaggle/kaggle.json`. With only
  kaggle.json (what `KaggleApi.authenticate()` reads by default) all of
  those calls return 401, EVEN IF the key in kaggle.json is valid. Lost a
  lot of time thinking it was a token-scope problem. Fix: put a valid
  KGAT_ token in `~/.kaggle/access_token` (plain text, no quotes); the
  "key" in kaggle.json and access_token can be the same new token. OLDER
  methods (`dataset_list()`, `model_list()`) and the old REST endpoints
  (`https://www.kaggle.com/api/v1/kernels/status`, `.../kernels/output`)
  work fine with kaggle.json basic auth; only the kagglesdk path needs
  access_token. `.../kernels/output` returns
  {"log": <JSON-encoded STRING, not a list -- json.loads() it first>,
  "files": [{"fileName", "url"}, ...]}; each "url" is a short-lived
  signed URL, just GET it (no auth) for the file you need instead of
  printing the whole `files` list (228 entries last time).

================================================================
2. SCRIPTS WRITTEN -- STATUS OF RUNS
================================================================
- Multi-seed results: DONE 2026-09-16, n=3 (the planned number).
  colab_run.py's run_multiseed() (3 seeds x 2 architectures x 100 epochs)
  needs more GPU hours than one account's weekly quota (my rabeya100x
  quota was already used up, see section 1), so I ran it as two Kaggle
  kernels instead (kaggle_kernels/kaggle_multiseed_kernel.py seed=1, and
  kaggle_multiseed_kernel_seed2.py seed=2) on a second Kaggle account
  (nargisakterr). Both fit in that account's weekly quota (seed=1 used
  ~9788s of 21600s, so seed=2 could run the same day). Together with the
  existing 100-epoch checkpoint (seed not set) that gives n=3 per
  architecture -- numbers and discussion in test.sh section 3.11. Summary:
  the OOD collapse happens in all 3 seeds, so it is not a one-off; the
  in-distribution ranking, unclear at n=2, now leans toward
  Spacing-Conditioned LoRA (2 of 3 seeds, both by a wide margin). Still
  only n=3, no formal significance claim. Per-seed numbers in
  multiseed_seed1_results.json / multiseed_seed2_results.json. No more
  seeds planned for now.
- Full hyperparameter sweep: DONE 2026-09-15 (kernel
  rabeya100x/hip-vesseladapt-full-hp-sweep, COMPLETE). Results in
  full_sweep_results.json (18 runs: 3 LRs x 3 loss_alphas x 2 architectures,
  15 epochs each). **This REVERSES the reduced sweep**: at 15 epochs all 18
  configs are close (Dice 0.126-0.172), and Plain LoRA is best overall
  (lr=2e-4, loss_alpha=0.5, Dice=0.1724). The "loss_alpha=0.25,
  spacing_lora wins" result from the reduced sweep does not hold with more
  epochs. Best loss_alpha at 15 epochs is 0.5-0.75, not 0.25. If
  retraining from sweep results, use the full sweep, not the reduced one.
  NOTE: this used up the rest of the week's Kaggle GPU quota --
  `api.quota_view()` (works now with KAGGLE_API_TOKEN set) showed 42814s
  used of the 21600s weekly allowance, i.e. ~5.9h OVER, resets
  2026-09-19. No more Kaggle GPU kernels until then; use Colab if GPU is
  needed before that.
- Vascular graph validation (evaluate_vascular_graph.py, utils/vascular_
  graph_metrics.py): RUN 2026-09-15, locally on CPU (no GPU needed, it is
  only skeletonization + graph stats). Run on both 100-epoch GPU
  checkpoints, in-distribution and OOD, same protocol as test.sh sections
  3.2-3.4. Write-up in test.sh section 3.10, per-patch numbers in
  vascular_graph_results.json. Summary: a THIRD separate piece of evidence
  for the OOD collapse (besides 3.5 FOV confound and 3.6 FiLM), and the two
  architectures fail OOD in different ways, not just by different amounts.
  Plain LoRA's OOD predictions fragment a bit but keep branch-point counts
  close to GT; Spacing-Conditioned LoRA's OOD predictions are 11x more
  fragmented and have 31x more branch points than GT, i.e. scattered
  disconnected pieces instead of the vessel tree, not a smaller version of
  it. Removed from section 3 below (it said "no script exists", but the
  script was already there, just not run).

================================================================
3. IN PROGRESS -- DATA READY, TRAINING SETUP + DECISION NEEDED
================================================================
!! UPDATE 2026-09-23: DO NOT train on the 6.5um and 2.6um HOA volumes below.
   The 6.5um VOI-02 is the same scan as the 13um test volume released by
   Jain et al. (kidney_additional_voi_13um), and kidney_1_voi (5.2um) is very
   likely the 2.6um VOI-01.1 binned 2x. Both are test data now (section 7).
- Train on a wider range of real spacings. CORRECTION 2026-09-16: I had
  marked this as "no script yet", but I had already started it earlier
  (config.py's HOA_KIDNEY_MULTISPACING_CONFIG, utils/
  convert_hoa_parquet_to_memmap.py) and forgot about it until I went
  through the whole codebase again. Current status:
    - DATA DOWNLOADED: 3 more native spacings (25um, 6.5um, 2.6um) of the
      SAME specimen as kidney_1_dense/kidney_1_voi (LADAF-2021-17-Right),
      from the Human Organ Atlas HF mirror (eternalaudrey/hipct_3d).
      Parquet shards are in data/raw/hoa_kidney_right_shards/ (~1.4GB,
      gitignored).
    - CONVERTED 2026-09-16: ran utils/convert_hoa_parquet_to_memmap.py,
      all 3 volumes are now memmaps in data/memmap/ (2176x224x224 uint16
      each, 0.22GB). Has to be run as
      `python3 -m utils.convert_hoa_parquet_to_memmap` (plain
      `python3 utils/convert_hoa_parquet_to_memmap.py` fails on the
      `from config import ...` line -- run it as a module, or with the
      project root on PYTHONPATH).
    - DATASET: NO CHANGES NEEDED -- utils/dataset.py's
      MultiOrganHiPCTDataset takes any config dict
      (`MultiOrganHiPCTDataset(config=HOA_KIDNEY_MULTISPACING_CONFIG, ...)`
      works as is). But no training script uses this config yet; that part
      still has to be written.
    - OPEN PROBLEM: this data has NO expert GT (see the comment in
      config.py: Frangi pseudo-labels only, must NOT be mixed
      quantitatively with the RealGTKidneyDataset numbers, same rule as
      README.md section 6). It could be used to train over a wider spacing
      range (pseudo-labels) and still be evaluated on the REAL GT OOD test
      (kidney_1_voi, 5.2um). But two of the three new spacings (6.5um,
      2.6um) are very close to the 5.2um test point. Training on them
      before testing there makes the OOD test weaker (the spacing would
      not really be unseen anymore, even though the GT is still held out).
      Need to decide first (e.g. drop the two closest spacings and use
      only 25um as the "third point", or keep all three and call it
      "interpolation" instead of "extrapolation") before training on it.
      NOT done yet.

================================================================
4. OUT OF SCOPE (decided earlier)
================================================================
- More baselines: nnU-Net, VesselFM, VesselSDF, HyperSpace-style
  baseline. Not implemented, not planned until the OOD question is
  clearer.
- Cross-organ generalization (pseudo-label organs vs. real-GT kidney) --
  must not be mixed quantitatively (README.md section 6), not attempted.
- Few-shot experiments.
- CollapseDetectionHead training/validation (in models/collapse_head.py,
  never trained or evaluated, postponed earlier).

================================================================
5. WRITING / PUBLISHING
================================================================
- Paper: DRAFT WRITTEN 2026-09-16 in PAPER_DRAFT.md; Introduction/
  Related Work/Conclusion finished the same day. Has Abstract/Intro/
  Related Work/Methods/Results/Discussion/Conclusion/Limitations, all
  numbers from test.sh. 10 citations, each checked (title/authors/venue)
  -- related-work search covered nnU-Net, LoRA-on-SAM variants,
  HyperSpace, FiLM metadata conditioning. Some names I came across early
  on (VesselFM/TopoLoRA-SAM/MorVess/HyperSpace) I couldn't find at first;
  HyperSpace (MICCAI 2024) and vesselFM (CVPR 2025) turned out to be real
  papers and are now used (see PAPER_MEDIA.md); TopoLoRA-SAM and MorVess
  still couldn't be found, so they are left out. Remaining
  `[TODO]`s: venue (affects formatting/cover letter), the wider-spacing
  decision, public release, and a few more related-work citations (see
  the note at the end of section 2 in PAPER_DRAFT.md). Needs a full
  read-through before sending anywhere.
- Journal version: DONE 2026-09-16, PAPER_SUBMISSION_SCIREP.md.
  Venue: Scientific Reports (Nature Portfolio) -- fits the scope, can
  reformat for another journal later since PAPER_DRAFT.md keeps the full
  version. Checked against nature.com's submission guidelines (the
  official page needs a login, so also checked a mirror): abstract
  199/200 words; main text ~1621/4500 words; figures+tables exactly 8/8
  (6 figures + 2 main tables; the other 5 results tables moved to
  Supplementary Information, nothing removed); references renumbered in
  order of first appearance (fixed an ordering mistake from the first
  version). Declarations section added (data/code availability filled in;
  ethics/competing interests/funding/author contributions/acknowledgements
  are placeholders for a single author -- need to confirm each one).
  Still to do: confirm the venue, the wider-spacing decision, and a full
  read-through.
- Figures: 6 of 6 done. UPDATE 2026-09-16: dropped the earlier
  no-plotting rule -- figures 4-6 (the data ones) are now made by
  generate_figures.py (matplotlib, already installed in venv), which reads
  multiseed_seed*_results.json, film_layer_analysis.json and
  vascular_graph_results.json and saves PNGs to figures/ (not gitignored;
  still need to decide whether to commit the PNGs or regenerate them).
  Figures 1-3 (method diagrams, not data) are drawn separately with a
  diagram tool (not in this repo). Figure 5 needed data I didn't have yet:
  added an --output-json flag to analyze_film_layers.py, re-ran it on the
  100-epoch checkpoint, full 24-layer output in film_layer_analysis.json
  (new file). Not done: qualitative prediction examples, FOV-confound bar
  chart, 3D skeleton overlays (see the figures section of PAPER_DRAFT.md).
- Summary tables for the paper: DONE 2026-09-15 --
  generate_summary_tables.py (text only, no plots). Reads whichever
  *_results.json files exist (sweep_results.json, full_sweep_results.json,
  vascular_graph_results.json, multiseed_seed1_results.json /
  multiseed_results.json once they exist) and uses constants copied from
  test.sh (with the section number) for results that were never saved as
  JSON (FOV confound, FiLM breakdown, clDice ablation). Section 7
  (multiseed) prints "not yet available" until the multiseed kernel below
  finishes. Re-run to refresh: python generate_summary_tables.py
- GitHub push: postponed for now, not blocking anything. When I do it:
  git identity (user.name/user.email) is not set in this repo yet, and
  there is no GitHub login on this machine (no gh CLI, no SSH key, no
  credential helper) -- both need to be set up first (see the git notes
  in README.md).

================================================================
6. OPEN QUESTIONS
================================================================
- The cause of the OOD collapse is not FULLY confirmed, only narrowed
  down: field-of-view mismatch (section 5.5, significant) and FiLM's
  spacing-specific modulation going back to near-identity OOD (section
  5.6, a lead, not proven) are the two best explanations right now. They
  were tested separately and not yet combined into one explanation.
- The ASSD=inf in the original 10-epoch CPU run (README.md section 5.2)
  is a DIFFERENT problem from the HD95/ASSD=nan in the 100-epoch GPU run
  (section 5.4, cause found and fixed in section 5.8). RESOLVED
  2026-09-15 with diagnose_assd_inf.py (see test.sh section 3.2
  addendum): the "completely empty prediction" idea, earlier tested on 15
  patches with 0 confirmed, IS the cause after all. A new, bigger draw
  (92 non-empty patches for Spacing-Conditioned LoRA) gave one ASSD=inf
  case, and it was a fully empty prediction (pred_sum=0) on a patch with
  a tiny annotated target (22 voxels, 0.01% of the patch). It is a rare
  event that depends on which patches are drawn, caused by the model
  completely missing tiny/sparse vessel pieces at 10 epochs, not a fixed
  property of either checkpoint. The same script on PhysicsAwareAdapter's
  checkpoint found 0/125 such cases, which fits "rare" rather than
  "never". The earlier "0 of 15" result most likely just did not draw
  one of these rare patches.

================================================================
7. PLAN FOR MEDICAL IMAGE ANALYSIS (MedIA) -- written 2026-09-23
================================================================
Paper draft: PAPER_MEDIA.md (all [TODO]s there depend on the runs below).
Main things I learned from reading the related work and re-checking the code:

  a) The old "in-distribution" numbers were measured on random patches from
     the TRAINING kidneys. Not a held-out test. MedIA reviewers will reject
     that. -> new held-out benchmark (benchmark/), see step 1.
  b) Jain et al. (Nat Commun 2026) released the test labels + an extra 13um
     VOI + the top-5 teams' predictions on Zenodo 19685381. So we now have
     labelled test data at 63.08 / 50.28 / 13 / 5.2 um. Top-5 teams can be
     scored without any GPU.
  c) The FiLM input never varied in training (50.00/50.16 um only). HyperSpace
     trains with every batch resampled to a random spacing. -> spacing
     augmentation (--fov-scale 0.25 2.0).
  d) SAM-Med3D gives a 16^3 mask for a 64^3 input (checked on CPU), 32^3 for
     128^3. Thin vessels can't be represented. -> 128^3 patches +
     models/refine_head.py (--refine).
  e) Old training skipped empty patches: ~5 real steps per epoch. -> 250
     steps/epoch with 50% foreground-centred patches (utils/slab_dataset.py).
  f) The published nnU-Net baseline on the same kidneys gets Dice 0.86-0.95
     (Yagis et al., Sci Rep 2024); the Kaggle winner gets Dice ~0.82 on our
     kidney_6 ROIs. Our models are far below. Must be reported openly.
  g) HyperSpace also collapses at its highest resolutions and fixes it by
     resampling to the training spacing -> our FOV finding agrees with it.
     The old drafts said the opposite; fixed.
  h) Zenodo VOI-13um: 3729 of 5955 image files are empty; only slices
     0-2225 are usable.

Steps (in order):
  1. [ ] Build benchmark ROIs on Kaggle (CPU, internet on):
         kaggle_kernels/kaggle_rois_kernel.py -> save output as dataset
         rabeya100x/hipct-benchmark-rois. (kidney_6 ROIs + team scores were
         already made locally; Zenodo is too slow here for the VOIs.)
  2. [ ] Upload the code again as rabeya100x/hip-vesseladapt-code (now needs
         benchmark/, train_lora_v2.py, utils/slab_dataset.py,
         models/refine_head.py).
  3. [ ] Score the Kaggle top-5 on all ROIs (CPU, local or Kaggle):
         python -m utils.fetch_zenodo_subset team_predictions
         python -m benchmark.import_team_rle && python -m benchmark.score ...
  4. [ ] Benchmark the OLD 100-epoch checkpoints (3 seeds x 2 models) on the
         new ROIs: kaggle_v2_kernel.py with RUN["checkpoint"] set.
  5. [ ] New training runs, kaggle_v2_kernel.py, 3 seeds each:
         - plain_base      --model plain   --patch 128 (no fov-scale, no refine)
         - spacing_base    --model spacing --patch 128
         - spacing_aug     --model spacing --patch 128 --fov-scale 0.25 2.0
         - plain_aug       --model plain   --patch 128 --fov-scale 0.25 2.0
         - spacing_aug_ref --model spacing --patch 128 --fov-scale 0.25 2.0 --refine
         - plain_aug_ref   --model plain   --patch 128 --fov-scale 0.25 2.0 --refine
         = 18 runs. Each ~6-8 h on a T4 (estimate; check after epoch 1).
         Kaggle free quota is far too small for this -> need Colab Pro /
         university GPU / a cloud credit grant. Minimum set if GPU is short:
         spacing_base, spacing_aug, plain_aug_ref, spacing_aug_ref with 3 seeds.
  6. [ ] Baselines: kaggle_nnunet_kernel.py, kaggle_vesselfm_kernel.py,
         SAM-Med3D zero-shot (benchmark.infer with no checkpoint).
  7. [ ] Spacing/FOV 2x2 (native / tell50 / fov50) comes out of step 4/5
         automatically. Also re-run analyze_film_layers.py on the new models.
  8. [ ] Optional but strong for MedIA: HyperSpace-style baseline (their
         models.py with our loader), a second organ, or fine-tuned vesselFM.
  9. [ ] Fill every [TODO] in PAPER_MEDIA.md, make the figures (graphical
         abstract is required and must not be AI-generated), convert to the
         Elsevier LaTeX template (elsarticle), write the cover letter.
 10. [ ] Fill in the Elsevier declarations truthfully, incl. the generative AI
         declaration.
