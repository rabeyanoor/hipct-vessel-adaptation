HiP-VesselAdapt -- Remaining Work (backlog, not a script)
================================================================
Everything below is genuinely not done yet, as of this writing. Nothing
here is implied complete -- see README.md / test.sh for what IS done, with
real numbers. This file exists so remaining work isn't lost when picked
back up later, in one place, honestly stated.

================================================================
1. DONE THIS SESSION (2026-09-15)
================================================================
- Reduced hyperparameter sweep (kaggle_kernels/kaggle_sweep_kernel.py,
  kernel rabeya100x/hip-vesseladapt-hp-sweep on Kaggle GPU) finished and
  results were pulled: sweep_results.json now has all 12 configs (3 LR x
  2 loss_alpha x {plain_lora, spacing_lora}, 5 epochs each). Winner:
  spacing_lora, lr=1e-4, loss_alpha=0.25, dice=0.1364 -- clearly best of
  the 12, and every spacing_lora config beat every plain_lora config.
  lr=1e-4 matches TRAIN_CONFIG's existing default, but loss_alpha=0.25 is
  DIFFERENT from train_spacing_lora.py's existing default (loss_alpha=0.5)
  -- so per this section's own criterion below, the full 3x3 sweep is now
  worth running, not skippable.
- NOTE for next time using the Kaggle API/CLI: the `kaggle` pip package
  (v2.2.4)'s NEWER methods -- `KaggleApi.kernels_status()`, `.kernels_list()`,
  and the CLI's `kaggle datasets version` / `kaggle kernels push` -- go
  through an internal kagglesdk client that authenticates via a BEARER
  token from `KAGGLE_API_TOKEN` env var or `~/.kaggle/access_token`, NOT
  via `~/.kaggle/kaggle.json`'s (username, key) basic auth. Using only
  kaggle.json (which is what `KaggleApi.authenticate()` reads by default)
  makes every one of those calls 401, REGARDLESS of whether the key in
  kaggle.json is valid -- this cost a lot of time this session chasing it
  as a token-scope problem before finding the actual cause. Fix: put a
  valid KGAT_-format token in `~/.kaggle/access_token` (plain text, no
  quotes) -- both the token you paste into kaggle.json's "key" field and
  the one in access_token can be the same fresh token. OLDER methods
  (`dataset_list()`, `model_list()`) and the plain legacy REST endpoints
  (`https://www.kaggle.com/api/v1/kernels/status`, `.../kernels/output`)
  accept kaggle.json's basic auth fine either way -- only the kagglesdk
  path needs access_token. The `.../kernels/output` endpoint returns
  {"log": <JSON-encoded STRING, not a list -- json.loads() it first>,
  "files": [{"fileName", "url"}, ...]}; each file's "url" is a short-lived
  signed URL, GET it directly (no auth needed) for just the one file you
  want instead of printing the whole `files` list (228 entries last time).

================================================================
2. NOT STARTED -- SCRIPT ALREADY WRITTEN, JUST NEEDS TO BE RUN
================================================================
- Full multi-seed statistical reporting: DONE 2026-09-16, n=3 reached (the
  originally-planned size). colab_run.py's run_multiseed() (3 seeds x 2
  architectures x 100 epochs) needs more GPU-hours than one account's
  weekly quota allows (this project's own rabeya100x account was already
  exhausted -- section 1), so this ran instead via two Kaggle-kernel
  scripts (kaggle_kernels/kaggle_multiseed_kernel.py seed=1, and
  kaggle_multiseed_kernel_seed2.py seed=2) on a second Kaggle account
  (nargisakterr) -- both fit in that account's single weekly quota
  (seed=1 used ~9788s of 21600s, leaving room to run seed=2 the same day
  rather than waiting for a reset). Combined with the existing 100-epoch
  checkpoint (unspecified seed) as n=3 per architecture -- see test.sh
  section 3.11 for the real numbers and full interpretation. Headline:
  the OOD collapse is now 3-for-3 confirmed across independent seeds (not
  a single-run fluke); the in-distribution ranking, unresolved at n=2,
  now resolves toward Spacing-Conditioned LoRA winning (2 of 3 seeds, by
  a wide margin both times) -- still only n=3, not a formal significance
  claim, but no longer "unresolved." Raw per-seed numbers in
  multiseed_seed1_results.json / multiseed_seed2_results.json. No further
  seeds currently planned -- this reaches the originally-scoped target.
- Full (not reduced) hyperparameter sweep -- DONE 2026-09-15 (kernel
  rabeya100x/hip-vesseladapt-full-hp-sweep, COMPLETE). Results in
  full_sweep_results.json (18 runs: 3 LRs x 3 loss_alphas x 2 architectures,
  15 epochs each). **This REVERSES the reduced sweep's conclusion**: at 15
  epochs, Dice values across all 18 configs are close together (0.126-0.172),
  and Plain LoRA now wins overall (lr=2e-4, loss_alpha=0.5, Dice=0.1724) --
  the reduced sweep's "loss_alpha=0.25, spacing_lora wins" call does not hold
  up with more epochs. Best loss_alpha range at 15 epochs is 0.5-0.75, not
  0.25. Treat the reduced sweep's config choice as superseded; if retraining
  off sweep results, use this full sweep's numbers, not the reduced one's.
  IMPORTANT: fetching this result burned the rest of the week's Kaggle GPU
  quota -- `api.quota_view()` (works now with KAGGLE_API_TOKEN set) showed
  42814s used against a 21600s weekly allowance, i.e. already ~5.9h OVER
  the free quota, refreshing 2026-09-19. No more Kaggle GPU kernels
  until then -- use Colab (human-driven) if GPU work is needed before that.
- Vascular graph validation (evaluate_vascular_graph.py, utils/vascular_
  graph_metrics.py): RUN FOR REAL 2026-09-15, locally on CPU (no GPU
  needed -- it's mask skeletonization + graph stats, not training).
  Against both real 100-epoch GPU checkpoints, in-distribution and OOD,
  same protocol as test.sh sections 3.2-3.4. Full write-up in test.sh
  section 3.10, raw per-patch numbers in vascular_graph_results.json.
  Headline: a THIRD independent piece of evidence for the OOD collapse
  (alongside 3.5's FOV confound and 3.6's FiLM lead), and it shows the two
  architectures fail OOD in qualitatively different ways, not just by
  different amounts -- Plain LoRA's OOD predictions moderately fragment
  but keep branch-point counts close to real GT; Spacing-Conditioned
  LoRA's OOD predictions fragment 11x worse and hyper-branch 31x worse
  than GT, i.e. a scattered field of spurious disconnected fragments
  replacing the real vessel tree, not a smaller/weaker version of it. This
  item is REMOVED from section 3 below (it used to say "no script exists"
  -- it existed already, just hadn't been run).

================================================================
3. IN PROGRESS -- DATA READY, TRAINING WIRING + DESIGN CALL STILL NEEDED
================================================================
- Train on a wider range of real spacings. CORRECTION 2026-09-16: this was
  wrongly marked "no script yet" above -- a previous session had already
  started on this (config.py's HOA_KIDNEY_MULTISPACING_CONFIG, utils/
  convert_hoa_parquet_to_memmap.py) and it went unnoticed until a full
  codebase read-through this session. Real status now:
    - REAL DATA, ALREADY DOWNLOADED: 3 more native spacings (25um, 6.5um,
      2.6um) of the SAME physical specimen as kidney_1_dense/kidney_1_voi
      (LADAF-2021-17-Right), from the Human Organ Atlas HF mirror
      (eternalaudrey/hipct_3d). Parquet shards were already pulled to
      data/raw/hoa_kidney_right_shards/ (~1.4GB, gitignored).
    - CONVERTED 2026-09-16 (this session): ran
      utils/convert_hoa_parquet_to_memmap.py -- all 3 volumes now real
      memmap arrays in data/memmap/ (2176x224x224 uint16 each, 0.22GB).
      Needed `python3 -m utils.convert_hoa_parquet_to_memmap` (plain
      `python3 utils/convert_hoa_parquet_to_memmap.py` fails on the
      `from config import ...` line -- run as a module, or from a context
      with the project root on PYTHONPATH).
    - DATASET WIRING: NONE NEEDED -- utils/dataset.py's
      MultiOrganHiPCTDataset takes any config dict generically
      (`MultiOrganHiPCTDataset(config=HOA_KIDNEY_MULTISPACING_CONFIG, ...)`
      works as-is, no code changes). No training script actually calls it
      with this config yet, though -- that part is still not written.
    - IMPORTANT CAVEAT, not yet resolved: this data has NO real expert GT
      (config.py's own comment: Frangi pseudo-labeled only, must NOT be
      combined quantitatively with RealGTKidneyDataset's real-GT numbers,
      per README.md section 6's existing rule). It could be used to train
      across a wider spacing range (pseudo-labeled) and then still
      evaluated on the REAL GT OOD test (kidney_1_voi, 5.2um) -- but two
      of the three new spacings (6.5um, 2.6um) are very close to that
      5.2um held-out test point. Training on those specific spacings
      before evaluating there risks weakening the OOD test's validity
      (no longer testing a fully unseen spacing regime, even though the
      GT itself stays held out) -- this needs a real decision (e.g. drop
      the two closest spacings and train on 25um alone as the "third
      point," or keep all three and reframe the claim as "interpolation"
      rather than "extrapolation" robustness) before training anything on
      it. NOT done unilaterally this session -- flagged for the user.

================================================================
4. EXPLICITLY OUT OF SCOPE (locked earlier this session, not forgotten)
================================================================
- Additional baselines: nnU-Net, VesselFM, VesselSDF, HyperSpace-style
  baseline. Not implemented, not planned until the core OOD question is
  further resolved.
- Cross-organ generalization test (pseudo-label organs vs. real-GT
  kidney) -- explicitly must not be mixed quantitatively per README.md
  section 6, and not attempted.
- Few-shot experiments.
- CollapseDetectionHead training/validation (exists in
  models/collapse_head.py, never trained or evaluated -- explicitly
  deferred earlier in the project).

================================================================
5. WRITING / PUBLISHING (not code, but not done)
================================================================
- The paper itself -- DRAFT WRITTEN 2026-09-16, PAPER_DRAFT.md, Introduction/
  Related Work/Conclusion COMPLETED same day (were placeholder markers,
  now real prose). Full Abstract/Intro/Related Work/Methods/Results/
  Discussion/Conclusion/Limitations, grounded in test.sh's real numbers.
  10 citations now, all verified directly (title/authors/venue checked
  live, not recalled) -- includes a real literature search this session
  for the mechanism/baseline-relevant related work (nnU-Net, LoRA-on-SAM
  variants, HyperSpace, FiLM-metadata-conditioning). Worth noting: an
  earlier pass had surfaced "VesselFM/TopoLoRA-SAM/MorVess/HyperSpace" as
  unverifiable and excluded all four -- this session's real search found
  HyperSpace IS a real, verifiable MICCAI 2024 paper after all (now
  cited properly); the other three still were not found and remain
  excluded. Remaining `[NEEDS AUTHOR INPUT]` markers: target venue
  (affects formatting/cover letter), the wider-spacing design decision,
  public-release decision, and a few more related-work citations for a
  fully thorough submission (see PAPER_DRAFT.md section 2's own note).
  Not yet reviewed by a human -- do that before anything goes out.
- Journal-formatted submission draft: DONE 2026-09-16, PAPER_SUBMISSION_SCIREP.md.
  Target venue picked (Scientific Reports, Nature Portfolio -- chosen as
  a concrete, realistic fit for this paper's actual scope, not a
  guaranteed acceptance; easy to reformat for a different venue later
  since PAPER_DRAFT.md stays the untrimmed source of truth). Verified
  live against nature.com's real submission guidelines (cross-checked
  against an independent mirror since the official page needs a login):
  abstract checked at 191/200 words; main text checked at ~1621/4500
  words; combined figures+tables checked at exactly 8/8 (the 6 figures
  plus 2 consolidated main tables -- the other 5 results tables moved to
  a Supplementary Information outline, nothing dropped, just relocated);
  references renumbered and verified in strict first-appearance order
  (caught and fixed a real ordering bug from the first pass). Added a
  full Declarations block (Data/Code availability with real facts filled
  in; Ethics/Competing interests/Funding/Author contributions/
  Acknowledgements left as clearly-flagged solo-author-consistent
  defaults, not fabricated -- confirm or correct each one). Still
  blocked on: actually deciding this is the venue (easy to change),
  the wider-spacing design call, and a human read-through.
- Figures: 6 of 6 built. UPDATE 2026-09-16: the no-plotting rule was
  explicitly reversed by the user (see feedback_no_plotting_in_repo.md
  memory, superseded) -- figures 4-6 (the real-data ones) are now
  generated by running generate_figures.py (matplotlib, new dependency
  already in venv), which reads multiseed_seed*_results.json,
  film_layer_analysis.json, and vascular_graph_results.json and saves
  PNGs to figures/ (not yet gitignored either way -- your call whether to
  commit the PNGs or regenerate on demand). Figures 1-3 (the mechanism
  diagrams -- conceptual, not data, so left as a diagram tool rather than
  matplotlib) are still a separate online artifact:
  (online figure)
  Figure 5 needed data that didn't exist yet: analyze_film_layers.py
  gained an --output-json flag, re-run against the 100-epoch checkpoint,
  full 24-layer breakdown now in film_layer_analysis.json (new file).
  Still not done: qualitative prediction renders, FOV-confound bar chart,
  3D skeleton overlays (see PAPER_DRAFT.md's figures section).
- Final summary tables for the paper: DONE 2026-09-15 --
  generate_summary_tables.py (text-only, no plotting, per project rule).
  Pulls live numbers from whatever *_results.json files exist on disk
  (sweep_results.json, full_sweep_results.json, vascular_graph_results.json,
  multiseed_seed1_results.json / multiseed_results.json once they exist) and
  falls back to constants transcribed from test.sh (cited by section number)
  for numbers that were never saved as JSON by the scripts that produced
  them (FOV confound, FiLM breakdown, clDice ablation). Section 7
  (multiseed) currently prints "not yet available" and will auto-populate
  once the multiseed kernel below completes. Re-run it any time to refresh
  the report as more results land: python generate_summary_tables.py
- GitHub push -- explicitly deferred by the user's own choice this
  session ("GitHub push korte hobe na"), not a blocker. If revisited
  later: git identity (user.name/user.email) is still not configured in
  this repo, and there is still no GitHub authentication method available
  in this environment (no gh CLI, no SSH key, no credential helper) -- both
  need to be set up by the user directly, not by the assistant (see README.md's
  git safety notes / this session's chat history for the exact commands).

================================================================
6. KNOWN OPEN QUESTIONS (not tasks exactly, but not resolved)
================================================================
- Root cause of the OOD collapse is not FULLY confirmed, only
  substantially narrowed: field-of-view mismatch (section 5.5, confirmed
  significant) and FiLM's spacing-specific modulation degenerating toward
  near-identity OOD (section 5.6, a lead, not proven) are the two current
  best explanations, evaluated independently, not combined into one
  unified causal story yet.
- The original 10-epoch CPU run's ASSD=inf finding (README.md section
  5.2) is a DIFFERENT anomaly from the 100-epoch GPU run's HD95/ASSD=nan
  (section 5.4, root-caused and fixed in section 5.8). RESOLVED 2026-09-15
  by diagnose_assd_inf.py (see test.sh section 3.2 addendum): the
  "completely empty prediction" hypothesis, previously tested on 15
  patches with 0 confirmed and left inconclusive, IS the real cause after
  all -- a fresh, larger draw (92 non-empty patches for Spacing-Conditioned
  LoRA) reproduced one genuine ASSD=inf case, and it was a fully-empty
  prediction (pred_sum=0) on a patch with a vanishingly small annotated
  target (22 voxels, 0.01% of the patch). It's a rare, patch-draw-dependent
  event tied to the model completely missing tiny/sparse vessel fragments
  at 10 epochs, not a deterministic property of either checkpoint -- the
  same script's run against PhysicsAwareAdapter's checkpoint found 0/125
  such cases, consistent with rarity rather than absence. The earlier "0
  confirmed of 15" result most likely just didn't happen to draw one of
  these rare small-target patches.
