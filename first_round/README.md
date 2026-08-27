# First round of experiments

The first experiments (September 2026): plain vs. spacing-conditioned LoRA on
64³ patches, FiLM on the raw spacing in µm, 10 epochs on CPU and 100 epochs on a
Kaggle T4, evaluated on random patches of the training volumes and of
`kidney_1_voi`. They are reported in Section 5.6 of the paper only to show how
the question arose; the held-out benchmark in `benchmark/` replaces their
evaluation. Their checkpoints are re-scored on the benchmark ROIs as the
"first round" rows of Table 1.

* `train_plain_lora.py`, `train_spacing_lora.py`, `train_real_gt.py`, `train.py` — training
* `evaluate_*.py`, `significance_test_fov.py` — patch-based evaluation and checks
* `analyze_film_layers.py` — per-layer FiLM analysis (`results/film_layer_analysis.json`)
* `generate_summary_tables.py`, `generate_figures.py` — tables and figures from `results/`
* `kaggle/` — the Kaggle kernels used for these runs
* `results/` — sweeps, multi-seed and vascular-graph results; `figures/` — figures 4-6 of the first draft

The scripts import the shared code (`config.py`, `models/`, `utils/`, `losses/`)
from the repository root. Run them from the root with the root on the path, e.g.

```bash
PYTHONPATH=. python first_round/generate_summary_tables.py
```

The Kaggle kernels in `kaggle/` expect the old flat layout, in which these
scripts sat in the repository root; check out the tag `v0-flat-layout` to rerun
them exactly as they were run. The dated log of these experiments is
`docs/results_log.txt` (sections 3.x) and `docs/research_log.md`.
