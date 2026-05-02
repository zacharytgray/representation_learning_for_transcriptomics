# Linux + GPU handoff

If you're a fresh agent picking this up on the Linux box: **read `PROJECT_PLAN.md` §0 in the Google Drive `Grad Report/` folder first** for the high-level state. This file is just the machine-specific setup and the next concrete commands to run.

The Google Drive `Grad Report/` folder is the writing/context dir; this repo is the code dir. They were intentionally split so git operations don't fight Google Drive sync. Both locations are referenced by `Grad Report/CLAUDE.md`.

## Why this handoff exists

The OT geneset sanity check ran end-to-end on a Mac (CPU only) and confirmed the pipeline is correct — 11 of 15 published-target cells within tolerance. Rather than reconcile partial CPU results with new GPU results, we wiped the local results and are re-running everything once on the Linux box (RTX 5060 Ti, 16 GB VRAM) for one consistent result set.

The hyperparameter grids in `src/replication.py` were briefly trimmed during the CPU pass to keep wall-clock manageable; they have been restored to **full paper-spec** (24 elastic-net combos, 12 XGBoost combos). No CPU compromises remain in the code.

## What's already in this repo

- `representation_learning_for_transcriptomics/` — the upstream package, patched per `PROJECT_PLAN.md` §7.4 (np.int → np.int64, dropped cytoolz, threaded random_state through CV).
- `src/models.py` — kNN, elastic-net, XGBoost wrappers. `XGBClassifier_skl` auto-detects CUDA via `cuda_available()` and sets `device='cuda'` if a GPU is present.
- `src/download.py` — pulls the figshare tarballs and extracts only the 5 task files.
- `src/replication.py` — main runner. `--geneset OT` (sanity) or `--geneset all` (headline).
- `src/compare.py` — diffs our results against `PROJECT_PLAN.md` Appendix A targets.
- `requirements.txt` — pinned. xgboost 2.1.3 wheel ships with CUDA support on Linux.

## Setup

Assumes a recent NVIDIA driver supporting CUDA 12.x is already installed (`nvidia-smi` should work). The 5060 Ti is Blackwell (sm_120) — needs a recent driver.

```bash
# clone (if not already)
git clone https://github.com/zacharytgray/representation_learning_for_transcriptomics.git
cd representation_learning_for_transcriptomics

# python env. uv is fastest; venv works equally well.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .

# verify GPU is wired
.venv/bin/python -c "from src.models import cuda_available; print('CUDA:', cuda_available())"
# expected: CUDA: True
```

If `cuda_available()` returns False on a machine that does have a GPU:
- Check `nvidia-smi` works.
- Check `python -c "import xgboost; print(xgboost.build_info())"` shows `USE_CUDA: True`. The standard pip wheel for xgboost 2.1.3 on Linux does, but if you're on a weird wheel/CPU-only fallback, reinstall with `pip install --force-reinstall xgboost==2.1.3`.
- If sm_120 isn't supported by the wheel's bundled CUDA runtime, you may need to build xgboost from source against CUDA 12.8+. Check the xgboost release notes for the Blackwell support cutoff.

## Run

```bash
# 1. download data (OT 101 MB + all 2.96 GB), extract our 5 task files
.venv/bin/python -m src.download --all

# 2. OT sanity replication — should take ~10–15 min wall clock on this hardware
.venv/bin/python -m src.replication --geneset OT
.venv/bin/python -m src.compare --geneset OT
# expected: ≥11/15 cells within tolerance, similar to the CPU pass

# 3. headline run on the full 57992-gene set — the actual report-grade result
.venv/bin/python -m src.replication --geneset all
.venv/bin/python -m src.compare --geneset all
# wall clock estimate: 30–90 min on the 5060 Ti
```

Outputs land in `results/replication_OT.csv`, `results/replication_all.csv`, and the corresponding `replication_vs_published_*.csv` files.

## After the runs land

These are the §3-4 items in `PROJECT_PLAN.md` §0 — to be picked up on this machine once the headline run is done:

1. Per-task paired Wilcoxon signed-rank vs l2 LR baseline (use the per-fold scores in the `fold_scores` column of the CSV).
2. Across-task shifted-statistic plot in Smith et al. Fig. 3 style. Reuse the `score - per_task_median` shape from the paper.
3. Elastic-net sparsity table — refit elastic-net on each task with its CV-selected hyperparameters and count `np.sum(model.coef_ != 0)`.
4. Optional: top-K gene overlap between l2 LR and elastic-net coefficient rankings.
5. Per-task AUC/accuracy bar chart; replication-vs-published scatter.

Drop figures into `figures/`, drop summary tables into `results/`. The 10-page report itself lives in `Grad Report/` (Google Drive), not in this repo.

## Gotchas to know about

- `multi_class='auto'` in `LogisticRegressor_skl` triggers a deprecation FutureWarning in sklearn 1.5+. Filtered globally in `src/replication.py`.
- The `n_jobs` knob on `XGBClassifier_skl` only applies when `device='cpu'` — gpu fits don't oversubscribe on that knob.
- `ElasticNetLR_skl` defaults to `max_iter=5000`, sklearn-default `tol=1e-4`. Don't trim these here — the project plan calls them out as paper-faithful and the CPU machine briefly trimmed them during sanity-checking.
- `SEED = 42` and `N_OUTER = N_INNER = 5` in `src/replication.py`. Matches `PROJECT_PLAN.md` §5.3.
- Force CPU at runtime with `RLT_FORCE_CPU=1 python -m src.replication ...` (useful if you want a CPU baseline for comparison).

## Where to update this if the plan changes

`PROJECT_PLAN.md` §0 (in the Google Drive `Grad Report/` folder) is the canonical "where we are" doc. Update it when you finish a chunk so the next agent doesn't re-derive state from the code. This file (`LINUX_HANDOFF.md`) only changes if the *machine setup* changes.
