"""
Replication runner for the OT and "all" geneset / CLR transform.

Runs all 5 supervised models (l2 LR, RF, kNN, elastic-net LR, XGBoost) under
the paper's nested 5-fold CV protocol on the 5 chosen tasks, with a fixed
random_state so paired-fold comparisons across models are valid.

Run via:
    python -m src.replication            # default: OT geneset
    python -m src.replication --geneset all   # full 57992-gene set (the headline run)

Output: results/replication_<geneset>.csv with one row per (task, model) pair.

Hyperparameter grids match PROJECT_PLAN §5.2 in full — no trimming for runtime.
GPU is auto-detected for XGBoost via src.models.cuda_available().
"""

import argparse
import warnings
from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd

from representation_learning_for_transcriptomics.supervised import (
    LogisticRegressor_skl, RandomForestClassifier_skl,
    CVmodel, scorers, Predictor,
)
from src.models import (
    KNNClassifier_skl, ElasticNetLR_skl, XGBClassifier_skl,
    cuda_available,
)

warnings.filterwarnings('ignore')

# (task_label, filename_pattern_keys, metric_name)
# Per project plan §4 + Appendix A: GSE50244 is multiclass (accuracy), rest binary (AUC)
# Filenames follow <geneset>_<norm>_<group>_<task>.h5 — `group` is "train" or "validate" per Tables 2-4.
TASKS = [
    ('COAD_stage', 'train',    'COAD_stage', 'AUC'),
    ('KIRC_stage', 'train',    'KIRC_stage', 'AUC'),
    ('LGG_grade',  'train',    'LGG_grade',  'AUC'),
    ('GSE65832',   'train',    'GSE65832',   'AUC'),
    ('GSE50244',   'validate', 'GSE50244',   'accuracy'),
]

SEED = 42
N_OUTER = 5
N_INNER = 5

# --- hyperparameter grids: full per PROJECT_PLAN §5.2 ---

# l2 LR — paper / example script
LR_C_INV = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

# RF — example script grid (paper text differs, see PROJECT_PLAN §7.4)
RF_DEPTHS = [4, 8, 16, 32, 64, 128]

# kNN — project plan §5.2
KNN_K = [1, 3, 5, 7, 9]

# Elastic-net — project plan §5.2: C log-spaced 1e-4 to 1e2 in 8 steps; l1_ratio in {0.1, 0.5, 0.9}
ENET_C = list(np.logspace(-4, 2, 8))
ENET_L1 = [0.1, 0.5, 0.9]
ENET_GRID = list(product(ENET_C, ENET_L1))

# XGBoost — project plan §5.2: depth ∈ {3, 6, 9}, n_est ∈ {100, 300}, lr ∈ {0.05, 0.1}
XGB_DEPTH = [3, 6, 9]
XGB_NEST = [100, 300]
XGB_LR = [0.05, 0.1]
XGB_GRID = list(product(XGB_DEPTH, XGB_NEST, XGB_LR))


def task_path(geneset, group, task):
    """e.g. data/extracted/tasks_OT_clr/OT_clr_train_LGG_grade.h5"""
    return Path(f'data/extracted/tasks_{geneset}_clr/{geneset}_clr_{group}_{task}.h5')


def load_task(path):
    """Load expression + labels from an h5 file, standardize per upstream binary_test.py."""
    with pd.HDFStore(str(path), mode='r') as store:
        X = store['expression'].values
        Y = store['labels'].values.ravel()
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # guard against constant features
    X = (X - mu) / std
    return X, Y


def make_cvmodel(model_name, n_classes):
    """Return a CVmodel configured for the given model name."""
    if model_name == 'l2_LR':
        return CVmodel(LogisticRegressor_skl, LR_C_INV, 'C^-1',
                       solver='lbfgs', max_iter=5000)
    if model_name == 'RF':
        return CVmodel(RandomForestClassifier_skl, RF_DEPTHS, 'max_depth',
                       n_estimators=100, n_jobs=-1, random_state=SEED)
    if model_name == 'kNN':
        return CVmodel(KNNClassifier_skl, KNN_K, 'n_neighbors', n_jobs=-1)
    if model_name == 'elastic_net':
        return CVmodel(ElasticNetLR_skl, ENET_GRID, '(C, l1_ratio)',
                       random_state=SEED)
    if model_name == 'XGBoost':
        return CVmodel(XGBClassifier_skl, XGB_GRID, '(depth, n_est, lr)',
                       random_state=SEED)
    raise ValueError(model_name)


MODELS = ['l2_LR', 'RF', 'kNN', 'elastic_net', 'XGBoost']


def run_one(task_label, path, metric, model_name):
    X, Y = load_task(path)
    n_classes = len(np.unique(Y))

    if metric == 'AUC':
        scorer = scorers.roc_auc_scorer
    elif metric == 'accuracy':
        scorer = scorers.accuracy_scorer
    else:
        raise ValueError(metric)

    cvm = make_cvmodel(model_name, n_classes)
    pred = Predictor(cvm, scorer)
    t0 = time.time()
    errs = pred.cross_validate(X, Y,
                               outer_folds=N_OUTER, inner_folds=N_INNER,
                               stratified=True, random_state=SEED)
    elapsed = time.time() - t0
    return {
        'task': task_label, 'model': model_name, 'metric': metric,
        'n_samples': X.shape[0], 'n_features': X.shape[1], 'n_classes': n_classes,
        'mean': float(errs.mean()), 'std': float(errs.std()),
        'fold_scores': ';'.join(f'{e:.4f}' for e in errs),
        'seconds': round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geneset', choices=['OT', 'all'], default='OT',
                        help='which figshare geneset bundle to run on')
    args = parser.parse_args()

    print(f'geneset={args.geneset}  CUDA available for XGBoost: {cuda_available()}')
    out_path = Path(f'results/replication_{args.geneset}.csv')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume support: if the CSV already exists, skip (task, model) pairs
    # that landed on a previous run. each successful task appends to the CSV.
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done = set(zip(existing['task'], existing['model']))
        rows = existing.to_dict('records')
        print(f'resuming: {len(done)} (task, model) cells already on disk')
    else:
        done = set()
        rows = []

    for task_label, group, task_name, metric in TASKS:
        path = task_path(args.geneset, group, task_name)
        if not path.exists():
            print(f'  [skip] {path} not found — run src.download first')
            continue
        for model_name in MODELS:
            if (task_label, model_name) in done:
                print(f'[{task_label:12s}] {model_name:11s} ... [skip, already done]')
                continue
            print(f'[{task_label:12s}] {model_name:11s} ... ', end='', flush=True)
            r = run_one(task_label, path, metric, model_name)
            r['geneset'] = args.geneset
            rows.append(r)
            # checkpoint after every cell — minimizes loss on crash
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f'mean={r["mean"]:.4f} std={r["std"]:.4f}  ({r["seconds"]:.1f}s)')
    df = pd.DataFrame(rows)
    print(f'\nwrote {out_path}')
    return df


if __name__ == '__main__':
    main()
