# Zachary Gray
#
# nested 5-fold CV runner over 5 supervised models on the 5 chosen tasks
# from Smith et al. (2020). fixed seed so paired-fold comparisons are valid.
#
#   python -m src.replication
#   python -m src.replication --geneset all

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

# (task_label, group, filename_task, metric)
# GSE50244 is multiclass (accuracy); rest binary (AUC).
# filenames are <geneset>_<norm>_<group>_<task>.h5; group is "train" or "validate".
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

# l2 LR grid from the original binary_test.py
LR_C_INV = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

# RF max_depth grid from binary_test.py (paper text differs slightly)
RF_DEPTHS = [4, 8, 16, 32, 64, 128]

KNN_K = [1, 3, 5, 7, 9]

# elastic-net: C log-spaced 1e-4..1e2 in 8 steps, l1_ratio in {0.1, 0.5, 0.9} -> 24 combos
ENET_C = list(np.logspace(-4, 2, 8))
ENET_L1 = [0.1, 0.5, 0.9]
ENET_GRID = list(product(ENET_C, ENET_L1))

# xgboost: depth in {3,6,9}, n_est in {100,300}, lr in {0.05,0.1} -> 12 combos
XGB_DEPTH = [3, 6, 9]
XGB_NEST = [100, 300]
XGB_LR = [0.05, 0.1]
XGB_GRID = list(product(XGB_DEPTH, XGB_NEST, XGB_LR))


def task_path(geneset, group, task):
    return Path(f'data/extracted/tasks_{geneset}_clr/{geneset}_clr_{group}_{task}.h5')


# load expression + labels, then standardize per the original binary_test.py
def load_task(path):
    with pd.HDFStore(str(path), mode='r') as store:
        X = store['expression'].values
        Y = store['labels'].values.ravel()
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # constant-feature guard
    X = (X - mu) / std
    return X, Y


def make_cvmodel(model_name, n_classes):
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
    parser.add_argument('--geneset', choices=['OT', 'all'], default='OT')
    args = parser.parse_args()

    print(f'geneset={args.geneset}  CUDA available for XGBoost: {cuda_available()}')
    out_path = Path(f'results/replication_{args.geneset}.csv')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume support: skip (task, model) cells already in the CSV from a prior run
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
            print(f'  [skip] {path} not found - run src.download first')
            continue
        for model_name in MODELS:
            if (task_label, model_name) in done:
                print(f'[{task_label:12s}] {model_name:11s} ... [skip, already done]')
                continue
            print(f'[{task_label:12s}] {model_name:11s} ... ', end='', flush=True)
            r = run_one(task_label, path, metric, model_name)
            r['geneset'] = args.geneset
            rows.append(r)
            # checkpoint after every cell; minimizes loss on crash
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f'mean={r["mean"]:.4f} std={r["std"]:.4f}  ({r["seconds"]:.1f}s)')
    df = pd.DataFrame(rows)
    print(f'\nwrote {out_path}')
    return df


if __name__ == '__main__':
    main()
