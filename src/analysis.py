"""
Post-replication analysis pass.

Produces every figure and table the report needs from the two CSVs that
src.replication.py writes:

  figures/per_task_auc.png         — per-task AUC across 5 models (all geneset)
  figures/replication_scatter.png  — ours vs published (OT + all)
  figures/shifted_statistic.png    — Smith Fig. 3 style across-task summary
  results/elasticnet_sparsity.csv  — nonzero-coef counts per task (CV-selected hyperparams)
  results/topk_overlap.csv         — top-K gene overlap between l2 LR and elastic-net rankings

Run:
  python -m src.analysis
"""

import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold

from src.compare import PUBLISHED_OT, PUBLISHED_ALL
from src.replication import TASKS, ENET_GRID, LR_C_INV, SEED, task_path, load_task

warnings.filterwarnings('ignore')

MODELS = ['l2_LR', 'RF', 'kNN', 'elastic_net', 'XGBoost']
MODEL_LABELS = {
    'l2_LR': 'l2 LR', 'RF': 'RF', 'kNN': 'kNN',
    'elastic_net': 'Elastic-net', 'XGBoost': 'XGBoost',
}
COLORS = {
    'l2_LR': '#1f77b4', 'RF': '#ff7f0e', 'kNN': '#2ca02c',
    'elastic_net': '#d62728', 'XGBoost': '#9467bd',
}

FIG_DIR = Path('figures')
RES_DIR = Path('results')


def parse_fold_scores(s):
    return np.array([float(x) for x in s.split(';')])


def load_replication(geneset):
    df = pd.read_csv(RES_DIR / f'replication_{geneset}.csv')
    df['fold_scores'] = df['fold_scores'].map(parse_fold_scores)
    return df


# --- figure 1: per-task AUC bar chart on the all geneset ---
def per_task_bar(df_all):
    tasks = [t for t, *_ in TASKS]
    models = MODELS
    width = 0.16
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(models):
        means = [float(df_all[(df_all['task'] == t) & (df_all['model'] == m)]['mean'].iloc[0])
                 for t in tasks]
        stds = [float(df_all[(df_all['task'] == t) & (df_all['model'] == m)]['std'].iloc[0])
                for t in tasks]
        ax.bar(x + (i - 2) * width, means, width, yerr=stds,
               label=MODEL_LABELS[m], color=COLORS[m], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={int(df_all[df_all["task"]==t]["n_samples"].iloc[0])})'
                        for t in tasks], fontsize=9)
    ax.set_ylabel('Score (AUC, or accuracy for GSE50244)')
    ax.set_title('Per-task nested-CV score, all-geneset CLR')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.6)
    ax.legend(ncol=5, frameon=False, fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'per_task_auc.png', dpi=200)
    plt.close(fig)
    print('wrote figures/per_task_auc.png')


# --- figure 2: replication scatter (ours vs published) ---
def replication_scatter(df_ot, df_all):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, df, pub_dict, marker in [('OT', df_ot, PUBLISHED_OT, 'o'),
                                         ('all', df_all, PUBLISHED_ALL, 's')]:
        xs, ys = [], []
        for (task, model), (pub, metric) in pub_dict.items():
            m = df[(df['task'] == task) & (df['model'] == model)]
            if m.empty:
                continue
            xs.append(pub)
            ys.append(float(m['mean'].iloc[0]))
        ax.scatter(xs, ys, marker=marker, s=42, alpha=0.85, label=f'{label} geneset')
    lo, hi = 0.5, 0.85
    ax.plot([lo, hi], [lo, hi], color='grey', linestyle='--', linewidth=0.8)
    # tolerance bands
    ax.fill_between([lo, hi], [lo - 0.03, hi - 0.03], [lo + 0.03, hi + 0.03],
                    color='grey', alpha=0.1, label='±0.03 AUC tol')
    ax.set_xlabel('Published score (Smith et al. 2020)')
    ax.set_ylabel('Our replication')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title('Replication of published nested-CV scores\n(15 cells × 2 genesets, 3 models)')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_aspect('equal')
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'replication_scatter.png', dpi=200)
    plt.close(fig)
    print('wrote figures/replication_scatter.png')


# --- figure 3: shifted-statistic plot (Smith Fig. 3 style) ---
def shifted_statistic(df_all):
    """Per task, subtract the across-model median; plot each model's offset across tasks."""
    tasks = [t for t, *_ in TASKS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    for t in tasks:
        sub = df_all[df_all['task'] == t].set_index('model')
        med = sub['mean'].median()
        for m in MODELS:
            rows.append({'task': t, 'model': m,
                         'shifted': float(sub.loc[m, 'mean']) - med})
    sdf = pd.DataFrame(rows)
    x = np.arange(len(tasks))
    for i, m in enumerate(MODELS):
        ys = [sdf[(sdf.task == t) & (sdf.model == m)].shifted.iloc[0] for t in tasks]
        ax.plot(x, ys, marker='o', label=MODEL_LABELS[m], color=COLORS[m], linewidth=1.4)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel('score − per-task median')
    ax.set_title('Across-task shifted statistic (Smith Fig. 3 style)')
    ax.legend(ncol=5, frameon=False, fontsize=9)
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'shifted_statistic.png', dpi=200)
    plt.close(fig)
    print('wrote figures/shifted_statistic.png')


# --- table: elastic-net sparsity ---
def elasticnet_sparsity(geneset='all'):
    """For each task, pick (C, l1_ratio) by 5-fold CV mean AUC on the full task data,
    refit on all data with those params, count nonzero coefs.
    Comparable l2 LR refit included for context."""
    rows = []
    for task_label, group, task_name, metric in TASKS:
        path = task_path(geneset, group, task_name)
        if not path.exists():
            continue
        X, Y = load_task(path)
        n_classes = len(np.unique(Y))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        # search elastic-net grid by mean fold AUC (or accuracy if multiclass)
        from sklearn.metrics import roc_auc_score, accuracy_score
        is_binary = n_classes == 2
        best = (None, -np.inf)
        for C, l1 in ENET_GRID:
            scores = []
            alpha = 1.0 / (C * len(Y))
            for tr, va in skf.split(X, Y):
                m = SGDClassifier(loss='log_loss', penalty='elasticnet',
                                  alpha=alpha, l1_ratio=l1,
                                  max_iter=5000, random_state=SEED)
                m.fit(X[tr], Y[tr])
                if is_binary:
                    scores.append(roc_auc_score(Y[va], m.decision_function(X[va])))
                else:
                    scores.append(accuracy_score(Y[va], m.predict(X[va])))
            mean = np.mean(scores)
            if mean > best[1]:
                best = ((C, l1), mean)
        C_star, l1_star = best[0]
        # final refit
        en = SGDClassifier(loss='log_loss', penalty='elasticnet',
                           alpha=1.0 / (C_star * len(Y)), l1_ratio=l1_star,
                           max_iter=5000, random_state=SEED)
        en.fit(X, Y)
        en_nonzero = int(np.sum(np.abs(en.coef_) > 1e-10))

        # comparable l2 LR refit (best C by inner CV) — for the "did sparsity cost AUC" comparison
        best_lr = (None, -np.inf)
        for C in LR_C_INV:
            scores = []
            for tr, va in skf.split(X, Y):
                m = LogisticRegression(C=C, solver='lbfgs', max_iter=5000)
                m.fit(X[tr], Y[tr])
                if is_binary:
                    scores.append(roc_auc_score(Y[va], m.decision_function(X[va])))
                else:
                    scores.append(accuracy_score(Y[va], m.predict(X[va])))
            mean = np.mean(scores)
            if mean > best_lr[1]:
                best_lr = (C, mean)
        C_lr = best_lr[0]
        lr = LogisticRegression(C=C_lr, solver='lbfgs', max_iter=5000)
        lr.fit(X, Y)
        lr_nonzero = int(np.sum(np.abs(lr.coef_) > 1e-10))

        rows.append({
            'task': task_label,
            'n_features': X.shape[1],
            'metric': metric,
            'enet_C': C_star, 'enet_l1_ratio': l1_star,
            'enet_cv_score': round(best[1], 4),
            'enet_nonzero': en_nonzero,
            'enet_sparsity': round(1 - en_nonzero / X.shape[1], 4),
            'lr_C': C_lr,
            'lr_cv_score': round(best_lr[1], 4),
            'lr_nonzero': lr_nonzero,
        })
        print(f'  [{task_label}] EN  ({C_star:.3g}, {l1_star}) -> {en_nonzero}/{X.shape[1]} nonzero; '
              f'LR C={C_lr:.3g} -> {lr_nonzero} nonzero')

    out = pd.DataFrame(rows)
    out_path = RES_DIR / f'elasticnet_sparsity_{geneset}.csv'
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path}')
    return out


# --- topK gene overlap (optional) ---
def topk_overlap(geneset='all', k=50):
    """Refit l2 LR (best C) and elastic-net (best (C,l1)) on each task's full data,
    rank features by |coef|, report top-K Jaccard overlap."""
    from sklearn.metrics import roc_auc_score, accuracy_score
    rows = []
    for task_label, group, task_name, metric in TASKS:
        path = task_path(geneset, group, task_name)
        if not path.exists():
            continue
        X, Y = load_task(path)
        n_classes = len(np.unique(Y))
        is_binary = n_classes == 2
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        best_en = (None, -np.inf)
        for C, l1 in ENET_GRID:
            scores = []
            alpha = 1.0 / (C * len(Y))
            for tr, va in skf.split(X, Y):
                m = SGDClassifier(loss='log_loss', penalty='elasticnet',
                                  alpha=alpha, l1_ratio=l1,
                                  max_iter=5000, random_state=SEED)
                m.fit(X[tr], Y[tr])
                scores.append(roc_auc_score(Y[va], m.decision_function(X[va])) if is_binary
                              else accuracy_score(Y[va], m.predict(X[va])))
            if np.mean(scores) > best_en[1]:
                best_en = ((C, l1), np.mean(scores))
        C_en, l1_en = best_en[0]
        en = SGDClassifier(loss='log_loss', penalty='elasticnet',
                           alpha=1.0 / (C_en * len(Y)), l1_ratio=l1_en,
                           max_iter=5000, random_state=SEED).fit(X, Y)

        best_lr = (None, -np.inf)
        for C in LR_C_INV:
            scores = []
            for tr, va in skf.split(X, Y):
                m = LogisticRegression(C=C, solver='lbfgs', max_iter=5000).fit(X[tr], Y[tr])
                scores.append(roc_auc_score(Y[va], m.decision_function(X[va])) if is_binary
                              else accuracy_score(Y[va], m.predict(X[va])))
            if np.mean(scores) > best_lr[1]:
                best_lr = (C, np.mean(scores))
        lr = LogisticRegression(C=best_lr[0], solver='lbfgs', max_iter=5000).fit(X, Y)

        # collapse coefs to per-feature importance
        en_imp = np.abs(en.coef_).sum(axis=0)
        lr_imp = np.abs(lr.coef_).sum(axis=0)
        en_top = set(np.argsort(en_imp)[::-1][:k])
        lr_top = set(np.argsort(lr_imp)[::-1][:k])
        overlap = len(en_top & lr_top)
        jaccard = overlap / len(en_top | lr_top)
        rows.append({'task': task_label, 'k': k,
                     'overlap': overlap, 'jaccard': round(jaccard, 3)})
        print(f'  [{task_label}] top-{k}: {overlap}/{k} overlap (Jaccard {jaccard:.2f})')

    out = pd.DataFrame(rows)
    out_path = RES_DIR / f'topk_overlap_{geneset}.csv'
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path}')
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    df_ot = load_replication('OT')
    df_all = load_replication('all')

    print('--- figures ---')
    per_task_bar(df_all)
    replication_scatter(df_ot, df_all)
    shifted_statistic(df_all)

    print('\n--- elastic-net sparsity (all geneset) ---')
    elasticnet_sparsity('all')

    print('\n--- top-50 gene overlap (all geneset) ---')
    topk_overlap('all', k=50)


if __name__ == '__main__':
    main()
