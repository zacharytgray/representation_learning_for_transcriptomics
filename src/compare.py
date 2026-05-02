# Zachary Gray
#
# compare our nested-CV scores vs the published replication targets.
#   python -m src.compare
#   python -m src.compare --geneset all
# tolerance: +/- 0.03 AUC, +/- 0.05 accuracy.

import argparse
import pandas as pd

# Smith et al. (2020) supervised-results table, no embedding, gene_set=all, clr
PUBLISHED_ALL = {
    ('COAD_stage', 'l2_LR'):       (0.6865, 'AUC'),
    ('KIRC_stage', 'l2_LR'):       (0.7900, 'AUC'),
    ('LGG_grade',  'l2_LR'):       (0.7693, 'AUC'),
    ('GSE65832',   'l2_LR'):       (0.7000, 'AUC'),
    ('GSE50244',   'l2_LR'):       (0.6333, 'accuracy'),
    ('COAD_stage', 'RF'):          (0.6616, 'AUC'),
    ('KIRC_stage', 'RF'):          (0.7502, 'AUC'),
    ('LGG_grade',  'RF'):          (0.7687, 'AUC'),
    ('GSE65832',   'RF'):          (0.7750, 'AUC'),
    ('GSE50244',   'RF'):          (0.6058, 'accuracy'),
    ('COAD_stage', 'kNN'):         (0.5740, 'AUC'),
    ('KIRC_stage', 'kNN'):         (0.6892, 'AUC'),
    ('LGG_grade',  'kNN'):         (0.6787, 'AUC'),
    ('GSE65832',   'kNN'):         (0.5688, 'AUC'),
    ('GSE50244',   'kNN'):         (0.6317, 'accuracy'),
}

# same table, gene_set=OT, clr (sanity check)
PUBLISHED_OT = {
    ('COAD_stage', 'l2_LR'):       (0.6809, 'AUC'),
    ('KIRC_stage', 'l2_LR'):       (0.7849, 'AUC'),
    ('LGG_grade',  'l2_LR'):       (0.7626, 'AUC'),
    ('GSE65832',   'l2_LR'):       (0.7000, 'AUC'),
    ('GSE50244',   'l2_LR'):       (0.6592, 'accuracy'),
    ('COAD_stage', 'RF'):          (0.6921, 'AUC'),
    ('KIRC_stage', 'RF'):          (0.7570, 'AUC'),
    ('LGG_grade',  'RF'):          (0.7513, 'AUC'),
    ('GSE65832',   'RF'):          (0.7500, 'AUC'),
    ('GSE50244',   'RF'):          (0.5925, 'accuracy'),
    ('COAD_stage', 'kNN'):         (0.5949, 'AUC'),
    ('KIRC_stage', 'kNN'):         (0.6759, 'AUC'),
    ('LGG_grade',  'kNN'):         (0.6789, 'AUC'),
    ('GSE65832',   'kNN'):         (0.6000, 'AUC'),
    ('GSE50244',   'kNN'):         (0.6717, 'accuracy'),
}

PUBLISHED = {'OT': PUBLISHED_OT, 'all': PUBLISHED_ALL}

TOL_AUC = 0.03
TOL_ACC = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geneset', choices=['OT', 'all'], default='OT')
    args = parser.parse_args()

    df = pd.read_csv(f'results/replication_{args.geneset}.csv')
    targets = PUBLISHED[args.geneset]

    rows = []
    for (task, model), (pub, metric) in targets.items():
        m = df[(df['task'] == task) & (df['model'] == model)]
        if m.empty:
            continue
        ours = float(m['mean'].iloc[0])
        delta = ours - pub
        tol = TOL_ACC if metric == 'accuracy' else TOL_AUC
        ok = abs(delta) <= tol
        rows.append({
            'task': task, 'model': model, 'metric': metric,
            'ours': round(ours, 4), 'pub': pub,
            'delta': round(delta, 4), 'tol': tol,
            'within_tol': ok,
        })

    out = pd.DataFrame(rows)
    out.to_csv(f'results/replication_vs_published_{args.geneset}.csv', index=False)
    print(out.to_string(index=False))
    print()
    print(f'Within tolerance: {out["within_tol"].sum()}/{len(out)} '
          f'(AUC tol={TOL_AUC}, accuracy tol={TOL_ACC})')

    # new models have no published target
    new_models = ['elastic_net', 'XGBoost']
    new = df[df['model'].isin(new_models)][['task', 'model', 'metric', 'mean', 'std']].copy()
    new['mean'] = new['mean'].round(4)
    new['std'] = new['std'].round(4)
    print('\nNew models (no published target):')
    print(new.to_string(index=False))


if __name__ == '__main__':
    main()
