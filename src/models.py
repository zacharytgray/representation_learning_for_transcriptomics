"""
Extra classifier wrappers matching the upstream LogisticRegressor_skl /
RandomForestClassifier_skl interface used by CVmodel:

    fit(X_train, Y_train, X_validate=None, Y_validate=None, cv_param=..., **fit_kwargs)
    predict(X)         -> class probabilities
    predict_class(X)   -> integer class labels

Three new models for the project extension:
  * KNNClassifier_skl        — paper baseline, missing from upstream repo
  * ElasticNetLR_skl         — new: l1+l2 logistic regression
  * XGBClassifier_skl        — new: gradient-boosted trees (GPU-aware)

For models with multiple hyperparameters (elastic net, xgboost), cv_param is a
tuple unpacked inside fit(). CVmodel still iterates a flat list of "params"
under one cv_param_name (e.g. "config" or "(C, l1_ratio)").
"""

import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier


def cuda_available():
    """
    True iff XGBoost was built with CUDA AND a GPU is visible.
    Set RLT_FORCE_CPU=1 to disable even when present (useful for benchmarking).
    """
    if os.environ.get('RLT_FORCE_CPU') == '1':
        return False
    try:
        import xgboost as xgb
        build_info = xgb.build_info()
        if not build_info.get('USE_CUDA', False):
            return False
        # try a tiny GPU op to confirm a device is actually usable
        import numpy as np
        dtest = xgb.DMatrix(np.zeros((2, 2)), label=np.array([0, 1]))
        xgb.train({'device': 'cuda', 'tree_method': 'hist'}, dtest, num_boost_round=1)
        return True
    except Exception:
        return False


class KNNClassifier_skl:
    """k-NN wrapper. cv_param = n_neighbors (int)."""

    def __init__(self, **kwargs):
        self.model_kwargs = kwargs
        self.model = None

    def fit(self, X_train, Y_train, X_validate=None, Y_validate=None,
            cv_param=5, **fit_kwargs):
        self.model = KNeighborsClassifier(n_neighbors=int(cv_param), **self.model_kwargs)
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        assert self.model is not None
        return self.model.predict_proba(X)

    def predict_class(self, X):
        assert self.model is not None
        return self.model.predict(X)


class ElasticNetLR_skl:
    """
    Elastic-net logistic regression wrapper.
    cv_param = (C, l1_ratio) tuple — same interface as the LR wrappers.

    Uses SGDClassifier(loss='log_loss', penalty='elasticnet') under the hood.
    This matches the optimizer the Smith et al. (2020) paper text describes
    for LR (SGD-trained). LogisticRegression(solver='saga', penalty='elasticnet')
    was tried first but is ~400x slower per fit and made the full grid infeasible
    even on the OT geneset. SGD also produces genuinely sparse coefficients
    (saga's L1 path leaves many tiny non-zeros that saga calls "zero" only at
    convergence — sklearn issue #21196).

    C -> alpha mapping: SGD's alpha is 1/(C * n_samples), the conventional
    bridge between LogisticRegression's C and SGD's regularization scale.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('loss', 'log_loss')
        kwargs.setdefault('penalty', 'elasticnet')
        kwargs.setdefault('max_iter', 5000)
        # tol left at sklearn default (1e-3 for SGDClassifier)
        self.model_kwargs = kwargs
        self.model = None

    def fit(self, X_train, Y_train, X_validate=None, Y_validate=None,
            cv_param=(1.0, 0.5), **fit_kwargs):
        C, l1_ratio = cv_param
        alpha = 1.0 / (C * len(Y_train))
        self.model = SGDClassifier(
            alpha=alpha, l1_ratio=l1_ratio, **self.model_kwargs)
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        assert self.model is not None
        return self.model.predict_proba(X)

    def predict_class(self, X):
        assert self.model is not None
        return self.model.predict(X)


class XGBClassifier_skl:
    """
    XGBoost classifier wrapper. GPU-aware: device='cuda' is auto-selected when
    cuda_available() returns True, else CPU 'hist'.

    cv_param = (max_depth, n_estimators, learning_rate) tuple.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('subsample', 0.8)
        kwargs.setdefault('tree_method', 'hist')
        # max_bin=128 (default 256) halves histogram VRAM. Required to fit
        # max_depth=9 × 57992-feature trees in 16 GB on the 5060 Ti — the
        # default OOM'd partway through COAD on the all-geneset run.
        kwargs.setdefault('max_bin', 128)
        kwargs.setdefault('eval_metric', 'logloss')
        kwargs.setdefault('verbosity', 0)
        if 'device' not in kwargs:
            kwargs['device'] = 'cuda' if cuda_available() else 'cpu'
        # cap threads only on cpu — gpu fits don't oversubscribe via this knob
        if kwargs['device'] == 'cpu':
            kwargs.setdefault('n_jobs', 4)
        self.model_kwargs = kwargs
        self.model = None

    def fit(self, X_train, Y_train, X_validate=None, Y_validate=None,
            cv_param=(6, 100, 0.1), **fit_kwargs):
        max_depth, n_estimators, learning_rate = cv_param
        self.model = XGBClassifier(
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            **self.model_kwargs,
        )
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        assert self.model is not None
        return self.model.predict_proba(X)

    def predict_class(self, X):
        assert self.model is not None
        return self.model.predict(X)
