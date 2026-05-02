# Zachary Gray
#
# extra classifier wrappers matching the original LogisticRegressor_skl /
# RandomForestClassifier_skl interface used by CVmodel.
# fit(X_train, Y_train, X_validate, Y_validate, cv_param), predict, predict_class.

import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier


# true iff xgboost was built with cuda AND a gpu is visible
# RLT_FORCE_CPU=1 disables even when present (for cpu/gpu benchmarking)
def cuda_available():
    if os.environ.get('RLT_FORCE_CPU') == '1':
        return False
    try:
        import xgboost as xgb
        build_info = xgb.build_info()
        if not build_info.get('USE_CUDA', False):
            return False
        # tiny gpu op to confirm a device is actually usable
        import numpy as np
        dtest = xgb.DMatrix(np.zeros((2, 2)), label=np.array([0, 1]))
        xgb.train({'device': 'cuda', 'tree_method': 'hist'}, dtest, num_boost_round=1)
        return True
    except Exception:
        return False


# k-NN wrapper. cv_param = n_neighbors (int).
class KNNClassifier_skl:

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


# elastic-net LR via SGDClassifier(loss='log_loss', penalty='elasticnet').
# matches Smith et al. (2020) p.15: LR "trained via stochastic gradient descent".
# cv_param = (C, l1_ratio).
#
# tried LogisticRegression(solver='saga', penalty='elasticnet') first, ~400x slower
# per fit, full grid would take ~25h on OT alone. saga's L1 path also leaves many tiny
# nonzeros that only collapse at convergence (sklearn issue #21196), so SGD's coefs
# are actually sparser in practice.
#
# C -> alpha mapping: alpha = 1/(C * n_samples), the conventional bridge between
# LogisticRegression's C and SGD's regularization scale.
class ElasticNetLR_skl:

    def __init__(self, **kwargs):
        kwargs.setdefault('loss', 'log_loss')
        kwargs.setdefault('penalty', 'elasticnet')
        kwargs.setdefault('max_iter', 5000)
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


# xgboost wrapper, gpu-aware: device='cuda' if cuda_available() else 'cpu'.
# cv_param = (max_depth, n_estimators, learning_rate).
class XGBClassifier_skl:

    def __init__(self, **kwargs):
        kwargs.setdefault('subsample', 0.8)
        kwargs.setdefault('tree_method', 'hist')
        # max_bin=128 (default 256) halves histogram VRAM. default OOM'd partway
        # through COAD on the all-geneset run with max_depth=9 x 57992 features
        # on the 5060 Ti's 16GB.
        kwargs.setdefault('max_bin', 128)
        kwargs.setdefault('eval_metric', 'logloss')
        kwargs.setdefault('verbosity', 0)
        if 'device' not in kwargs:
            kwargs['device'] = 'cuda' if cuda_available() else 'cpu'
        # cap threads only on cpu; gpu fits don't oversubscribe via this knob
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
