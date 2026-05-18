import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        sin = np.sin(2 * np.pi * X_arr / self.period)
        cos = np.cos(2 * np.pi * X_arr / self.period)
        return np.hstack([sin, cos])

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else ["feat"]
        return np.array([f"{n}_{fn}" for n in base for fn in ("sin", "cos")])