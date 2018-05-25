import pandas as pd
import numpy as np

from deepimpute.util import log1x, exp1x, libNorm


class Normalizer(object):

    def __init__(self, name="", factorFn=None, activations=[None, None], **params):
        self.name = name
        self._activation = activations[0]
        self._revActivation = activations[1]
        self._factorFn = factorFn
        self.factor = None
        for key, val in params.items():
            setattr(self, key, val)

    @classmethod
    def fromName(cls, name):
        if name is None:
            return cls()
        elif name == "log_or_exp":
            return cls(name=name, activations=[log1x, exp1x])
        elif name == "libSizeLog":
            return cls(name=name, activations=[log1x, exp1x], factorFn=libNorm)
        else:
            print("Unknown method " + name)

    def copy(self):
        return Normalizer(
            name=self.name,
            factorFn=self.factorFn,
            activations=[self._activation, self._revActivation],
        )

    @property
    def factorFn(self):
        if self._factorFn is None:

            def one(x):
                return 1

            return one
        else:
            return self._factorFn

    @factorFn.setter
    def factorFn(self, value):
        self._factorFn = value

    @property
    def activation(self):
        if self._activation is None:
            return lambda x: x
        else:
            return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def revActivation(self):
        if self._activation is None:
            return lambda x: x
        else:
            return self._revActivation

    @revActivation.setter
    def revActivation(self, value):
        self._revActivation = value

    def fit(self, X):
        self.factor = np.apply_along_axis(self.factorFn, 1, np.array(X)).astype(float)
        return self

    def transform(self, X, rev=False):
        X_np = np.array(X)
        if rev:
            X_norm = np.divide(self.revActivation(X_np).T, self.factor).T
        else:
            X_norm = self.activation(np.multiply(X_np.T, self.factor).T)
        if type(X) is pd.DataFrame:
            return pd.DataFrame(X_norm, index=X.index, columns=X.columns)
        else:
            return X_norm
