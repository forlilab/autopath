"""Reducers: standardised features -> CV weights. HLDA (Piccini/Mendels/Parrinello 2018)."""
from typing import Protocol

import numpy as np
from scipy.linalg import eigh


class Reducer(Protocol):
    def fit(self, X: np.ndarray, labels: np.ndarray, order: list) -> np.ndarray: ...


def _orient(W, X, labels, order):
    W = W / np.linalg.norm(W, axis=0)
    for k in range(W.shape[1]):
        s = X @ W[:, k]
        if s[labels == order[0]].mean() > s[labels == order[-1]].mean():
            W[:, k] = -W[:, k]
    return W


class HLDA:
    """ridge is in units of the pooled-standardised feature variance.

    S_b is centred on the UNWEIGHTED mean of class means, not the overall data mean of Capelli et al. 2019
    SI eq. 3: sMD frame counts per class reflect the pulling protocol, not physics (identical for 2 classes)."""

    def __init__(self, n_components=2, ridge=0.1):
        self.n_components, self.ridge = n_components, ridge

    def fit(self, X, labels, order):
        X = np.asarray(X, float)
        n = X.shape[1]
        classes = [c for c in order if np.any(labels == c)]
        mus = np.array([X[labels == c].mean(0) for c in classes])
        prec = sum(np.linalg.inv(np.cov(X[labels == c].T) + self.ridge * np.eye(n)) for c in classes)
        Sw = np.linalg.inv(prec)
        d = mus - mus.mean(0)
        Sb = d.T @ d
        lam, V = eigh(Sb, Sw + self.ridge * np.eye(n))
        k = min(self.n_components, len(classes) - 1)
        return _orient(V[:, np.argsort(lam)[::-1][:k]].copy(), X, labels, order)


def lda_reference(X, labels, n_components, order):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=min(n_components, len(order) - 1)).fit(X, labels)
    return _orient(lda.scalings_[:, :n_components].copy(), X, labels, order)
