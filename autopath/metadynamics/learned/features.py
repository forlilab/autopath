"""Feature definitions and their numpy evaluator (nm, no PBC: aligned, imaged frames)."""
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

KINDS = ("pair", "contacts", "centroid_distance", "centroid_count", "cos")
CONTACT_CUTOFF_FACTOR = 4.0
CONTACT_SWITCH_FACTOR = 3.5


@dataclass
class Feature:
    name: str
    kind: str
    family: str
    pairs: Optional[list] = None
    group_a: Optional[list] = None
    group_b: Optional[list] = None
    groups: Optional[list] = None
    r0: float = 0.0

    def __post_init__(self):
        # JSON round trips turn tuples into lists; normalise so equality holds
        if self.pairs is not None:
            self.pairs = [tuple(int(i) for i in p) for p in self.pairs]
        for k in ("group_a", "group_b"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, [int(i) for i in v])
        if self.groups is not None:
            self.groups = [[int(i) for i in g] for g in self.groups]
        self.r0 = float(self.r0)


def features_to_dict(features) -> dict:
    return {"features_json": np.array([json.dumps(asdict(f)) for f in features])}


def features_from_dict(d) -> list:
    return [Feature(**json.loads(str(s))) for s in d["features_json"]]


def _switch(r, r0):
    rs, rc = CONTACT_SWITCH_FACTOR * r0, CONTACT_CUTOFF_FACTOR * r0
    x = np.clip((r - rs) / (rc - rs), 0.0, 1.0)
    return np.where(r >= rc, 0.0, 1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5)


def _one(f, X):
    if f.kind == "pair":
        i, j = np.array(f.pairs).T
        r = np.linalg.norm(X[:, j] - X[:, i], axis=-1)
        return np.sum(1 / (1 + (r / f.r0) ** 6), axis=1)
    if f.kind == "contacts":
        a, b = X[:, f.group_a], X[:, f.group_b]
        r = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)
        return np.sum(_switch(r, f.r0) / (1 + (r / f.r0) ** 6), axis=(1, 2))
    if f.kind == "centroid_distance":
        c1, c2 = (X[:, g].mean(1) for g in f.groups)
        return np.linalg.norm(c2 - c1, axis=-1)
    if f.kind == "centroid_count":
        c = X[:, f.groups[0]].mean(1)
        r = np.linalg.norm(X[:, f.group_b] - c[:, None, :], axis=-1)
        return np.sum(1 / (1 + (r / f.r0) ** 8), axis=1)
    if f.kind == "cos":
        c1, c2, c3, c4 = (X[:, g].mean(1) for g in f.groups)
        a, b = c2 - c1, c4 - c3
        return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    raise ValueError(f"unknown feature kind {f.kind!r}; expected one of {KINDS}")


def evaluate(features, X) -> np.ndarray:
    """(n_frames, n_atoms, 3) nm -> (n_frames, n_features)."""
    X = np.asarray(X, dtype=float)
    return np.stack([_one(f, X) for f in features], axis=1)
