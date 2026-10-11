"""Feature definitions and their numpy evaluator (nm; cn uses the PBC minimum image, others none)."""
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.lib.mdamath import triclinic_box

KINDS = ("pair", "contacts", "centroid_distance", "centroid_count", "cos", "cn")
CN_R0, CN_DMAX, CN_NM, CN_DELTA = 0.35, 1.2, (6, 10), 1e-3
CONTACT_CUTOFF_FACTOR = 4.0
CONTACT_SWITCH_FACTOR = 3.5
COUNT_CUTOFF_FACTOR = 2.5
COUNT_SWITCH_FACTOR = 2.0


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
    terms: Optional[list] = None
    exclude: Optional[list] = None

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
        if self.terms is not None:
            self.terms = [[[int(i) for i in a], [int(i) for i in b], float(w)] for a, b, w in self.terms]
            for a, b, _ in self.terms:
                if len(set(a)) != len(a) or len(set(b)) != len(b):
                    raise ValueError(f"feature {self.name!r}: repeated atom index inside a term side is not supported")
        if self.exclude is not None:
            self.exclude = [sorted(int(i) for i in p) for p in self.exclude]
        self.r0 = float(self.r0)


def features_to_dict(features) -> dict:
    return {"features_json": np.array([json.dumps(asdict(f)) for f in features])}


def features_from_dict(d) -> list:
    return [Feature(**json.loads(str(s))) for s in d["features_json"]]


def _switch(r, r0, switch=CONTACT_SWITCH_FACTOR, cutoff=CONTACT_CUTOFF_FACTOR):
    rs, rc = switch * r0, cutoff * r0
    x = np.clip((r - rs) / (rc - rs), 0.0, 1.0)
    return np.where(r >= rc, 0.0, 1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5)


def _rational(x, n, m):
    # (1-x^n)/(1-x^m); first-order expansion within CN_DELTA of x = 1 (0/0 there)
    x = np.asarray(x, dtype=float)
    near = np.abs(x - 1.0) < CN_DELTA
    xs = np.where(near, 0.5, x)
    out = (1 - xs ** n) / (1 - xs ** m)
    lim = (n / m) * (1 + 0.5 * (n - m) * (x - 1.0))
    return np.where(near, lim, out)


def cn_switch(r, r0=CN_R0, dmax=CN_DMAX, n=CN_NM[0], m=CN_NM[1]):
    """Shifted rational coordination switching, exactly 0 for r >= dmax (nm)."""
    r = np.asarray(r, dtype=float)
    c = _rational(dmax / r0, n, m)
    return np.where(r < dmax, (_rational(r / r0, n, m) - c) / (1 - c), 0.0)


_IMAGES = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


def _min_image_dist(dv, box):
    # float64 exact minimum image: reduce to the home cell, then search 27 images
    dv = dv - np.round(dv @ np.linalg.inv(box)) @ box
    cand = dv[:, None, :] + (_IMAGES @ box)[None]
    return np.linalg.norm(cand, axis=-1).min(axis=1)


def _cn(f, X, box):
    n_at = X.shape[1]
    ex = np.array(f.exclude or [], dtype=np.int64).reshape(-1, 2)
    excl = np.concatenate([ex[:, 0] * n_at + ex[:, 1], ex[:, 1] * n_at + ex[:, 0]])
    out = np.zeros(len(X))
    for t in range(len(X)):
        bt = np.asarray(box[t], dtype=float)
        if CN_DMAX >= 0.95 * bt[2, 2] / 2:
            raise ValueError("cn: box too small for d_MAX (d_MAX < 0.95 * c_z/2 required); not supported")
        dims = triclinic_box(*(bt * 10.0))
        for a, b, w in f.terms:
            ia, ib = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
            pairs = capped_distance(X[t, ia] * 10.0, X[t, ib] * 10.0, max_cutoff=CN_DMAX * 10.0 + 1e-3,
                                    box=dims, method="bruteforce", return_distances=False)
            if not len(pairs):
                continue
            gi, gj = ia[pairs[:, 0]], ib[pairs[:, 1]]
            keep = gi != gj
            if len(excl):
                keep &= ~np.isin(gi * n_at + gj, excl)
            gi, gj = gi[keep], gj[keep]
            d = _min_image_dist(X[t, gj] - X[t, gi], bt)
            out[t] += w * cn_switch(d).sum()
    return out


def _one(f, X, box=None):
    if f.kind == "cn":
        if box is None:
            raise ValueError("cn features need box vectors")
        return _cn(f, X, box)
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
        s = _switch(r, f.r0, COUNT_SWITCH_FACTOR, COUNT_CUTOFF_FACTOR)
        return np.sum(s / (1 + (r / f.r0) ** 8), axis=1)
    if f.kind == "cos":
        c1, c2, c3, c4 = (X[:, g].mean(1) for g in f.groups)
        a, b = c2 - c1, c4 - c3
        return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    raise ValueError(f"unknown feature kind {f.kind!r}; expected one of {KINDS}")


def evaluate(features, X, box=None) -> np.ndarray:
    """(n_frames, n_atoms, 3) nm -> (n_frames, n_features); box (F, 3, 3) or (3, 3) nm, needed by cn."""
    X = np.asarray(X, dtype=float)
    if box is not None:
        box = np.asarray(box, dtype=float)
        if box.ndim == 2:
            box = np.broadcast_to(box, (len(X), 3, 3))
        elif len(box) != len(X):
            raise ValueError(f"box has {len(box)} frames but X has {len(X)}")
    return np.stack([_one(f, X, box) for f in features], axis=1)
