"""HILLS v2: every deposited hill plus what is needed to rebuild the bias.

``bias_from_hills`` rebuilds the grid bias exactly as OpenMM deposits it
(per-axis Gaussians on the grid, periodic wrap, McGovern–de Pablo factor), like
PLUMED ``sum_hills``. A walker's HILLS file holds its own hills only; walkers
sharing a bias directory are summed, plus a preseed ``bias_0_0.npy`` if present.
"""

import glob
import os
from types import SimpleNamespace

import numpy as np
import openmm.unit as unit

from .BoundaryCorrection import boundary_factor

HILLS_VERSION = 2
# every key hills_record (with record intervals and TT extras) writes
HILLS_KEYS = frozenset({
    "hills_version", "run_id", "finished", "bias_dir", "started_at", "time_ns", "wall_s", "cv", "height_kj",
    "grid_min", "grid_max", "grid_points", "sigma", "periodic", "bounds_lo", "bounds_hi", "bias_factor",
    "temperature", "tempering", "cv_names", "cv_length_per_unit_ang", "origin", "record_dt_ns", "dcd_interval_ns",
    "vstar_kj", "tt_basins", "tt_threshold", "tt_bias_factor"})


def hills_record(meta, run_id, started_at, ps_per_step, specs, tempering, finished=False,
                 **intervals) -> dict:
    """HILLS v2 arrays for ``np.savez`` from a (BoundaryCorrected|TransitionTempered)Metadynamics."""
    n = len(meta.variables)
    h = np.array(meta.hills, dtype=float).reshape(-1, n + 2)
    lo = np.array([np.nan if b is None else b[0] for b in meta.bounds], dtype=float)
    hi = np.array([np.nan if b is None else b[1] for b in meta.bounds], dtype=float)
    rec = dict(
        hills_version=HILLS_VERSION, run_id=str(run_id), finished=bool(finished),
        bias_dir=os.path.abspath(meta.biasDir) if meta.biasDir else "",
        started_at=float(started_at),
        time_ns=h[:, 0] * ps_per_step / 1000.0, wall_s=np.array(meta.hill_wall, dtype=float),
        cv=h[:, 1:-1], height_kj=h[:, -1],
        grid_min=np.array([v.minValue for v in meta.variables], dtype=float),
        grid_max=np.array([v.maxValue for v in meta.variables], dtype=float),
        grid_points=np.array([v.gridWidth for v in meta.variables], dtype=int),
        sigma=np.array([v.biasWidth for v in meta.variables], dtype=float),
        periodic=np.array([bool(v.periodic) for v in meta.variables]),
        bounds_lo=lo, bounds_hi=hi,                   # NaN = no edge on that CV
        bias_factor=float(meta.biasFactor),
        temperature=float(meta.temperature.value_in_unit(unit.kelvin)),
        tempering=str(tempering),
        cv_names=np.array([s.name for s in specs]),
        cv_length_per_unit_ang=np.array([np.nan if s.cv_length_per_unit_ang is None
                                         else s.cv_length_per_unit_ang for s in specs], dtype=float),
        origin=np.array([np.nan if s.origin is None else s.origin for s in specs], dtype=float),
        **intervals)
    if hasattr(meta, "vstar_log"):
        rec.update(vstar_kj=np.array(meta.vstar_log, dtype=float),
                   tt_basins=np.array(meta.basin_points, dtype=float),
                   tt_threshold=float(meta.threshold),
                   tt_bias_factor=np.nan if meta.ttBiasFactor is None else float(meta.ttBiasFactor))
    return rec


def load_hills(out_dir) -> list:
    """All HILLS v2 walkers in ``out_dir`` (older files lack the metadata and are skipped)."""
    walkers = []
    for f in sorted(glob.glob(os.path.join(out_dir, "HILLS_*.npz"))):
        with np.load(f) as d:
            if "hills_version" not in d.files or int(d["hills_version"]) < HILLS_VERSION:
                continue
            w = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
        w["file"] = f
        walkers.append(w)
    return walkers


def _axes(w):
    return [SimpleNamespace(minValue=float(a), maxValue=float(b), gridWidth=int(n), periodic=bool(p),
                            biasWidth=float(s), _scaledVariance=(float(s) / (float(b) - float(a))) ** 2)
            for a, b, n, p, s in zip(w["grid_min"], w["grid_max"], w["grid_points"],
                                     w["periodic"], w["sigma"])]


def _bounds(w):
    return [None if np.isnan(lo) else (float(lo), float(hi))
            for lo, hi in zip(w["bounds_lo"], w["bounds_hi"])]


def _axis_gaussians(v, x):
    """(n_hills, gridWidth) Gaussians along one axis, as Metadynamics._addGaussian."""
    s = (np.asarray(x, dtype=float) - v.minValue) / (v.maxValue - v.minValue)
    if v.periodic:
        s = s % 1.0
    dist = np.abs(np.linspace(0, 1.0, num=v.gridWidth)[None, :] - s[:, None])
    if v.periodic:
        dist = np.minimum(dist, np.abs(dist - 1))
        dist[:, -1] = dist[:, 0]
    return np.exp(-0.5 * dist * dist / v._scaledVariance)


def _sum_hills(axes, cv, height, chunk=4096):
    """Sum of h * prod_axis g_axis on the grid (axes in reverse variable order, like OpenMM)."""
    out = np.zeros(tuple(v.gridWidth for v in reversed(axes)))
    cv = np.asarray(cv, dtype=float).reshape(len(height), len(axes))
    for i in range(0, len(height), chunk):
        h = np.asarray(height[i:i + chunk], dtype=float)
        g = [_axis_gaussians(v, cv[i:i + chunk, k]) for k, v in enumerate(axes)]
        if len(axes) == 1:
            out += h @ g[0]
        elif len(axes) == 2:
            out += g[1].T @ (h[:, None] * g[0])
        else:
            out += np.einsum("k,kz,ky,kx->zyx", h, g[2], g[1], g[0], optimize=True)
    return out


def _check_same_grid(walkers):
    ref = walkers[0]
    for w in walkers[1:]:
        for k in ("grid_min", "grid_max", "grid_points", "sigma", "periodic"):
            if not np.array_equal(np.asarray(w[k]), np.asarray(ref[k])):
                raise ValueError(f"{w['run_id']} and {ref['run_id']} differ in {k}")
        if not np.array_equal(np.asarray(w["bounds_lo"]), np.asarray(ref["bounds_lo"]), equal_nan=True) or \
                not np.array_equal(np.asarray(w["bounds_hi"]), np.asarray(ref["bounds_hi"]), equal_nan=True):
            raise ValueError(f"{w['run_id']} and {ref['run_id']} differ in boundary correction")


def _preseed(walkers):
    total = 0.0
    for d in {w["bias_dir"] for w in walkers if w["bias_dir"]}:
        p = os.path.join(d, "bias_0_0.npy")
        if os.path.exists(p):
            total = total + np.load(p)
    return total


def bias_from_hills(walkers, until=None) -> np.ndarray:
    """Total bias (kJ/mol) on the grid from the hills of ``walkers``.

    ``until``: None (all hills), a wall-clock time (hills with ``wall_s`` <= it),
    or a dict ``{run_id: t_ns}`` (each walker's hills up to its own time).
    """
    _check_same_grid(walkers)
    axes = _axes(walkers[0])
    cv, height = [], []
    for w in walkers:
        keep = np.ones(len(w["height_kj"]), dtype=bool)
        if isinstance(until, dict):
            keep = w["time_ns"] <= until.get(w["run_id"], np.inf)
        elif until is not None:
            keep = w["wall_s"] <= until
        cv.append(np.asarray(w["cv"]).reshape(len(keep), len(axes))[keep])
        height.append(np.asarray(w["height_kj"])[keep])
    cv, height = np.concatenate(cv), np.concatenate(height)
    return _sum_hills(axes, cv, height) * boundary_factor(axes, _bounds(walkers[0])) + _preseed(walkers)


def fes_scale(walker) -> float:
    """-F/V: 1 for transition tempering alone, γ/(γ-1) when well tempering is on."""
    if walker["tempering"] == "tt":
        return 1.0
    g = float(walker["bias_factor"])
    return g / (g - 1.0)


def fes_history(walkers, n=60, cuts=None) -> list:
    """FES snapshots per bias (walkers grouped by ``bias_dir``).

    Cuts are wall-clock times, so concurrent, sequential and staggered shared
    walkers are all handled in deposition order; x is the summed walker time
    reached at each cut (one walker: its own time). ``cuts`` (wall-clock times)
    replaces the ``n`` evenly spaced cuts; use with a single bias. Returns one dict per bias:
    ``run_ids``, ``bias_dir``, ``wall_s``, ``t_ns`` (n,) and ``fes`` (n, *grid).
    """
    groups = {}
    for w in walkers:
        groups.setdefault(w["bias_dir"] or w["file"], []).append(w)
    out = []
    for key, ws in groups.items():
        ws = [w for w in ws if len(w["height_kj"])]
        if not ws:
            continue
        _check_same_grid(ws)
        axes = _axes(ws[0])
        wall = np.concatenate([w["wall_s"] for w in ws])
        cv = np.concatenate([np.asarray(w["cv"]).reshape(-1, len(axes)) for w in ws])
        height = np.concatenate([w["height_kj"] for w in ws])
        order = np.argsort(wall, kind="stable")
        wall, cv, height = wall[order], cv[order], height[order]
        cs = np.linspace(wall[0], wall[-1], n + 1)[1:] if cuts is None else np.asarray(cuts, float)
        if cuts is None:
            cs[-1] = wall[-1]
        bf = boundary_factor(axes, _bounds(ws[0]))
        pre = _preseed(ws)
        scale = fes_scale(ws[0])
        acc = np.zeros(tuple(v.gridWidth for v in reversed(axes)))
        fes, t_ns, start = [], [], 0
        for c in cs:
            stop = int(np.searchsorted(wall, c, side="right"))
            if stop > start:
                acc += _sum_hills(axes, cv[start:stop], height[start:stop])
                start = stop
            fes.append(-scale * (acc * bf + pre))
            t_ns.append(sum(float(w["time_ns"][w["wall_s"] <= c][-1]) if np.any(w["wall_s"] <= c) else 0.0
                            for w in ws))
        out.append(dict(bias_dir=key, run_ids=[w["run_id"] for w in ws], wall_s=cs,
                        t_ns=np.array(t_ns), fes=np.array(fes)))
    return out
