"""Bands, recrossings, reweighting, the shared/isolated combination rule, free energies and ΔG°."""
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from openmm import unit

from ..BoundaryCorrection import boundary_factor
from ..Hills import _axes, _axis_gaussians, _bounds, bias_from_hills, fes_history, fes_scale

FUNNEL_Z = "funnel_z"
SITE_FRAC = 0.2
BULK_MARGIN_ANG = 1.5
EDGE_FRAC = 0.1
HIGH_BOUND_PREFIX = "contacts"

ESTIMATORS = ("rw_ct", "rw_final", "rw_balanced", "bias")
LABELS = {"rw_ct": "reweight c(t)", "rw_final": "reweight final bias", "rw_balanced": "reweight balanced",
          "bias": "bias-derived"}
PRIMARY = "rw_ct"


@dataclass
class Band:
    bound: tuple
    unbound: tuple
    bound_end: str
    source: str


def _ang(fp, key):
    """Funnel quantity ``fp[key]`` in Å."""
    return float(fp[key].value_in_unit(unit.angstrom))


def _spec(run, name):
    return next((c for c in run.cvs if c.name == name), None)


def ang_per_unit(run, name):
    """Å per CV unit: the HILLS ``cv_length_per_unit_ang``, else 10 (nm) for funnel z, else None."""
    spec = _spec(run, name)
    if spec is not None and spec.ang_per_unit:
        return spec.ang_per_unit
    return 10.0 if name == FUNNEL_Z else None


def length_unit(run, name):
    """``"nm"``, ``"Å"`` or None (no or another length scale)."""
    return {10.0: "nm", 1.0: "Å"}.get(ang_per_unit(run, name))


def funnel_z_low(run, a):
    """Lowest funnel z (Å) for the bound band and the funnel outline.

    In order: the funnel ``z_low``; the biased funnel z grid minimum; the observed minimum (monitored z).
    ``a`` is Å per CV unit.
    """
    if "z_low" in run.funnel:
        return _ang(run.funnel, "z_low")
    spec = _spec(run, FUNNEL_Z)
    if spec is not None:
        return spec.grid_min * a
    return float(min(np.min(w.cvs[FUNNEL_Z]) for w in run.walkers)) * a


def _bound_high(spec, name):
    """Contact counts are bound at their high end: from the recorded CV type, else the name prefix."""
    kind = spec.kind if spec is not None else None
    return (kind if kind is not None else name).startswith(HIGH_BOUND_PREFIX)


def _default_band(run, name):
    spec = _spec(run, name)
    if name == FUNNEL_Z and run.funnel is not None:
        a = ang_per_unit(run, name)
        z_low = funnel_z_low(run, a)
        zmax, zcc = _ang(run.funnel, "z_max"), _ang(run.funnel, "z_cc")
        return Band((z_low / a, SITE_FRAC * zmax / a), (zcc / a, (zmax - BULK_MARGIN_ANG) / a), "low", "funnel")
    lcv = run.learned_cv
    if name == "learned_sw" and lcv is not None and "basin_B" in lcv and "basin_U" in lcv:
        # basins are class medians; hill width is half the bound-class SD, so ±2 widths = ±1 SD (σ_U not stored)
        w = 2.0 * float(lcv["sw_hill_width"])
        b, u = float(np.atleast_1d(lcv["basin_B"])[-1]), float(np.atleast_1d(lcv["basin_U"])[-1])
        return Band((b - w, b + w), (u - w, u + w), "low" if b < u else "high", "learned_cv")
    if spec is None:
        allv = np.concatenate([w.cvs[name] for w in run.walkers])
        lo, hi, src = float(np.min(allv)), float(np.max(allv)), "observed"
    else:
        lo, hi, src = spec.grid_min, spec.grid_max, "grid"
        if spec.bounds is not None:
            blo, bhi = spec.bounds
            lo, hi = (blo if np.isfinite(blo) else lo), (bhi if np.isfinite(bhi) else hi)
            src = "clamp" if np.isfinite(blo) or np.isfinite(bhi) else src
    e = EDGE_FRAC * (hi - lo)
    low, high = (lo, lo + e), (hi - e, hi)
    if _bound_high(spec, name):
        return Band(high, low, "high", src)
    return Band(low, high, "low", src)


def bands(run, overrides=None) -> dict:
    """Bound / unbound interval per CV (biased and monitored), in CV units.

    Defaults: funnel z from the funnel geometry; a learned CV from its B/U basins ± 2 hill widths; otherwise
    the lowest/highest ``EDGE_FRAC`` of the clamp bounds (finite edges), else of the grid (observed range for
    a monitored CV). The bound end is high for contact counts (CV type ``contacts`` recorded in the HILLS
    metadata, else a name starting with ``contacts``) and low otherwise. ``overrides`` maps a CV name to any
    of ``bound``, ``unbound``, ``bound_end``.
    """
    out = {}
    for name in [c.name for c in run.cvs] + list(run.monitored):
        b = _default_band(run, name)
        ov = (overrides or {}).get(name, {})
        if ov.get("bound_end") and ov["bound_end"] != b.bound_end:
            b = Band(b.unbound, b.bound, ov["bound_end"], b.source)
        if "bound" in ov or "unbound" in ov:
            b = Band(tuple(ov.get("bound", b.bound)), tuple(ov.get("unbound", b.unbound)), b.bound_end, "override")
        out[name] = b
    return out


def recrossings(walker, cv, band) -> dict:
    """Full bound→unbound→bound cycles and time fractions in each band."""
    x = np.asarray(walker.cvs[cv])
    inb = (x >= band.bound[0]) & (x <= band.bound[1])
    inu = (x >= band.unbound[0]) & (x <= band.unbound[1])
    state, n = 0, 0
    for b, u in zip(inb, inu):
        if b:
            if state == 2:
                n += 1
            state = 1
        elif u and state == 1:
            state = 2
    return dict(n_cycles=int(n), frac_bound=float(inb.mean()), frac_unbound=float(inu.mean()))


KT_PER_K = 0.0083144626


def clamp_fractions(run) -> dict:
    """Fraction of hills deposited at a CV bound: ``{cv: {rid or "pooled": fraction}}``.

    Isolated walkers are reported one by one; shared and single runs have one bias and are pooled.
    CVs without bounds are omitted.
    """
    out = {}
    pooled = run.layout != "isolated"
    for j, c in enumerate(run.cvs):
        if c.bounds is None:
            continue
        lo, hi = c.bounds
        tol = 1e-6 * (c.grid_max - c.grid_min)

        def frac(x):
            return float(np.mean((x <= lo + tol) | (x >= hi - tol)))
        if pooled:
            out[c.name] = {"pooled": frac(np.concatenate([w.hill_cv[:, j] for w in run.walkers]))}
        else:
            out[c.name] = {w.rid: frac(w.hill_cv[:, j]) for w in run.walkers}
    return out


@dataclass
class Combined:
    layout: str
    pooled: object = None
    per_walker: dict = field(default_factory=dict)
    mean: Optional[float] = None
    sd: Optional[float] = None
    spread: Optional[float] = None
    n_walkers: Optional[int] = None
    n_defined: Optional[int] = None
    ess: object = None
    n_masked: object = None
    kept_frac: object = None
    bin_info: object = None


def kish_ess(w) -> float:
    """Kish effective sample size (Σw)²/Σw²; 0 for no frames or no weight."""
    w = np.asarray(w, float)
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w)) ** 2 / s2 if s2 > 0 else 0.0


def _weights_arrays(bias, ct, temperature, burn=0.2, normalise=True):
    """Kept-frame mask and c(t) weights exp[(V - c)/kT]: normalised, or log-weights if not ``normalise``."""
    if bias is None:
        raise ValueError("no REWEIGHT data: reweighting not supported")
    n = len(bias)
    m = np.arange(n) >= int(burn * n)
    lw = (bias[m] - ct[m]) / (KT_PER_K * temperature)
    if not normalise:
        return m, lw
    w = np.exp(lw - lw.max())
    return m, w / w.sum()


def weights(walker, temperature, burn=0.2):
    """c(t) reweighting weights over frames after the first ``burn`` fraction of the walker.

    Returns
    -------
    mask : ndarray of bool
    w : ndarray, normalised over the kept frames
    """
    return _weights_arrays(walker.bias, walker.ct, temperature, burn)


def _grids(h):
    """Grid coordinates of every CV of a HILLS record (first CV first)."""
    return [np.linspace(float(a), float(b), int(n)) for a, b, n in
            zip(np.atleast_1d(h["grid_min"]), np.atleast_1d(h["grid_max"]), np.atleast_1d(h["grid_points"]))]


_BIAS_CACHE = OrderedDict()
_BIAS_CACHE_MAX = 8


def _raw_key(raw):
    return tuple((id(h), id(h["cv"]), id(h["height_kj"]), id(h["time_ns"]), id(h["wall_s"])) for h in raw)


def final_bias(raw):
    """Final bias (kJ/mol) of one bias from its HILLS records, as ``bias_from_hills``; cached per record set.

    The cache keeps the records alive, so their ids identify them; do not modify records in place.
    """
    key = _raw_key(raw)
    hit = _BIAS_CACHE.get(key)
    if hit is None:
        hit = _BIAS_CACHE[key] = (list(raw), np.asarray(bias_from_hills(raw)))
        while len(_BIAS_CACHE) > _BIAS_CACHE_MAX:
            _BIAS_CACHE.popitem(last=False)
    else:
        _BIAS_CACHE.move_to_end(key)
    return hit[1]


def _bias_group(run, w):
    """Walkers sharing ``w``'s bias: ``[w]`` if isolated, else every loaded walker."""
    return [w] if run.layout == "isolated" else list(run.walkers)


def _group_raw(group):
    return [h for x in group for h in x.hills_raw]


def _final_at_frames(run, w):
    """Final bias of ``w``'s bias (kJ/mol) at each frame's logged CVs, linearly interpolated on the grid.

    Frames outside the grid are clipped to it on a CV with clamp bounds; on any other CV they raise.
    Periodic CVs are wrapped into one period.
    """
    raw = _group_raw(_bias_group(run, w))
    V = final_bias(raw)
    grids = _grids(raw[0])
    if V.shape != tuple(len(g) for g in reversed(grids)):
        raise ValueError(f"bias grid shape {V.shape} is not (last CV, ..., first CV): not supported")
    pts = []
    for c, g in zip(run.cvs, grids):
        x, lo, hi = np.asarray(w.cvs[c.name], float), g[0], g[-1]
        if c.periodic:
            x = lo + np.mod(x - lo, hi - lo)
        else:
            tol = 1e-9 * (hi - lo)
            if c.bounds is None and np.any((x < lo - tol) | (x > hi + tol)):
                raise ValueError(f"frame CV {c.name} outside the bias grid: not supported")
            x = np.clip(x, lo, hi)
        pts.append(x)
    return RegularGridInterpolator(tuple(reversed(grids)), V)(np.stack(pts[::-1], 1))


def _omega_weights(c, g):
    """Integration weights over Ω of a function linear between the grid points ``g``, and the length of Ω.

    Ω is the clamp range where recorded (finite edges, within the grid), else the grid; a periodic CV
    spans one period (its last grid point repeats the first).
    """
    n = len(g)
    if c.periodic:
        w = np.full(n, (g[-1] - g[0]) / (n - 1))
        w[-1] = 0.0
        return w, float(g[-1] - g[0])
    lo, hi = float(g[0]), float(g[-1])
    if c.bounds is not None:
        lo = max(lo, c.bounds[0]) if np.isfinite(c.bounds[0]) else lo
        hi = min(hi, c.bounds[1]) if np.isfinite(c.bounds[1]) else hi
    x0, x1 = g[:-1], g[1:]
    d = x1 - x0
    a, b = np.clip(lo, x0, x1), np.clip(hi, x0, x1)
    w = np.zeros(n)
    w[:-1] += ((x1 - a) ** 2 - (x1 - b) ** 2) / (2 * d)
    w[1:] += ((b - x0) ** 2 - (a - x0) ** 2) / (2 * d)
    return w, hi - lo


def hill_volumes(run, raw, chunk=4096):
    """Per-hill average over Ω of each deposited hill (boundary factor included), unit height.

    Times the hill height, this is the hill's contribution to the CV-space average of the bias.

    Returns
    -------
    ndarray
        One value per hill, records in order.
    """
    h0 = raw[0]
    axes, grids = _axes(h0), _grids(h0)
    n = len(axes)
    if n > 3:
        raise ValueError(f"{n} biased CVs: balanced reweighting not supported")
    B = np.asarray(boundary_factor(axes, _bounds(h0)), float)
    vol = 1.0
    for j, (c, g) in enumerate(zip(run.cvs, grids)):
        w, L = _omega_weights(c, g)
        shape = [1] * n
        shape[n - 1 - j] = len(w)
        B = B * w.reshape(shape)
        vol *= L
    out = []
    for h in raw:
        cv = np.asarray(h["cv"], float).reshape(len(h["height_kj"]), n)
        for i in range(0, len(cv), chunk):
            g = [_axis_gaussians(v, cv[i:i + chunk, k]) for k, v in enumerate(axes)]
            if n == 1:
                out.append(g[0] @ B)
            elif n == 2:
                out.append(np.einsum("ky,ky->k", g[1] @ B, g[0]))
            else:
                out.append(np.einsum("kz,ky,kx,zyx->k", g[2], g[1], g[0], B, optimize=True))
    return np.concatenate(out) / vol if out else np.zeros(0)


def _frame_wall(w, wall):
    """Wall-clock time of every frame of ``w``: walker time interpolated on its own (hill time, wall_s) pairs,
    segment by segment; linear extrapolation (segment mean rate) before the first and after the last hill."""
    if len(w.segment_of_record) != len(w.t):
        raise ValueError(f"{w.rid}: segment_of_record does not match the frames: not supported")
    ends = np.cumsum([len(np.atleast_1d(h["height_kj"])) for h in w.hills_raw])
    out = np.empty(len(w.t))
    for k, (s, e) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
        m = w.segment_of_record == k
        ht, hw, t = w.hill_t[s:e], wall[s:e], w.t[m]
        rate = (hw[-1] - hw[0]) / (ht[-1] - ht[0]) if e - s > 1 and ht[-1] > ht[0] else 0.0
        # strictly before the first hill: never at or after it
        out[m] = np.where(t < ht[0], np.minimum(hw[0] + (t - ht[0]) * rate, np.nextafter(hw[0], -np.inf)),
                          np.where(t > ht[-1], hw[-1] + (t - ht[-1]) * rate, np.interp(t, ht, hw)))
    return out


def balanced_offsets(run, group):
    """CV-space average of the bias of ``group`` (one bias), ⟨V(·,t)⟩_Ω in kJ/mol, at every frame.

    Sum over hills deposited up to the frame of height · ``hill_volumes``. One walker: hills up to the frame's
    walker time. Several walkers: hills ordered by wall-clock time; a frame's wall time is interpolated on its
    walker's (hill time, wall_s) pairs, segment by segment. Constant terms (preseed) are left out.

    Returns
    -------
    dict
        ``{rid: ndarray}`` for every walker of ``group``.
    """
    raw = _group_raw(group)
    key = ("balanced",) + _raw_key(raw)
    cache = run.__dict__.setdefault("_rw_cache", {})
    hit = cache.get(key)
    if hit is not None and all(hit[1].get(w.rid, (None,))[0] is w.t for w in group):
        return {r: v[1] for r, v in hit[1].items()}
    contrib = np.concatenate([np.asarray(h["height_kj"], float) for h in raw]) * hill_volumes(run, raw)
    out = {}
    if len(group) == 1:
        w = group[0]
        o = np.argsort(w.hill_t, kind="stable")
        cum = np.cumsum(contrib[o])
        idx = np.searchsorted(w.hill_t[o], w.t, side="right")
        out[w.rid] = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    else:
        walls = [np.concatenate([np.atleast_1d(np.asarray(h["wall_s"], float)) for h in w.hills_raw]) for w in group]
        if any(np.any(np.diff(x) < 0) for x in walls):
            raise ValueError("non-monotonic hill wall times: not supported")
        allw = np.concatenate(walls)
        o = np.argsort(allw, kind="stable")
        cum, sw = np.cumsum(contrib[o]), allw[o]
        for w, wall in zip(group, walls):
            idx = np.searchsorted(sw, _frame_wall(w, wall), side="right")
            out[w.rid] = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    cache[key] = (raw, {w.rid: (w.t, out[w.rid]) for w in group})
    return out


def _lw_ct(run, w):
    return (w.bias - w.ct) / (KT_PER_K * run.temperature)


def _lw_final(run, w):
    return _final_at_frames(run, w) / (KT_PER_K * run.temperature)


def _lw_balanced(run, w):
    return (w.bias - balanced_offsets(run, _bias_group(run, w))[w.rid]) / (KT_PER_K * run.temperature)


# per-frame log-weights over all frames of one walker (unnormalised)
RW_METHODS = {"rw_ct": _lw_ct, "rw_final": _lw_final, "rw_balanced": _lw_balanced}


def log_weights(run, w, method="rw_ct"):
    """Unnormalised per-frame log-weights of every frame of walker ``w``.

    ``rw_ct``: β(V(s,t) - c(t)) from the recorded V and c(t). ``rw_final``: βV(s, t_f), the final bias of the
    walker's bias at the logged CVs. ``rw_balanced``: β(V(s,t) - ⟨V(·,t)⟩_Ω), recorded V minus the CV-space
    average of the bias at the frame (see ``balanced_offsets``). Shared and single biases use all walkers'
    hills, isolated ones the walker's own.
    """
    if method not in RW_METHODS:
        raise ValueError(f"unknown reweighting method {method!r}: not supported")
    if w.bias is None:
        raise ValueError("no REWEIGHT data: reweighting not supported")
    if method == "rw_ct":
        return _lw_ct(run, w)
    cache = run.__dict__.setdefault("_rw_cache", {})
    deps = (w.t, w.bias) + tuple(w.cvs[c.name] for c in run.cvs)
    rk = _raw_key(_group_raw(_bias_group(run, w)))
    hit = cache.get((method, w.rid))
    if hit is None or hit[1] != rk or len(hit[0]) != len(deps) or any(a is not b for a, b in zip(hit[0], deps)):
        hit = cache[(method, w.rid)] = (deps, rk, RW_METHODS[method](run, w))
    return hit[2]


def _free_energy(p, kt):
    """-kT ln p with empty bins as NaN, shifted to min 0."""
    with np.errstate(divide="ignore"):
        F = -kt * np.log(p)
    F[~np.isfinite(F)] = np.nan
    return F - np.nanmin(F) if np.isfinite(F).any() else F


def apply(run, fn: Callable, scalar: Optional[Callable] = None, reweighted=False) -> Combined:
    """Shared/single bias: ``fn`` on all walkers pooled. Isolated: ``fn`` per walker, never pooled.

    ``reweighted`` uses only walkers with REWEIGHT data. For isolated runs with ``scalar``, ``n_walkers`` is
    the number of loaded walkers, ``n_defined`` the number with a finite scalar; mean, sd and spread are over
    the defined walkers (sd and spread NaN with fewer than two).
    """
    ws = run.reweighted() if reweighted else run.walkers
    if not ws:
        raise ValueError("no REWEIGHT data: reweighting not supported")
    if run.layout in ("shared", "single"):
        return Combined(run.layout, pooled=fn(ws))
    per = {w.rid: fn([w]) for w in ws}
    c = Combined(run.layout, per_walker=per)
    if scalar is not None:
        v = np.array([scalar(r) for r in per.values()], float)
        v = v[np.isfinite(v)]
        c.n_walkers, c.n_defined = len(run.walkers), len(v)
        if len(v):
            c.mean = float(v.mean())
            c.sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            c.spread = float(v.max() - v.min()) if len(v) > 1 else float("nan")
    return c


def _fes_bias_group(ws, hill_mask_sigmas):
    """(grids, F) of one bias from the hills of ``ws``; F in kJ/mol, min 0, NaN outside the hill mask."""
    raw = [h for w in ws for h in w.hills_raw]
    V = final_bias(raw)
    h0 = raw[0]
    grids = _grids(h0)
    F = np.where(hill_mask(raw, hill_mask_sigmas), -fes_scale(h0) * V, np.nan)
    return grids, F - np.nanmin(F)


def fes_bias(run, hill_mask_sigmas=2.0) -> Combined:
    """Bias-derived FES (kJ/mol, min 0): -γ/(γ-1)·V for well tempering, -V for transition tempering alone.

    Grid points farther than ``hill_mask_sigmas`` hill widths from every deposited hill are NaN.
    """
    return apply(run, lambda ws: _fes_bias_group(ws, hill_mask_sigmas))


def fes_marginal(F, axis, kt):
    """Boltzmann marginal ``-kt ln sum exp(-F/kt)`` over ``axis`` of ``F``, shifted to min 0 (units of ``F``).

    NaN entries of ``F`` are left out of the integral; positions with no finite entry are NaN.
    """
    F = np.asarray(F, float)
    ok = np.isfinite(F)
    a = np.where(ok, -F / kt, -np.inf)
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        g = -kt * (np.log(np.exp(a - m).sum(axis=axis)) + np.squeeze(m, axis=axis))
    g[~ok.any(axis=axis)] = np.nan
    return g - np.nanmin(g) if np.isfinite(g).any() else g


def hill_mask(raw, nsig=2.0, until=None):
    """Grid points within ``nsig`` hill widths (σ units) of a deposited hill centre.

    Parameters
    ----------
    raw : list of dict
        HILLS records of one bias (same grid).
    nsig : float
        Radius in σ units; ``inf`` marks every point sampled.
    until : dict, optional
        ``{run_id: t_ns}``: only hills deposited up to each time count.

    Returns
    -------
    ndarray of bool
        Shape of ``bias_from_hills`` output (axes in reverse variable order).
    """
    h0 = raw[0]
    gmin, gmax = np.atleast_1d(h0["grid_min"]).astype(float), np.atleast_1d(h0["grid_max"]).astype(float)
    npts = np.atleast_1d(h0["grid_points"]).astype(int)
    sig, per = np.atleast_1d(h0["sigma"]).astype(float), np.atleast_1d(h0["periodic"]).astype(bool)
    shape = tuple(reversed(npts))
    if not np.isfinite(nsig):
        return np.ones(shape, bool)
    cs = []
    for h in raw:
        c = np.asarray(h["cv"], float).reshape(len(h["height_kj"]), len(npts))
        if until is not None:
            c = c[np.asarray(h["time_ns"]) <= until.get(h["run_id"], np.inf)]
        cs.append(c)
    c = np.concatenate(cs)
    if not len(c):
        return np.zeros(shape, bool)
    grids = [np.linspace(a, b, n) for a, b, n in zip(gmin, gmax, npts)]
    pts = np.stack([g.ravel() for g in np.meshgrid(*grids, indexing="ij")], 1)
    box = np.where(per, (gmax - gmin) / sig, 0.0)

    def scale(x):
        u = (x - gmin) / sig
        return np.where(per, u % np.where(per, box, 1.0), u)
    tree = cKDTree(scale(c), boxsize=box if per.any() else None)
    d, _ = tree.query(scale(pts), distance_upper_bound=nsig)
    return np.isfinite(d).reshape(tuple(npts)).T


def bias_fes_scale(run):
    """Bias-to-FES prefactor ``-F/V`` of the run (see ``fes_scale``)."""
    return fes_scale(run.walkers[0].hills_raw[0])


def tt_basins(run):
    """Recorded transition-tempering basins as an (n_basins, n_cvs) array in CV units, or None."""
    for w in run.walkers:
        for h in w.hills_raw:
            if "tt_basins" in h:
                return np.atleast_2d(np.asarray(h["tt_basins"], float))
    return None


def _neighbourhood(p, n, s2, sigma):
    """Kernel-weighted neighbourhood count and Kish ESS per bin, for a peak-1 Gaussian kernel (σ in bins).

    ``p``, ``n``, ``s2`` are the per-bin sums of weight, count and squared weight. With ``sigma`` 0 they
    are returned as the raw count and ``p**2 / s2``.
    """
    if sigma > 0:
        d = p.ndim
        a1, a2 = (2 * np.pi * sigma ** 2) ** (d / 2), (np.pi * sigma ** 2) ** (d / 2)
        n = gaussian_filter(n, sigma, mode="nearest") * a1
        p = gaussian_filter(p, sigma, mode="nearest") * a1
        s2 = gaussian_filter(s2, sigma / np.sqrt(2), mode="nearest") * a2
    with np.errstate(divide="ignore", invalid="ignore"):
        return n, np.where(s2 > 0, p ** 2 / s2, 0.0)


def _masked_hist(sample, Wt, bins, kt, min_count, min_bin_ess, smooth_sigma=0.0, min_density=0.0):
    """Weighted histogram -> (edges, F, n_masked, kept_frac).

    The weight histogram is smoothed with a Gaussian of ``smooth_sigma`` bins (0: none) before F = -kT ln p.
    Bins are NaN where the smoothed frame count (mean frames per bin) is below ``min_density``, where the
    kernel-weighted neighbourhood count is below ``min_count`` or where the neighbourhood Kish ESS is below
    ``min_bin_ess`` (``smooth_sigma`` 0: the raw per-bin rules); F is shifted to min 0 after masking.
    ``n_masked`` counts bins with any smoothed weight that were masked; ``kept_frac`` is the unmasked
    fraction of the bins holding at least one frame.
    """
    p, edges = np.histogramdd(sample, bins=bins, weights=Wt)
    n = np.histogramdd(sample, bins=edges)[0]
    s2 = np.histogramdd(sample, bins=edges, weights=Wt ** 2)[0]
    n_nb, ess = _neighbourhood(p, n, s2, smooth_sigma)
    dens = gaussian_filter(n, smooth_sigma, mode="nearest") if smooth_sigma > 0 else n
    if smooth_sigma > 0:
        tot = p.sum()
        p = gaussian_filter(p, smooth_sigma, mode="nearest")
        p = p * (tot / p.sum()) if p.sum() > 0 else p
    bad = (n_nb < min_count) | (ess < min_bin_ess) | (dens < min_density)
    nm = int(np.sum(bad & (p > 0)))
    occ = n > 0
    kept = float(np.sum(occ & ~bad & (p > 0)) / occ.sum()) if occ.any() else float("nan")
    p = np.where(bad, 0.0, p)
    return edges, _free_energy(p, kt), nm, kept


def _edges(vals, info, bins, per_hill, fallback):
    """Bin edges of one CV and the rule used.

    Width is hill width / ``per_hill`` over [min - hill, max + hill] clipped to the CV grid. A CV without a
    hill width (monitored only) gets ``fallback`` equal bins over its range padded by one bin at each end.
    An integer ``bins`` sets the bin count over that same padded range.
    """
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if info is not None and np.isfinite(info.sigma) and info.sigma > 0:
        lo, hi = max(lo - info.sigma, info.grid_min), min(hi + info.sigma, info.grid_max)
        width, rule = info.sigma / per_hill, "hill"
    else:
        span = (hi - lo) or 1.0
        pad = span / fallback
        lo, hi = lo - pad, hi + pad
        width, rule = (hi - lo) / fallback, "fallback"
    if bins is not None:
        return np.linspace(lo, hi, int(bins) + 1), "explicit"
    n = max(1, int(np.ceil((hi - lo) / width - 1e-9)))
    return lo + width * np.arange(n + 1), rule


def fes_reweighted(run, x, y=None, bins=None, burn=0.2, min_count=0, min_bin_ess=0.0, smooth_sigma=2.0,
                   bins_per_hill=4, bins_fallback=60, min_density=0.05, method="rw_ct") -> Combined:
    """Reweighted FES (kJ/mol, min 0) on one or two logged CVs (biased or monitored).

    ``method`` is a key of ``RW_METHODS`` (frame weights, see ``log_weights``).

    Bin width of a biased CV is its hill width / ``bins_per_hill`` over the sampled range padded by one hill
    width and clipped to the CV grid. A CV without a hill width uses ``bins_fallback`` equal bins over its
    padded sampled range. ``bins`` (int, or ``{cv: int}``) overrides both.

    The weighted histogram is Gaussian-smoothed (``smooth_sigma`` bins, 0 = off) before F = -kT ln p. Bins
    are NaN where the smoothed frame count is below ``min_density``; ``min_count`` and ``min_bin_ess`` add
    neighbourhood count and ESS rules (0 = off). The shift to min 0 follows. ``Combined.ess`` holds the Kish
    ESS of the frames used, ``n_masked`` the bins masked, ``kept_frac`` the unmasked fraction of the occupied
    bins and ``bin_info`` ``{cv: {"width", "n_bins", "rule"}}`` (each a number or dict, or ``{rid: value}``
    if isolated).
    """
    if not run.has_reweight:
        raise ValueError("no REWEIGHT data: reweighted FES not supported")
    kt = KT_PER_K * run.temperature
    infos = {c.name: c for c in run.cvs}
    names = [x] if y is None else [x, y]

    def f(ws):
        cols, lws = {k: [] for k in names}, []
        for w in ws:
            lw = log_weights(run, w, method)
            m = np.arange(len(lw)) >= int(burn * len(lw))
            for k in names:
                cols[k].append(w.cvs[k][m])
            lws.append(lw[m])
        _, Wt = _pool(cols[x], lws)
        X = [np.concatenate(cols[k]) for k in names]
        ed, info = [], {}
        for k, v in zip(names, X):
            nb = bins.get(k) if isinstance(bins, dict) else bins
            e, rule = _edges(v, infos.get(k), nb, bins_per_hill, bins_fallback)
            ed.append(e)
            info[k] = dict(width=float(e[1] - e[0]), n_bins=len(e) - 1, rule=rule)
        e, F, nm, kf = _masked_hist(np.stack(X, 1), Wt, ed, kt, min_count, min_bin_ess, smooth_sigma, min_density)
        cen = tuple(0.5 * (a[1:] + a[:-1]) for a in e)
        return (cen + (F,), kish_ess(Wt), nm, kf, info)
    c = apply(run, f, reweighted=True)
    if c.pooled is not None:
        c.pooled, c.ess, c.n_masked, c.kept_frac, c.bin_info = c.pooled
    else:
        c.ess = {k: v[1] for k, v in c.per_walker.items()}
        c.n_masked = {k: v[2] for k, v in c.per_walker.items()}
        c.kept_frac = {k: v[3] for k, v in c.per_walker.items()}
        c.bin_info = {k: v[4] for k, v in c.per_walker.items()}
        c.per_walker = {k: v[0] for k, v in c.per_walker.items()}
    return c


C0_PER_A3 = 1.0 / 1661.0


def dg_from_frames(z_ang, w, bound, bulk, r_cyl_ang, temperature) -> float:
    """Standard-state ΔG° (kJ/mol) from weighted frames.

    ΔG° = -kT ln(C0 π R² ∫_bound e^{-β(W - W_bulk)} dz); NaN if the bound or bulk band has no weight.
    The bulk density uses the nominal bulk-band width.
    """
    z_ang, w = np.asarray(z_ang, float), np.asarray(w, float)
    site = (z_ang >= bound[0]) & (z_ang <= bound[1])
    blk = (z_ang >= bulk[0]) & (z_ang <= bulk[1])
    if w[site].sum() <= 0 or w[blk].sum() <= 0:
        return float("nan")
    p_bulk = w[blk].sum() / (bulk[1] - bulk[0])
    kt = KT_PER_K * temperature
    return float(-kt * np.log(C0_PER_A3 * np.pi * r_cyl_ang ** 2 * w[site].sum() / p_bulk))


def band_dF(x, w, band, temperature) -> float:
    """F(bound) - F(unbound) (kJ/mol) from weighted frames; NaN if either band is empty."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    pb = w[(x >= band.bound[0]) & (x <= band.bound[1])].sum()
    pu = w[(x >= band.unbound[0]) & (x <= band.unbound[1])].sum()
    if pb <= 0 or pu <= 0:
        return float("nan")
    return float(-KT_PER_K * temperature * np.log(pb / pu))


def _pool(vals, lws):
    """Concatenate values and raw log-weights; one global shift and one normalisation."""
    if not vals:
        return np.zeros(0), np.zeros(0)
    v, lw = np.concatenate(vals), np.concatenate(lws)
    if lw.size == 0:
        return v, lw
    w = np.exp(lw - lw.max())
    return v, w / w.sum()


def _frames(ws, run, cv, burn, t_max=None, method="rw_ct"):
    """Pooled (values, weights) after truncating at ``t_max`` and dropping the first ``burn`` fraction."""
    vs, lws = [], []
    for w in ws:
        keep = np.ones(len(w.t), bool) if t_max is None else (w.t <= t_max)
        lw = log_weights(run, w, method)[keep]
        m = np.arange(len(lw)) >= int(burn * len(lw))
        vs.append(w.cvs[cv][keep][m]); lws.append(lw[m])
    return _pool(vs, lws)


def _window_raw(ws, run, cv, lo, hi, method="rw_ct"):
    """Per-walker values and raw log-weights of frames with lo <= t < hi."""
    vs, lws = [], []
    for w in ws:
        k = (w.t >= lo) & (w.t < hi)
        vs.append(w.cvs[cv][k]); lws.append(log_weights(run, w, method)[k])
    return vs, lws


def _frames_in_window(ws, run, cv, lo, hi, method="rw_ct"):
    """Pooled (values, weights) of frames with lo <= t < hi, no burn-in."""
    return _pool(*_window_raw(ws, run, cv, lo, hi, method))


def _frames_in_windows(ws, run, cv, windows, cache=None, method="rw_ct"):
    """Pooled (values, weights) of the frames of every (lo, hi) window, repeated windows counted repeatedly.

    ``cache`` is an optional dict reused across calls to avoid recomputing per-window raw weights.
    """
    vs, lws = [], []
    for lo, hi in windows:
        if cache is None:
            raw = _window_raw(ws, run, cv, lo, hi, method)
        else:
            raw = cache.get((lo, hi))
            if raw is None:
                raw = cache[(lo, hi)] = _window_raw(ws, run, cv, lo, hi, method)
        vs += raw[0]; lws += raw[1]
    return _pool(vs, lws)


def block_band_stats(ws, run, bands_, cv, lo, hi, method="rw_ct") -> dict:
    """Site and bulk Kish ESS and round trips of the frames with lo <= t < hi.

    Weights are pooled over ``ws`` with one normalisation; round trips are summed over walkers.
    """
    b = bands_[cv]
    vs, lws = _window_raw(ws, run, cv, lo, hi, method)
    v, w = _pool(vs, lws)
    ins = (v >= b.bound[0]) & (v <= b.bound[1])
    inu = (v >= b.unbound[0]) & (v <= b.unbound[1])
    n = sum(recrossings(SimpleNamespace(cvs={cv: x}), cv, b)["n_cycles"] for x in vs if len(x))
    return dict(ess_site=kish_ess(w[ins]), ess_bulk=kish_ess(w[inu]), recross=int(n))


def _value(run, bands_, cv, values, weights):
    """ΔG° or band ΔF of weighted frames, with the Kish ESS and frame count."""
    b = bands_[cv]
    a = ang_per_unit(run, cv)
    n = dict(ess=kish_ess(weights), n_frames=int(len(values)))
    if cv == FUNNEL_Z and run.funnel is not None:
        if "R_cylinder" not in run.funnel:
            raise ValueError("funnel without R_cylinder: ΔG° not supported")
        v = dg_from_frames(values * a, weights, (b.bound[0] * a, b.bound[1] * a),
                           (b.unbound[0] * a, b.unbound[1] * a), _ang(run.funnel, "R_cylinder"), run.temperature)
        return dict(value=v, unit="kJ/mol", kind="dG0",
                    reason=None if np.isfinite(v) else "no weight in bound or bulk band", **n)
    v = band_dF(values, weights, b, run.temperature)
    return dict(value=v, unit="kJ/mol", kind="band_dF",
                reason=None if np.isfinite(v) else "no weight in bound or unbound band", **n)


def binding(run, bands_, cv=FUNNEL_Z, burn=0.2, method="rw_ct") -> Combined:
    """ΔG° from the funnel (kind "dG0") or the bound-unbound band difference (kind "band_dF").

    Each value dict carries ``ess`` (Kish) and ``n_frames`` of the frames used. ``method`` is a key of
    ``RW_METHODS``.
    """
    if not run.has_reweight:
        raise ValueError("no REWEIGHT data: ΔG° not supported")
    return apply(run, lambda ws: _value(run, bands_, cv, *_frames(ws, run, cv, burn, method=method)),
                 scalar=lambda r: r["value"], reweighted=True)


def aggregate_time(ws, t):
    """Summed walker time sampled when each walker has reached ``t``: sum over walkers of min(t, T_w)."""
    t = np.asarray(t, float)
    return sum(np.minimum(t, float(w.t[-1])) for w in ws)


def dg_series(run, bands_, cv=FUNNEL_Z, burn=0.2, step_ns=1.0, method="rw_ct") -> Combined:
    """ΔG°(t): each point from every walker's frames with walker time <= t, first ``burn`` of each walker's
    truncated span dropped; ``ess`` and ``ess_frac`` (ESS over frame count) per point. ``method`` is a key of
    ``RW_METHODS``.

    The grid ends exactly at T, the longest pooled walker's end time; shorter walkers contribute all their
    frames, so the last point equals ``binding``. Shared and single runs also return ``t_aggregate``, the
    walker time summed over walkers at each point.
    """
    if not run.has_reweight:
        raise ValueError("no REWEIGHT data: ΔG°(t) not supported")

    def f(ws):
        T = max(w.t[-1] for w in ws)
        t = np.arange(step_ns, T + 1e-12, step_ns)
        if len(t) == 0 or abs(t[-1] - T) > 1e-9:
            t = np.append(t, T)
        v = [_value(run, bands_, cv, *_frames(ws, run, cv, burn, t_max=ti, method=method)) for ti in t]
        ess = np.array([x["ess"] for x in v])
        nf = np.array([x["n_frames"] for x in v], float)
        out = dict(t=t, dg=np.array([x["value"] for x in v]), ess=ess,
                   ess_frac=np.divide(ess, nf, out=np.full(len(t), np.nan), where=nf > 0), unit="kJ/mol")
        if run.layout != "isolated":
            out["t_aggregate"] = aggregate_time(ws, t)
        return out
    return apply(run, f, reweighted=True)


def _bias_curve(run, cv, grids, F):
    """W along ``cv`` (kJ/mol, min 0) from a bias FES: every other CV is Boltzmann-integrated out.

    The grid is stored (last CV, ..., first CV), so the axis of CV ``j`` is ``ncv - 1 - j``.
    """
    names = [c.name for c in run.cvs]
    if cv not in names:
        raise ValueError(f"{cv} is not a biased CV: bias-derived W not supported")
    n, j = len(names), names.index(cv)
    F = np.asarray(F, float)
    if F.shape != tuple(len(g) for g in reversed(grids)):
        raise ValueError(f"bias grid shape {F.shape} does not match the CV grids: not supported")
    kt = KT_PER_K * run.temperature
    for ax in range(n - 1, -1, -1):
        if ax != n - 1 - j:
            F = fes_marginal(F, ax, kt)
    return np.asarray(grids[j], float), F


def _bias_value(run, bands_, cv, z, W):
    """ΔG° (funnel) or band ΔF (kJ/mol) from W(z) on a grid; same bands and formula as ``_value``.

    The bulk density is the mean of exp(-W/kT) over the bulk band; grid sums stand for the integrals.
    """
    b, a, kt = bands_[cv], ang_per_unit(run, cv), KT_PER_K * run.temperature
    z, W = np.asarray(z, float), np.asarray(W, float)
    fin = np.isfinite(W)
    site = fin & (z >= b.bound[0]) & (z <= b.bound[1])
    oth = fin & (z >= b.unbound[0]) & (z <= b.unbound[1])
    funnel = cv == FUNNEL_Z and run.funnel is not None
    out = dict(value=float("nan"), unit="kJ/mol", kind="dG0" if funnel else "band_dF", z=z, W=W)
    if funnel and "R_cylinder" not in run.funnel:
        raise ValueError("funnel without R_cylinder: ΔG° not supported")
    if not site.any() or not oth.any():
        out["reason"] = "no bias-derived W in bound or " + ("bulk" if funnel else "unbound") + " band"
        return out
    boltz = np.exp(-(W - W[fin].min()) / kt)
    if funnel:
        dz = float(np.median(np.diff(z))) * a
        v = -kt * np.log(C0_PER_A3 * np.pi * _ang(run.funnel, "R_cylinder") ** 2 * boltz[site].sum() * dz
                         / boltz[oth].mean())
    else:
        v = -kt * np.log(boltz[site].sum() / boltz[oth].sum())
    out.update(value=float(v), reason=None)
    return out


def binding_bias(run, bands_, cv=FUNNEL_Z, hill_mask_sigmas=2.0) -> Combined:
    """ΔG° (kind "dG0") or band ΔF (kind "band_dF") from the final bias: F = -fes_scale·V.

    Other CVs are Boltzmann-integrated out. Shared and single runs use one bias (all hills pooled); isolated
    runs one bias per walker, never pooled. Each value dict carries the curve ``z`` (CV units) and ``W``
    (kJ/mol, min 0, NaN outside the hill mask).
    """
    def f(ws):
        grids, F = _fes_bias_group(ws, hill_mask_sigmas)
        return _bias_value(run, bands_, cv, *_bias_curve(run, cv, grids, F))
    return apply(run, f, scalar=lambda r: r["value"])


def dg_bias_series(run, bands_, cv=FUNNEL_Z, n=20, hill_mask_sigmas=2.0) -> Combined:
    """ΔG°(t) from the bias at ``n`` cuts through the deposition, per bias. The last point equals ``binding_bias``.

    Isolated: equally spaced wall-clock cuts, ``t`` the walker's own time. Shared and single: cuts after
    ``k N / n`` of the ``N`` hills in deposition order; ``t`` (also ``t_aggregate``) is the walker time reached
    at the cut, summed over walkers.
    """
    shared = run.layout != "isolated"

    def f(ws):
        raw = [r for w in ws for r in w.hills_raw]
        wall_all = np.sort(np.concatenate([r["wall_s"] for r in raw]))
        cuts = None
        if shared:
            idx = np.ceil(np.arange(1, n + 1) * len(wall_all) / n).astype(int) - 1
            cuts = wall_all[idx]
        hist = fes_history(raw, n=n, cuts=cuts)
        if len(hist) != 1:
            raise ValueError("bias snapshots of one bias expected: not supported")
        r0 = raw[0]
        grids = [np.linspace(float(a), float(b), int(m)) for a, b, m in
                 zip(np.atleast_1d(r0["grid_min"]), np.atleast_1d(r0["grid_max"]), np.atleast_1d(r0["grid_points"]))]
        wall = [(w, np.concatenate([r["wall_s"] for r in w.hills_raw])) for w in ws]
        agg = sum if shared else max
        dg, t = [], []
        for c, F in zip(hist[0]["wall_s"], hist[0]["fes"]):
            until = {r["run_id"]: (float(np.max(r["time_ns"][r["wall_s"] <= c])) if np.any(r["wall_s"] <= c)
                                   else -np.inf) for r in raw}
            Fm = np.where(hill_mask(raw, hill_mask_sigmas, until=until), F, np.nan)
            ok = np.isfinite(Fm).any()
            dg.append(_bias_value(run, bands_, cv, *_bias_curve(run, cv, grids, Fm - np.nanmin(Fm)))["value"]
                      if ok else float("nan"))
            t.append(agg((float(w.hill_t[ww <= c].max()) if np.any(ww <= c) else 0.0) for w, ww in wall))
        out = dict(t=np.array(t), dg=np.array(dg), unit="kJ/mol")
        if shared:
            out["t_aggregate"] = out["t"]
        return out
    return apply(run, f)


def block_analysis(t, dg, block_value, t_rej=0.2, n_blocks=10, n_boot=2000, seed=0, windows_value=None,
                   block_ess=None, block_stats=None) -> dict:
    """Convergence numbers after the rejection time ``t_rej`` (fraction of the run).

    Blocks are contiguous half-open windows [lo, hi) of walker time between ``t_rej * t[-1]`` and ``t[-1]``;
    the last window is closed at ``t[-1]``. The bootstrap resamples whole blocks with replacement and recomputes
    ΔG° from the pooled frames of the drawn blocks.

    Parameters
    ----------
    t, dg : array_like
        ΔG°(t) series.
    block_value : callable
        ``block_value(lo, hi)`` returns ΔG° from frames with lo <= t < hi (NaN if undefined).
    windows_value : callable
        ``windows_value([(lo, hi), ...])`` returns ΔG° from the frames of all windows, concatenated with
        multiplicity (NaN if undefined). Required.
    block_ess : callable, optional
        ``block_ess(lo, hi)`` returns the Kish ESS of the block's frames; adds ``block_ess``.
    block_stats : callable, optional
        ``block_stats(lo, hi)`` returns a dict with ``ess_site``, ``ess_bulk``, ``recross``; adds
        ``block_ess_site``, ``block_ess_bulk``, ``block_recross``.

    Returns
    -------
    dict
        time_mean, time_sd, n_finite (finite ΔG°(t) points used), block_dg, n_valid_blocks, nan_blocks
        (indices), boot_sd (SD over finite replicates; NaN if fewer than 2), n_boot_nan, last_blocks_diff, unit.
    """
    if windows_value is None:
        raise ValueError("frame-block bootstrap needs windows_value: not supported")
    t, dg = np.asarray(t, float), np.asarray(dg, float)
    t0 = t_rej * t[-1]
    after = t >= t0
    edges = np.linspace(t0, t[-1], n_blocks + 1)
    his = edges[1:].copy()
    his[-1] = np.nextafter(his[-1], np.inf)
    wins = list(zip(edges[:-1], his))
    blocks = np.array([block_value(lo, hi) for lo, hi in wins], float)
    valid = np.isfinite(blocks)
    rng = np.random.default_rng(seed)
    boots = np.array([windows_value([wins[i] for i in rng.integers(0, n_blocks, n_blocks)])
                      for _ in range(n_boot)], float)
    fin = np.isfinite(boots)
    boot_sd = float(np.std(boots[fin])) if fin.sum() >= 2 else float("nan")
    ok = np.isfinite(dg[after])
    out = dict(time_mean=float(np.mean(dg[after][ok])) if ok.any() else float("nan"),
                time_sd=float(np.std(dg[after][ok])) if ok.any() else float("nan"),
                n_finite=int(ok.sum()), block_dg=[float(b) for b in blocks], n_valid_blocks=int(valid.sum()),
                nan_blocks=[int(k) for k in np.flatnonzero(~valid)], boot_sd=boot_sd,
                n_boot_nan=int((~fin).sum()),
                last_blocks_diff=float(abs(blocks[-1] - blocks[-2])) if valid[-2:].all() else float("nan"),
                unit="kJ/mol")
    if block_ess is not None:
        out["block_ess"] = [float(block_ess(lo, hi)) for lo, hi in wins]
    if block_stats is not None:
        st = [block_stats(lo, hi) for lo, hi in wins]
        out["block_ess_site"] = [float(x["ess_site"]) for x in st]
        out["block_ess_bulk"] = [float(x["ess_bulk"]) for x in st]
        out["block_recross"] = [int(x["recross"]) for x in st]
    return out


def estimator_gap(rw, bias) -> dict:
    """Reweighted minus bias-derived binding value (kJ/mol): ``pooled`` (or None) and ``per_walker``."""
    def d(a, b):
        return float(a["value"] - b["value"])
    if rw.pooled is not None:
        return dict(pooled=d(rw.pooled, bias.pooled), per_walker={}, unit="kJ/mol")
    return dict(pooled=None, unit="kJ/mol",
                per_walker={k: d(v, bias.per_walker[k]) for k, v in rw.per_walker.items() if k in bias.per_walker})


def _values(c, keys):
    """Binding value (kJ/mol, NaN if undefined) of a Combined (or None) for each key ("pooled" or a walker id)."""
    out = {}
    for k in keys:
        v = None if c is None else (c.pooled if k == "pooled" else c.per_walker.get(k))
        out[k] = float(v["value"]) if v is not None and v.get("value") is not None else float("nan")
    return out


def estimator_spread(estimators, keys, methods=ESTIMATORS) -> dict:
    """Max - min binding value (kJ/mol) over ``methods``, per key ("pooled" or walker ids).

    ``estimators`` maps a method to ``{"binding": Combined or None, ...}``. A key is NaN when any method is
    missing or undefined there.

    Returns
    -------
    dict
        ``pooled`` (float or None), ``per_walker`` ({rid: float}), ``methods``, ``unit``.
    """
    vals = {m: _values((estimators.get(m) or {}).get("binding"), keys) for m in methods}
    sp = {}
    for k in keys:
        v = np.array([vals[m][k] for m in methods], float)
        sp[k] = float(v.max() - v.min()) if len(v) and np.all(np.isfinite(v)) else float("nan")
    return dict(pooled=sp.get("pooled"), per_walker={k: x for k, x in sp.items() if k != "pooled"},
                methods=list(methods), unit="kJ/mol")


def estimator_gaps(estimators, keys, methods=ESTIMATORS) -> dict:
    """Pairwise binding-value differences ``"a-b"`` (kJ/mol), each ``{"pooled", "per_walker"}`` like
    ``estimator_spread``."""
    vals = {m: _values((estimators.get(m) or {}).get("binding"), keys) for m in methods}
    out = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            d = {k: vals[a][k] - vals[b][k] for k in keys}
            out[f"{a}-{b}"] = dict(pooled=d.get("pooled"), per_walker={k: x for k, x in d.items() if k != "pooled"})
    return out


def _blocks(run, bands_, main, ser, method, t_rej, n_blocks, n_boot, seed, min_points, skipped):
    """Block analysis of one reweighting method: dict (pooled) or ``{rid: dict or None}`` (isolated)."""
    iso = run.layout == "isolated"
    groups = [(w.rid, [w]) for w in run.reweighted()] if iso else [("all", run.reweighted())]
    blocks = {}
    for lab, ws in groups:
        s = ser.per_walker[lab] if iso else ser.pooled
        if len(s["t"]) < min_points:
            skipped.append(f"{lab}: blocks ({LABELS[method]}): ΔG°(t) has {len(s['t'])} < {min_points} points"
                           if method != "rw_ct" else f"{lab}: blocks: ΔG°(t) has {len(s['t'])} < {min_points} points")
            blocks[lab] = None
            continue
        cache = {}

        def bv(lo, hi, ws=ws):
            return _value(run, bands_, main, *_frames_in_window(ws, run, main, lo, hi, method))["value"]

        def wv(wins, ws=ws, cache=cache):
            return _value(run, bands_, main, *_frames_in_windows(ws, run, main, wins, cache, method))["value"]

        def be(lo, hi, ws=ws):
            return kish_ess(_frames_in_window(ws, run, main, lo, hi, method)[1])

        def bs(lo, hi, ws=ws):
            return block_band_stats(ws, run, bands_, main, lo, hi, method)
        blocks[lab] = block_analysis(s["t"], s["dg"], bv, t_rej=t_rej, n_blocks=n_blocks, n_boot=n_boot,
                                     seed=seed, windows_value=wv, block_ess=be, block_stats=bs)
    return blocks if iso else blocks["all"]


def fe_numbers(run, bands_, cv=None, burn=0.2, t_rej=0.2, step_ns=1.0, min_points=12, n_blocks=10, n_boot=2000,
               seed=0, hill_mask_sigmas=2.0, bias_n=20, primary=PRIMARY, methods=ESTIMATORS) -> dict:
    """Binding free energy, ΔG°(t) and block analysis along one CV (per walker if isolated), per estimator.

    Parameters
    ----------
    cv : str, optional
        Default funnel z when it has a band, else the first biased CV.
    min_points : int
        Fewest ΔG°(t) points for a block analysis; shorter series give ``None`` blocks and a skip note.
    hill_mask_sigmas, bias_n
        Hill mask radius and number of snapshots of the bias-derived estimator.
    primary : str
        Reweighting estimator that drives the block colouring and the convergence checks.
    methods : sequence of str
        Estimators to compute (keys of ``ESTIMATORS``); ``rw_ct`` and ``bias`` are always computed.

    Returns
    -------
    dict
        ``cv``; ``primary``; ``estimators`` (``{method: {"binding", "series", "blocks"}}``; Combined values,
        ``blocks`` None for ``bias``; a method that is not supported is absent and noted in ``skipped``);
        ``estimator_spread`` and ``estimator_gaps`` (see ``estimator_spread``, ``estimator_gaps``);
        ``binding``, ``series``, ``blocks`` (``rw_ct``; None without REWEIGHT data), ``binding_bias``,
        ``dg_bias_series`` (``bias``; None if not supported), ``estimator_gap`` (``rw_ct`` - ``bias``, see
        ``estimator_gap``); ``skipped`` (list of str). Free energies in kJ/mol.
    """
    if primary not in RW_METHODS:
        raise ValueError(f"primary estimator {primary!r} is not a reweighting method: not supported")
    unknown = set(methods) - set(ESTIMATORS)
    if unknown:
        raise ValueError(f"unknown estimators {sorted(unknown)}: not supported")
    methods = [m for m in ESTIMATORS if m in set(methods) | {"rw_ct", "bias", primary}]
    main = cv or (FUNNEL_Z if FUNNEL_Z in bands_ else run.cvs[0].name)
    est = {}
    out = dict(cv=main, primary=primary, estimators=est, estimator_spread=None, estimator_gaps=None, binding=None,
               series=None, blocks=None, binding_bias=None, dg_bias_series=None, estimator_gap=None, skipped=[])
    try:
        bb = binding_bias(run, bands_, cv=main, hill_mask_sigmas=hill_mask_sigmas)
        bs = dg_bias_series(run, bands_, cv=main, n=bias_n, hill_mask_sigmas=hill_mask_sigmas)
        est["bias"] = dict(binding=bb, series=bs, blocks=None)
        out["binding_bias"], out["dg_bias_series"] = bb, bs
    except ValueError as e:
        out["skipped"].append(f"bias-derived ΔG°: {e}")
    keys = [w.rid for w in run.walkers] if run.layout == "isolated" else ["pooled"]
    if not run.has_reweight:
        out["skipped"].append("reweighted FES, ΔG° and blocks: no REWEIGHT data")
    else:
        for m in [m for m in methods if m in RW_METHODS]:
            try:
                b = binding(run, bands_, cv=main, burn=burn, method=m)
                ser = dg_series(run, bands_, cv=main, burn=burn, step_ns=step_ns, method=m)
            except ValueError as e:
                if m == "rw_ct":
                    raise
                out["skipped"].append(f"{LABELS[m]} ΔG°: {e}")
                continue
            blk = _blocks(run, bands_, main, ser, m, t_rej, n_blocks, n_boot, seed, min_points, out["skipped"])
            est[m] = dict(binding=b, series=ser, blocks=blk)
        if primary not in est:
            raise ValueError(f"primary estimator {primary} could not be computed: not supported")
        r = est["rw_ct"]
        out["binding"], out["series"], out["blocks"] = r["binding"], r["series"], r["blocks"]
        if out["binding_bias"] is not None:
            out["estimator_gap"] = estimator_gap(out["binding"], out["binding_bias"])
    order = [m for m in ESTIMATORS if m in methods]
    out["estimator_spread"] = estimator_spread(est, keys, order)
    out["estimator_gaps"] = estimator_gaps(est, keys, order)
    return out
