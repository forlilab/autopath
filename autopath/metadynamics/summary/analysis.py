"""Bands, recrossings, reweighting, the shared/isolated combination rule, free energies and ΔG°."""
from dataclasses import dataclass

import numpy as np
from openmm import unit

FUNNEL_Z = "funnel_z"
SITE_FRAC = 0.2
BULK_MARGIN_ANG = 1.5
EDGE_FRAC = 0.1


@dataclass
class Band:
    bound: tuple
    unbound: tuple
    bound_end: str
    source: str


def _ang(fp, key):
    return float(fp[key].value_in_unit(unit.angstrom))


def _default_band(run, name):
    spec = next((c for c in run.cvs if c.name == name), None)
    if name == FUNNEL_Z and run.funnel is not None:
        a = 10.0 if spec is None or spec.ang_per_unit is None else spec.ang_per_unit
        if "z_low" in run.funnel:
            z_low = _ang(run.funnel, "z_low")
        elif spec is not None:
            z_low = spec.grid_min * a
        else:  # monitor-only funnel z: no grid, use the observed minimum
            z_low = float(min(np.min(w.cvs[name]) for w in run.walkers)) * a
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
        lo, hi = float(np.min(allv)), float(np.max(allv))
    else:
        lo, hi = spec.grid_min, spec.grid_max
    e = EDGE_FRAC * (hi - lo)
    return Band((lo, lo + e), (hi - e, hi), "low", "grid")


def bands(run, overrides=None) -> dict:
    """Bound / unbound interval per CV (biased and monitored).

    Defaults: funnel z from the funnel geometry, a learned CV from its B/U basins ± its hill width, otherwise the
    lowest/highest ``EDGE_FRAC`` of the grid with the bound end low. ``overrides`` maps a CV name to any of
    ``bound``, ``unbound``, ``bound_end``.
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
    state, n, seen_u = 0, 0, False
    for b, u in zip(inb, inu):
        if b:
            if state == 2 and seen_u:
                n += 1
            state, seen_u = 1, False
        elif u and state == 1:
            state, seen_u = 2, True
    return dict(n_cycles=int(n), frac_bound=float(inb.mean()), frac_unbound=float(inu.mean()))
