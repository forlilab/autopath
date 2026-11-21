"""End-of-run metadynamics numbers: ΔG°_b, pKd and the convergence verdict (no figures).

Run modes:

* single walker    its own bias and FES history;
* shared bias      walkers sharing one bias directory form one surface; history on
                   summed walker time at wall-clock cuts (concurrent or sequential);
* isolated biases  each walker is an independent estimate, reported alone and as
                   mean ± sd (biases are never averaged).

Surfaces and histories are rebuilt from the HILLS v2 files (Hills.py); every ΔG
comes from correct_fe_for_funnel.
"""
import glob
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from autopath.metadynamics.Analysis import MetadynamicsAnalysis, bound_site_bins, correct_fe_for_funnel
from autopath.metadynamics.Funnel import load_funnel_params, funnel_basins
from autopath.metadynamics.Hills import bias_from_hills, fes_history, load_hills

logger = logging.getLogger("autopath")

KJ_TO_KCAL = 1.0 / 4.184
N_SNAPSHOTS = 40
Z_CLAMP_TOL = 1e-5   # nm; GPU platforms clamp funnel_z in float32


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------
@dataclass
class RunMode:
    layout: str                      # "single" | "shared" | "isolated"
    schedule: Optional[str]          # "concurrent" | "sequential" (shared only)
    run_ids: list

    def describe(self) -> str:
        n = len(self.run_ids)
        if self.layout == "single":
            return "single walker"
        if self.layout == "isolated":
            return f"{n} walkers, isolated biases"
        return f"{n} walkers, shared bias ({self.schedule})"


def _window(w) -> tuple:
    """(start, end) wall-clock seconds of one walker: its start to its last hill."""
    start = float(w["started_at"])
    return start, float(w["wall_s"][-1]) if len(w["wall_s"]) else start


def detect_mode(walkers) -> RunMode:
    """Walkers sharing one bias directory = shared bias, else isolated."""
    run_ids = sorted(w["run_id"] for w in walkers)
    if len(run_ids) <= 1:
        return RunMode("single", None, run_ids)
    if len({w["bias_dir"] for w in walkers}) > 1:
        return RunMode("isolated", None, run_ids)
    windows = sorted(_window(w) for w in walkers)
    concurrent = any(nxt[0] < cur[1] for cur, nxt in zip(windows, windows[1:]))
    return RunMode("shared", "concurrent" if concurrent else "sequential", run_ids)


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
@dataclass
class Estimate:
    label: str
    fes_kj: np.ndarray
    corr: Optional[dict]
    error: Optional[str]
    t_ns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dG_kj: np.ndarray = field(default_factory=lambda: np.zeros(0))
    resolved: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    snapshots: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    complete: bool = True

    @property
    def dG(self) -> Optional[float]:
        return None if self.corr is None else float(self.corr["dG_bind_std_kj_mol"])


def _well_depth(fes_kj, grid=(0.0, 1.0), origin=None, n_tail=10):
    """FES minimum in the bound site relative to the unbound end of the grid (kJ/mol)."""
    fe = np.asarray(fes_kj, float)
    return float((fe - fe[-n_tail:].mean())[:bound_site_bins(*grid, fe.size, origin)].min())


def reference_range_from_funnel(fp, grid, cv_is_funnel_z, sigma_ang, out_dir=None,
                                cv_len=None, n_sigma=3.0, hill_width_cv=None):
    """Unbound reference window in CV units: the cylinder, clear of the wall.

    z in [z_cc, z_max - n_sigma·σ], avoiding the hill-deficit artifact near the wall.
    Non-z CVs are mapped linearly from the bound site (grid_min) to the cap (grid_max). The window also stays
    n_sigma hill widths inside the grid edges. Returns None if no window fits.
    """
    z_cc = float(fp["z_cc"].value_in_unit(unit_ang()))
    z_max = float(fp["z_max"].value_in_unit(unit_ang()))
    z_lo, z_hi = z_cc, z_max - n_sigma * float(sigma_ang)
    if z_hi <= z_lo:
        logger.warning("Cylinder (%.1f-%.1f A) is shorter than %g sigma: no reference window.",
                       z_cc, z_max, n_sigma)
        return None
    if cv_is_funnel_z:
        lo, hi = z_lo / 10.0, z_hi / 10.0
    else:
        z0 = 2.0 * funnel_basins(fp)[0] - z_cc               # = the sMD bound-site z
        span = (grid[1] - grid[0]) / max(z_max - z0, 1e-9)
        lo, hi = grid[0] + (z_lo - z0) * span, grid[0] + (z_hi - z0) * span
        logger.info("Reference window mapped linearly bound site -> cap: approximate.")
    margin = n_sigma * float(hill_width_cv) if hill_width_cv else 0.0
    lo, hi = max(lo, grid[0] + margin), min(hi, grid[1] - margin)
    return (lo, hi) if hi > lo else None


def unit_ang():
    from openmm import unit
    return unit.angstrom


def _correct(fes_kj, fp, temperature, grid, cv_len, reference_range=None, origin=None):
    if fp is None:
        return None, "no funnel: standard-state correction needs the funnel cylinder"
    try:
        return correct_fe_for_funnel(fes_kj, fp, temperature=temperature,
                                     grid_min=grid[0], grid_max=grid[1],
                                     cv_length_per_unit_ang=cv_len, verbose=False,
                                     reference_range=reference_range, cv_origin=origin), None
    except Exception as exc:                       # surface not interpretable yet
        return None, str(exc)


def _history(snapshots, fp, temperature, grid, cv_len, reference_range=None, origin=None):
    dG = np.full(len(snapshots), np.nan)
    ok = np.zeros(len(snapshots), bool)
    if fp is None:                             # no correction possible: track the well depth
        for i, fe in enumerate(snapshots):
            dG[i] = _well_depth(fe, grid, origin)
            ok[i] = dG[i] <= -10.0
        return dG, ok
    for i, fe in enumerate(snapshots):
        try:
            r = correct_fe_for_funnel(fe, fp, temperature=temperature, grid_min=grid[0],
                                      grid_max=grid[1], cv_length_per_unit_ang=cv_len,
                                      strict=False, verbose=False,
                                      reference_range=reference_range, cv_origin=origin)
            dG[i], ok[i] = r["dG_bind_std_kj_mol"], r["bound_state_sampled"]
        except Exception:
            pass
    return dG, ok


def build_estimates(walkers, fp, temperature, grid, cv_len, reference_range=None,
                    origin=None, n_snapshots=N_SNAPSHOTS) -> list:
    """One estimate per bias: shared/single = one surface, isolated = one per walker."""
    estimates = []
    for h in fes_history(walkers, n=n_snapshots):
        fes = h["fes"][-1]
        corr, err = _correct(fes, fp, temperature, grid, cv_len, reference_range, origin)
        dG, ok = _history(h["fes"], fp, temperature, grid, cv_len, reference_range, origin)
        label = h["run_ids"][0] if len(h["run_ids"]) == 1 else "shared bias"
        done = all(bool(w.get("finished", False)) for w in walkers if w["run_id"] in h["run_ids"])
        estimates.append(Estimate(label, fes, corr, err, h["t_ns"], dG, ok, h["fes"], complete=done))
    return estimates


# ---------------------------------------------------------------------------
# Walker traces
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    run_id: str
    t_ns: np.ndarray
    cv: np.ndarray
    z_ang: Optional[np.ndarray]
    stats: dict


def walker_traces(out_dir, walkers, basins, grid, cv_is_funnel_z=False) -> list:
    ma = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    traces = []
    for w in sorted(walkers, key=lambda w: w["run_id"]):
        r = w["run_id"]
        cpath = os.path.join(out_dir, f"COLVAR_{r}.npy")
        if not os.path.exists(cpath):
            continue
        col = np.load(cpath)
        cv = col[:, 0] if col.ndim > 1 else col
        stats = ma.walker_transitions(cpath, basins, grid_min=grid[0], grid_max=grid[1],
                                      cv_is_funnel_z=cv_is_funnel_z)
        z = cv * 10.0 if stats["coordinate"] == "funnel_z" else None
        t_ns = np.arange(len(cv)) * float(w["record_dt_ns"])
        traces.append(Trace(r, t_ns, cv, z, stats))
    return traces


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _final_bias_at_frames(out_dir, run_id, s):
    """V_final (kJ/mol) of the bias this walker deposited into, at its biased-CV values s (n, ncv)."""
    ws = load_hills(out_dir)
    me = next(w for w in ws if w["run_id"] == run_id)
    group = [w for w in ws if w["bias_dir"] == me["bias_dir"]] if me["bias_dir"] else [me]
    V = bias_from_hills(group)
    axes = [np.linspace(a, b, int(n)) for a, b, n in zip(me["grid_min"], me["grid_max"], me["grid_points"])]
    f = RegularGridInterpolator(tuple(reversed(axes)), V, bounds_error=False, fill_value=None)
    return f(np.asarray(s)[:, ::-1])


def _colvar_column(out_dir, rid, name):
    """Index of CV ``name`` among the walker's COLVAR columns (HILLS cv_names), or None."""
    f = os.path.join(out_dir, f"HILLS_{rid}.npz")
    if not os.path.exists(f):
        return None
    with np.load(f) as d:
        names = [str(n) for n in d["cv_names"]] if "cv_names" in d.files else []
    return names.index(name) if name in names else None


def z_reweighted_dG(out_dir, fp, temperature, burn=0.2, bins=0.01):
    """ΔG°_b from W(z) of the funnel z, reweighted by eq 12 (final bias) and by c(t).

    Bound site: z <= 0.2*z_max of the funnel. z is the unbiased funnel_z monitor or, without
    one, the biased funnel_z COLVAR column; its clamped frames (z >= z_max) are dropped.
    The histogram spans [floor(min z), funnel z_max] (nm); empty bins are +inf (zero
    probability) and frames beyond z_max are dropped (``n_dropped``).
    Returns {"reason": ...} without numbers when it cannot run.
    """
    if fp is None:
        return {"reason": "no funnel: z reweighting needs the funnel cylinder"}
    kT = 0.0083144626 * temperature
    zs, l12, lct, from_colvar = [], [], [], []
    for f in sorted(glob.glob(os.path.join(out_dir, "REWEIGHT_*.npz"))):
        rw = np.load(f)
        rid = os.path.basename(f)[len("REWEIGHT_"):-4]
        zcol = None if "monitor_funnel_z" in rw.files else _colvar_column(out_dir, rid, "funnel_z")
        if "monitor_funnel_z" not in rw.files and zcol is None:
            continue
        n = len(rw["monitor_funnel_z"]) if zcol is None else len(rw["bias_kj"])
        if len(rw["bias_kj"]) < n or len(rw["rct_kj"]) < n:
            raise ValueError(f"{rid}: bias_kj/rct_kj shorter than the {n} monitored rows")
        keep = np.arange(n) >= burn * n
        s = np.load(os.path.join(out_dir, f"COLVAR_{rid}.npy"))
        if len(s) < n:
            raise ValueError(f"{rid}: COLVAR has {len(s)} rows, REWEIGHT {n}")
        s = s[:n].reshape(n, -1)
        z = rw["monitor_funnel_z"] if zcol is None else s[:, zcol]
        zs.append(z[keep])
        from_colvar.append(np.full(int(keep.sum()), zcol is not None))
        l12.append(_final_bias_at_frames(out_dir, rid, s[keep]) / kT)
        lct.append((rw["bias_kj"][:n] - rw["rct_kj"][:n])[keep] / kT)
    if not zs:
        return {"reason": "no REWEIGHT with monitor_funnel_z or a biased funnel_z COLVAR column in this run"}
    z = np.concatenate(zs)
    zmax = fp["z_max"]
    zmax = float(zmax.value_in_unit(unit_ang())) / 10.0 if hasattr(zmax, "value_in_unit") else float(zmax)
    inside = np.where(np.concatenate(from_colvar), z < zmax - Z_CLAMP_TOL, z <= zmax)
    n_dropped = int((~inside).sum())
    if not inside.any():
        return {"reason": "no monitored frames inside the funnel (z <= z_max)"}
    z = z[inside]
    lo = float(np.floor(z.min() / bins) * bins)
    edges = np.arange(lo, zmax + bins, bins)
    edges = edges[edges < zmax + 1e-12]
    edges = np.append(edges, zmax) if edges[-1] < zmax - 1e-12 else edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    out = {"n_frames": int(len(z)), "n_dropped": n_dropped, "reason": ""}
    for key, lw in (("eq12_kj", np.concatenate(l12)[inside]), ("ct_kj", np.concatenate(lct)[inside])):
        w = np.exp(lw - lw.max())
        h, _ = np.histogram(z, edges, weights=w)
        with np.errstate(divide="ignore"):
            fes = -kT * np.log(h / h.sum())              # empty bins -> +inf
        res = correct_fe_for_funnel(fes, fp, temperature=temperature, grid_min=centers[0],
                                    grid_max=centers[-1], cv_length_per_unit_ang=10.0,
                                    cv_origin=0.0, verbose=False)
        out[key] = float(res["dG_bind_std_kj_mol"])
    return out


SW_OOR_WARN, SW_OOR_MAX = 0.001, 0.01


SW_CLAMP_WARN, SW_CLAMP_RTOL = 0.05, 1e-6


def _edge_tol(lo, hi) -> float:
    return SW_CLAMP_RTOL * max(abs(lo), abs(hi), 1.0)


def sw_range_fraction(s, lo, hi) -> float:
    """Fraction of s outside [lo, hi] (0 for an empty array); values on an edge within float32 tolerance are inside."""
    s = np.asarray(s, float)
    tol = _edge_tol(lo, hi)
    return float(np.mean((s < lo - tol) | (s > hi + tol))) if s.size else 0.0


def check_sw_range(out_dir, npz_path, strict=False) -> dict:
    """Fraction of COLVAR s_w ("learned_sw" column, all walkers) outside the training range (sw_lo, sw_hi).

    The boundary correction treats that range as the reachable domain. Warns above SW_OOR_WARN;
    ``strict=True`` raises above SW_OOR_MAX. Rows on a clamp edge are in range; their fractions are
    reported and warned above SW_CLAMP_WARN.

    Returns
    -------
    dict with fraction, n, sw_lo, sw_hi, frac_at_lo_clamp, frac_at_hi_clamp and reason ("" when computed).
    """
    with np.load(npz_path, allow_pickle=False) as d:
        lo, hi = float(d["sw_lo"]), float(d["sw_hi"])
    vals = []
    for f in sorted(glob.glob(os.path.join(out_dir, "COLVAR_*.npy"))):
        col = _colvar_column(out_dir, os.path.basename(f)[len("COLVAR_"):-4], "learned_sw")
        if col is None:
            continue
        c = np.load(f)
        vals.append(c.reshape(len(c), -1)[:, col])
    if not vals:
        return {"reason": "no COLVAR with a learned_sw column in this run"}
    s = np.concatenate(vals)
    frac = sw_range_fraction(s, lo, hi)
    tol = _edge_tol(lo, hi)
    at_lo, at_hi = float(np.mean(np.abs(s - lo) <= tol)), float(np.mean(np.abs(s - hi) <= tol))
    out = {"fraction": frac, "n": int(len(s)), "sw_lo": lo, "sw_hi": hi,
           "frac_at_lo_clamp": at_lo, "frac_at_hi_clamp": at_hi, "reason": ""}
    logger.info("s_w at clamp: %.2f%% at sw_lo, %.2f%% at sw_hi", 100 * at_lo, 100 * at_hi)
    for name, f in (("sw_lo", at_lo), ("sw_hi", at_hi)):
        if f > SW_CLAMP_WARN:
            logger.warning("%.1f%% of COLVAR rows sit at the %s clamp: the bound basin may extend past it",
                           100 * f, name)
    msg = (f"s_w left its training range: boundary correction invalid "
           f"({100 * frac:.2f}% of {len(s)} COLVAR rows outside [{lo:.3f}, {hi:.3f}])")
    if strict and frac > SW_OOR_MAX:
        raise ValueError(msg)
    if frac > SW_OOR_WARN:
        logger.warning(msg)
    return out


def report(out_dir, temperature, grid_min=0.0, grid_max=1.0,
           cv_length_per_unit_ang=None, funnel_params=None, cv_is_funnel_z=False,
           hill_width=None) -> dict:
    grid = (float(grid_min), float(grid_max))
    origin = 0.0 if cv_is_funnel_z else grid[0]        # funnel z may extend below the site
    fp_path = os.path.join(out_dir, "funnel_params.npz")
    fp = funnel_params if funnel_params is not None else (
        load_funnel_params(fp_path) if os.path.exists(fp_path) else None)
    # No funnel: no standard-state correction, round trips on CV grid fractions.
    basins = funnel_basins(fp) if fp is not None else None
    if fp is None:
        logger.info("No funnel: reporting the uncorrected FES and its well depth.")

    walkers = load_hills(out_dir)
    if not walkers:
        logger.warning("No HILLS v2 files in %s: nothing to report.", out_dir)
    mode = detect_mode(walkers)
    logger.info("Metadynamics report: %s", mode.describe())
    # Unbound zero from the funnel geometry, 3 hill widths clear of the wall.
    ref_range = None
    if fp is not None:
        if hill_width is not None and cv_length_per_unit_ang:
            sigma_ang = float(hill_width) * float(cv_length_per_unit_ang)
        else:
            sigma_ang = 1.0
            logger.info("Hill width unknown: wall margin assumes sigma = 1 A.")
        ref_range = reference_range_from_funnel(fp, grid, cv_is_funnel_z, sigma_ang,
                                                out_dir=out_dir, cv_len=cv_length_per_unit_ang,
                                                hill_width_cv=hill_width)
        if ref_range is None:
            logger.warning("No geometric reference window: falling back to the last grid points.")
        else:
            logger.info("Unbound reference: CV %.3f-%.3f (cylinder from z_cc to z_max - 3 sigma).",
                        *ref_range)
    # Eq. 15 needs a length along the funnel axis; without one track the raw well depth.
    fp_corr = fp if cv_length_per_unit_ang else None
    if fp is not None and fp_corr is None:
        logger.info("CV has no length scale: no standard-state correction, reporting the well depth.")
    traces = walker_traces(out_dir, walkers, basins, grid, cv_is_funnel_z)
    estimates = build_estimates(walkers, fp_corr, temperature, grid, cv_length_per_unit_ang,
                                reference_range=ref_range, origin=origin)

    ma = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    walker_stats = {t.run_id: t.stats for t in traces}

    def visited_both(ids):
        return any(walker_stats[i]["frac_bound"] > 0.01 and walker_stats[i]["frac_unbound"] > 0.01
                   for i in ids if i in walker_stats)

    plateau_results, verdicts = {}, {}
    for e in estimates:
        ids = [e.label] if mode.layout == "isolated" else list(walker_stats)
        if e.t_ns.size:
            plateau_results[e.label] = ma.dG_plateau(e.t_ns, e.dG_kj, sampled=e.resolved,
                                                     both_basins=visited_both(ids))
        trans = [walker_stats[i] for i in ids if i in walker_stats]
        verdicts[e.label] = ma.assess_convergence(trans, e.fes_kj, grid_min=grid[0],
                                                  grid_max=grid[1],
                                                  dG_plateau_result=plateau_results.get(e.label),
                                                  reference_range=ref_range)

    good = [e.dG for e in estimates if e.dG is not None and e.complete]
    if mode.layout == "isolated":
        dG_main = float(np.mean(good)) if good else None
        dG_sd = float(np.std(good, ddof=1)) if len(good) > 1 else None
        all_conv = bool(verdicts) and all(v["converged"] for v in verdicts.values())
    else:
        dG_main = good[0] if good else None
        dG_sd = None
        all_conv = bool(verdicts) and next(iter(verdicts.values()))["converged"]
    if len(verdicts) > 1:
        reasons = [f"{lab}: {r}" for lab, v in verdicts.items() for r in v.get("reasons", [])]
    else:
        reasons = [r for v in verdicts.values() for r in v.get("reasons", [])]
    verdict = "converged" if all_conv else "NOT converged — " + "; ".join(reasons)
    kT_log10 = 8.314e-3 * float(temperature) * np.log(10)
    pKd = -dG_main / kT_log10 if dG_main is not None else None

    for e in estimates:
        logger.info("  %-28s ΔG°_b = %s kJ/mol%s", e.label,
                    f"{e.dG:.1f}" if e.dG is not None else f"n/a ({e.error})",
                    "" if e.complete else "  [incomplete walker]")
    if dG_main is not None:
        logger.info("  ΔG°_b = %.1f%s kJ/mol (%.1f kcal/mol), pKd = %.2f — %s",
                    dG_main, f" ± {dG_sd:.1f}" if dG_sd is not None else "",
                    dG_main * KJ_TO_KCAL, pKd, verdict)

    try:
        zr = z_reweighted_dG(out_dir, fp, temperature)
    except Exception as e:
        zr = {"reason": f"z reweighting failed: {e}"}
        logger.warning("%s", zr["reason"])
    if zr.get("reason"):
        logger.info("z reweighting skipped: %s", zr["reason"])
    else:
        logger.info("ΔG°_b reweighted onto funnel z: eq12 %.1f, c(t) %.1f kJ/mol (%d frames)",
                    zr["eq12_kj"], zr["ct_kj"], zr["n_frames"])

    return {
        "z_reweighted": zr,
        "mode": mode,
        "estimates": [(e.label, e.dG, e.complete) for e in estimates],
        "dG_bind_std_kj": dG_main,
        "funnel": fp is not None,
        "reference_range": ref_range,
        "well_depth_kj": [(e.label, _well_depth(e.fes_kj, grid, origin)) for e in estimates],
        "dG_bind_std_sd_kj": dG_sd,
        "dG_bind_std_kcal": dG_main * KJ_TO_KCAL if dG_main is not None else None,
        "pKd": pKd,
        "walker_stats": walker_stats,
        "n_walkers": len(traces),
        "n_reached_unbound": sum(1 for s in walker_stats.values() if s["reached_unbound"]),
        "n_round_trips": sum(s["n_round_trips"] for s in walker_stats.values()),
        "convergence": {"converged": all_conv, "verdict": verdict, "per_estimate": verdicts},
    }
