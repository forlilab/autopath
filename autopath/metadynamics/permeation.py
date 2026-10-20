"""Membrane permeation PMFs from metadynamics on a signed membrane z (``CV.membrane_z_cv``).

Every independent bias (one isolated walker, or one shared-bias group) is one PMF estimate;
an arm's PMF is the free-energy-space mean of its estimates with their SD (Sun et al. 2016
Eq 13). Frames are never pooled across independent biases. A converged PMF is symmetric
about the bilayer centre, so ``symmetry_error`` is the reference-free accuracy check.
"""
import glob
import os

import numpy as np
from openmm import unit

from .Hills import fes_history, load_hills
from .summary.analysis import fes_marginal

Z_NAME = "membrane_z"


def _kt(walker):
    return float((unit.MOLAR_GAS_CONSTANT_R * walker["temperature"] * unit.kelvin)
                 .value_in_unit(unit.kilojoules_per_mole))


def _z_axis(walker, z_name):
    names = [str(x) for x in np.atleast_1d(walker["cv_names"])]
    if z_name not in names:
        raise ValueError(f"{z_name} not among the CVs {names}")
    k = names.index(z_name)
    z = np.linspace(float(np.atleast_1d(walker["grid_min"])[k]), float(np.atleast_1d(walker["grid_max"])[k]),
                    int(np.atleast_1d(walker["grid_points"])[k]))
    return z, len(names) - 1 - k, len(names)          # grid axes run in reverse variable order


def bulk_mask(z, bulk):
    """|z| inside the bulk-water window ``bulk = (lo, hi)`` (nm)."""
    return (np.abs(z) >= bulk[0]) & (np.abs(z) <= bulk[1])


def pmf_history(run_dir, bulk, n=30, z_name=Z_NAME) -> list:
    """PMF(z) snapshots of every independent bias in ``run_dir``, other CVs integrated out.

    Parameters
    ----------
    run_dir : str
        Directory with the walkers' ``HILLS_*.npz``.
    bulk : tuple
        (lo, hi) |z| window (nm) whose mean is the zero of each PMF.
    n : int, optional
        Snapshots per bias, evenly spaced in wall-clock deposition time.
    z_name : str, optional

    Returns
    -------
    list of dict
        One per bias: ``bias_dir``, ``run_ids``, ``t_ns`` (n,) summed walker time of that bias,
        ``z`` (nz,), ``pmf`` (n, nz) in kJ/mol, ``kt``.
    """
    walkers = load_hills(run_dir)
    if not walkers:
        raise ValueError(f"no HILLS v2 files in {run_dir}")
    z, axis, ncv = _z_axis(walkers[0], z_name)
    kt = _kt(walkers[0])
    keep = bulk_mask(z, bulk)
    out = []
    for h in fes_history(walkers, n=n):
        f = h["fes"]
        for ax in sorted((a for a in range(ncv) if a != axis), reverse=True):
            f = np.stack([fes_marginal(s, ax, kt) for s in f])
        f = f - f[:, keep].mean(axis=1, keepdims=True)
        out.append(dict(bias_dir=h["bias_dir"], run_ids=h["run_ids"], t_ns=h["t_ns"], z=z, pmf=f, kt=kt))
    return out


def fes2d(run_dir, z_name=Z_NAME) -> list:
    """Final 2D FES (z, other CV) of every independent bias in ``run_dir``, minimum = 0.

    Returns
    -------
    list of dict
        One per bias: ``bias_dir``, ``z`` (nz,), ``y`` (ny,), ``y_name``, ``fes`` (ny, nz) kJ/mol,
        ``basins`` (TT basin points in (z, y) order, or None).

    Raises
    ------
    ValueError
        A run without exactly two CVs (not supported).
    """
    walkers = load_hills(run_dir)
    if not walkers:
        raise ValueError(f"no HILLS v2 files in {run_dir}")
    w = walkers[0]
    names = [str(x) for x in np.atleast_1d(w["cv_names"])]
    if len(names) != 2:
        raise ValueError(f"2D FES needs two CVs, got {names}: not supported")
    z, _, _ = _z_axis(w, z_name)
    k = 1 - names.index(z_name)
    y = np.linspace(float(np.atleast_1d(w["grid_min"])[k]), float(np.atleast_1d(w["grid_max"])[k]),
                    int(np.atleast_1d(w["grid_points"])[k]))
    basins = None
    if "tt_basins" in w:
        b = np.asarray(w["tt_basins"], float).reshape(-1, 2)
        basins = b[:, [names.index(z_name), k]]
    out = []
    for h in fes_history(walkers, n=1):
        f = h["fes"][-1]                                # grid axes (cv1, cv0)
        if names.index(z_name) == 1:
            f = f.T
        out.append(dict(bias_dir=h["bias_dir"], z=z, y=y, y_name=names[k], fes=f - f.min(), basins=basins))
    return out


def symmetrise(pmf):
    """(F(z) + F(-z)) / 2 on a grid symmetric about 0 (last axis)."""
    return 0.5 * (pmf + pmf[..., ::-1])


def symmetric_mean_sd(pmf):
    """Mean and SD over estimates and both mirror sides: F(z) and F(-z) of every estimate pooled.

    ``pmf`` is (n_est, nz) on a grid symmetric about 0; one estimate gives SD = |F(z) - F(-z)| / sqrt(2).
    """
    v = np.concatenate([pmf, pmf[..., ::-1]], axis=0)
    return v.mean(axis=0), v.std(axis=0, ddof=1)


def symmetry_error(z, pmf, zmax):
    """max and mean |F(z) - F(-z)| over |z| <= zmax (Sun et al. 2016 Eqs 8-9), along the last axis."""
    if not np.allclose(z, -z[::-1]):
        raise ValueError("z grid is not symmetric about 0: not supported")
    d = np.abs(pmf - pmf[..., ::-1])[..., np.abs(z) <= zmax]
    return d.max(axis=-1), d.mean(axis=-1)


def profile_features(z, pmf, inner):
    """Barrier at the centre, deepest well and highest interface point of a (symmetrised) PMF.

    ``inner`` (nm) splits the bilayer interior (|z| < inner) from the interface.
    Returns ``centre``, ``well``, ``z_well``, ``interface``, ``z_interface`` (PMF units, nm).
    """
    pmf = np.asarray(pmf, float)
    a = np.abs(z)
    i_in = np.where(a < inner)[0]
    i_out = np.where(a >= inner)[0]
    w, b = i_in[np.argmin(pmf[i_in])], i_out[np.argmax(pmf[i_out])]
    return dict(centre=float(np.interp(0.0, z, pmf)), well=float(pmf[w]), z_well=float(a[w]),
                interface=float(pmf[b]), z_interface=float(a[b]))


def transits(z, bulk_lo, inner, half_width):
    """Membrane transits of one walker's z trace (nm).

    A full crossing goes from bulk (|z| >= ``bulk_lo``) on one side through the interior
    (|z| < ``inner``) to bulk on the other side; a jump > ``half_width`` between records is
    a periodic-image move and resets it. Half transits count bulk <-> interior alternations.

    Returns
    -------
    dict
        ``full`` count, ``first_full`` (record index or None), ``half`` count.
    """
    side, inside, full, first = 0, False, 0, None
    state, half = None, 0
    for i, x in enumerate(np.asarray(z, float)):
        if i and abs(x - z[i - 1]) > half_width:
            side, inside = 0, False
        if abs(x) >= bulk_lo:
            s = int(np.sign(x))
            if side and s != side and inside:
                full += 1
                first = i if first is None else first
            side, inside = s, False
        elif abs(x) < inner and side:
            inside = True
        st = "bulk" if abs(x) >= bulk_lo else "in" if abs(x) < inner else None
        if st and st != state:
            half += state is not None
            state = st
    return dict(full=full, first_full=first, half=half)


def walker_transits(run_dir, bulk_lo, inner, z_name=Z_NAME) -> dict:
    """``transits`` of every walker's ``COLVAR_<run_id>.npy`` in ``run_dir``, keyed by run_id.

    ``first_full_ns`` uses the HILLS ``record_dt_ns``.
    """
    out = {}
    for w in load_hills(run_dir):
        rid = str(w["run_id"])
        path = os.path.join(run_dir, f"COLVAR_{rid}.npy")
        if not os.path.exists(path):
            continue
        names = [str(x) for x in np.atleast_1d(w["cv_names"])]
        col = np.load(path).reshape(-1, len(names))
        hw = float(np.atleast_1d(w["grid_max"])[names.index(z_name)])
        r = transits(col[:, names.index(z_name)], bulk_lo, inner, hw)
        dt = float(w["record_dt_ns"])
        r.update(ns=len(col) * dt, first_full_ns=None if r["first_full"] is None else r["first_full"] * dt)
        out[rid] = r
    return out


def arm_estimates(run_dirs, bulk, n=30, n_t=30, z_name=Z_NAME) -> dict:
    """Independent PMF estimates of an arm on a common per-estimate time axis.

    Parameters
    ----------
    run_dirs : list of str
        Run directories of the arm (isolated walkers or shared groups, any mix of directories).
    bulk : tuple
        Bulk |z| window (nm), the PMF zero.
    n, n_t : int, optional
        Snapshots per bias; points on the common time axis.

    Returns
    -------
    dict
        ``z``, ``t_ns`` (n_t,) time per estimate, ``agg_ns`` (n_t,) summed over estimates,
        ``pmf`` (n_est, n_t, nz) kJ/mol, ``labels``, ``kt``.
    """
    est = [e for d in run_dirs for e in pmf_history(d, bulk, n=n, z_name=z_name)]
    if not est:
        raise ValueError("no biases found")
    z = est[0]["z"]
    t_end = min(e["t_ns"][-1] for e in est)
    t = np.linspace(t_end / n_t, t_end, n_t)
    pmf = np.empty((len(est), n_t, len(z)))
    for k, e in enumerate(est):
        idx = np.clip(np.searchsorted(e["t_ns"], t, side="left"), 0, len(e["t_ns"]) - 1)
        pmf[k] = e["pmf"][idx]
    return dict(z=z, t_ns=t, agg_ns=t * len(est), pmf=pmf, kt=est[0]["kt"],
                labels=[os.path.basename(os.path.normpath(e["bias_dir"])) + ":" + ",".join(map(str, e["run_ids"]))
                        for e in est])
