"""FES minima, the frames of each basin and aligned multi-model PDBs of them.

python -m autopath.metadynamics.summary.frames RUN_DIR [--funnel F] [--learned-cv L] [--bands B]
    [--estimators rw_ct rw_final ...] [--pair x,y] [--at "x,y" ...] [--n-frames 10] [--radius-hills 1]
    [--max-minima 6] [--f-cut 6] [--depth 0.5] [--burn 0.2] [--seed 0] [--lig-resname UNK] [--keep all|shell|SEL]
    [--out DIR]
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.ndimage import minimum_filter

from .analysis import (ESTIMATORS, FUNNEL_Z, LABELS, RW_METHODS, _pool, ang_per_unit, bands, fes_bias,
                       fes_reweighted, kish_ess, length_unit, log_weights)
from .data import load_run

KJ_TO_KCAL = 1 / 4.184
WATER_RESNAMES = ("HOH", "WAT", "TIP3")
SHELL_SEL = "protein or resname {lig} or byres (resname {water} and around 5 resname {lig})"
FES_DEFAULTS = dict(min_count=0, min_bin_ess=0.0, smooth_sigma=2.0, bins=None, bins_per_hill=4,
                    bins_fallback=60, min_density=0.05)


# ---------------------------------------------------------------- minima

def _line_max(F, a, b):
    """Max of ``F`` on the nearest bins of the straight index-space line a→b; NaN counts as +inf."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(2 * np.max(np.abs(b - a))) + 2
    idx = np.rint(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)).astype(int)
    v = F[tuple(idx.T)]
    return float(np.max(np.where(np.isfinite(v), v, np.inf)))


def fes_minima(F, grids, size=3, depth_kcal=0.5, max_minima=6, min_separation=None, f_cut_kcal=6.0):
    """Local minima of a 1D or 2D free-energy map.

    A bin is a local minimum when it is the lowest finite value in its ``size``-bin neighbourhood (NaN bins
    are masked). Minima more than ``f_cut_kcal`` above the global minimum are dropped. Going up in F, a
    minimum is merged into an already kept, lower one when it lies within ``min_separation`` of it (an
    ellipse, one semi-axis per CV) or when the highest F on the straight line between them, minus its own F,
    is below ``depth_kcal``; masked bins on that line count as an infinite barrier.

    Parameters
    ----------
    F : ndarray
        Free energy (kcal/mol), axes in the order of ``grids``; NaN = masked.
    grids : sequence of ndarray
        Coordinates of each axis (CV units), uniform spacing.
    size : int or sequence of int
        Neighbourhood in bins per axis.
    depth_kcal : float
        Smallest barrier that keeps two minima apart.
    max_minima : int
    min_separation : float or sequence of float, optional
        Per-CV merge distance (CV units); default ``size`` bins per axis.
    f_cut_kcal : float

    Returns
    -------
    list of dict
        ``index`` (tuple of int), ``coords`` (tuple of float), ``F`` (float), sorted by F.
    """
    F = np.asarray(F, float)
    grids = [np.asarray(g, float) for g in grids]
    if F.ndim not in (1, 2):
        raise ValueError(f"{F.ndim}D map: minima not supported (1D or 2D)")
    if len(grids) != F.ndim or F.shape != tuple(len(g) for g in grids):
        raise ValueError(f"map shape {F.shape} does not match the grids: not supported")
    fin = np.isfinite(F)
    if not fin.any():
        return []
    size = np.broadcast_to(np.atleast_1d(np.asarray(size, int)), (F.ndim,))
    G = np.where(fin, F, np.inf)
    loc = fin & (G == minimum_filter(G, size=tuple(size), mode="constant", cval=np.inf))
    loc &= F <= F[fin].min() + f_cut_kcal
    step = np.array([g[1] - g[0] if len(g) > 1 else 1.0 for g in grids])
    sep = size * step if min_separation is None else np.broadcast_to(np.asarray(min_separation, float), (F.ndim,))
    idx = np.argwhere(loc)
    vals = F[tuple(idx.T)]
    coords = lambda p: np.array([g[i] for g, i in zip(grids, p)])
    kept = []
    for i in np.argsort(vals, kind="stable"):
        p = idx[i]
        if all(np.sum(((coords(p) - coords(q)) / sep) ** 2) > 1.0 and _line_max(F, q, p) - vals[i] >= depth_kcal
               for q in kept):
            kept.append(p)
            if len(kept) == max_minima:
                break
    return [dict(index=tuple(int(j) for j in p), coords=tuple(float(c) for c in coords(p)), F=float(F[tuple(p)]))
            for p in kept]


def _map_value(F, grids, x):
    """F at the nearest bin of each point ``x`` (n, ncv); NaN outside the grid."""
    x = np.atleast_2d(np.asarray(x, float))
    out = np.full(len(x), np.nan)
    ok = np.ones(len(x), bool)
    idx = []
    for j, g in enumerate(grids):
        d = g[1] - g[0] if len(g) > 1 else 1.0
        i = np.rint((x[:, j] - g[0]) / d).astype(int)
        ok &= (i >= 0) & (i < len(g)) & (np.abs(x[:, j] - g[np.clip(i, 0, len(g) - 1)]) <= 0.5 * d + 1e-12)
        idx.append(np.clip(i, 0, len(g) - 1))
    out[ok] = np.asarray(F, float)[tuple(i[ok] for i in idx)]
    return out


# ---------------------------------------------------------------- basin frames

def _widths(run, cvs, widths):
    if widths is not None:
        return np.broadcast_to(np.asarray(widths, float), (len(cvs),))
    info = {c.name: c for c in run.cvs}
    out = []
    for n in cvs:
        if n not in info or not np.isfinite(info[n].sigma) or info[n].sigma <= 0:
            raise ValueError(f"{n} has no hill width: pass widths= (basin radius not supported otherwise)")
        out.append(info[n].sigma)
    return np.array(out)


def basin_frames(run, ws, method, minimum, radius_hills=1.0, n_frames=10, seed=0, cvs=None, widths=None,
                 burn=0.2, fmap=None):
    """Weighted sample of the trajectory frames inside a basin.

    Candidates are records with a DCD frame, after the first ``burn`` fraction of each walker, whose CVs lie
    inside the ellipse of semi-axes ``radius_hills`` hill widths around ``minimum["coords"]``. Weights are the
    method's frame weights, pooled over ``ws`` with one normalisation over all post-burn records; ``"bias"``
    has no frame weights and uses ``rw_final``. ``n_frames`` are drawn without replacement with probability ∝
    weight (fewer if fewer candidates have weight).

    Parameters
    ----------
    ws : list of Walker
        Walkers of one bias (all for shared, one for isolated).
    method : str
        ``rw_ct``, ``rw_final``, ``rw_balanced`` or ``bias``.
    minimum : dict
        With ``coords`` (CV units, order of ``cvs``).
    cvs : sequence of str, optional
        CV names of ``coords``; default the biased CVs.
    widths : sequence of float, optional
        Hill width per CV; default the recorded hill widths.
    fmap : tuple, optional
        ``(grids, F)`` map (kcal/mol) for ``mean_F``.

    Returns
    -------
    dict
        ``frames`` (list of dict: ``rid``, ``segment``, ``segment_rid``, ``dcd``, ``pdb``, ``ref_pdb``,
        ``frame``, ``time_ns``, ``cvs``, ``weight``; by decreasing weight), ``n_available``, ``ess`` (Kish ESS
        of all candidates), ``mean_F`` (weight-averaged map F at the selected frames' bins, NaN without
        ``fmap``), ``weight_method``, ``reason`` (None, or why no frame was selected).
    """
    wm = "rw_final" if method == "bias" else method
    if wm not in RW_METHODS:
        raise ValueError(f"unknown method {method!r}: frame weights not supported")
    cvs = list(cvs or [c.name for c in run.cvs])
    c0 = np.asarray(minimum["coords"], float)
    if len(c0) != len(cvs):
        raise ValueError(f"minimum has {len(c0)} coordinates for {len(cvs)} CVs: not supported")
    sig = radius_hills * _widths(run, cvs, widths)
    vals, lws, info = [], [], []
    for w in ws:
        lw = log_weights(run, w, wm)
        keep = np.arange(len(lw)) >= int(burn * len(lw))
        vals.append(np.stack([np.asarray(w.cvs[n], float)[keep] for n in cvs], 1))
        lws.append(lw[keep])
        rows = np.flatnonzero(keep)
        info.append((w, rows))
    out = dict(frames=[], n_available=0, ess=0.0, mean_F=float("nan"), weight_method=wm, reason=None)
    if not ws:
        out["reason"] = "no walkers with REWEIGHT data"
        return out
    X, Wt = _pool(vals, lws)
    X = X.reshape(-1, len(cvs))
    owner = np.concatenate([np.full(len(r), k) for k, (_, r) in enumerate(info)])
    row = np.concatenate([r for _, r in info])
    frame = np.concatenate([w.record_frame[r] for w, r in info])
    inside = (frame >= 0) & (np.sum(((X - c0) / sig) ** 2, 1) <= 1.0)
    cand = np.flatnonzero(inside)
    out["n_available"], out["ess"] = int(len(cand)), kish_ess(Wt[cand])
    if not len(cand):
        out["reason"] = f"no DCD frame within {radius_hills:g} hill widths"
        return out
    p = Wt[cand]
    nz = int(np.sum(p > 0))
    if nz == 0:
        out["reason"] = "no weight on the basin frames"
        return out
    pick = np.random.default_rng(seed).choice(len(cand), size=min(int(n_frames), nz), replace=False, p=p / p.sum())
    pick = cand[pick[np.argsort(-p[pick], kind="stable")]]
    for i in pick:
        w, r = info[owner[i]][0], int(row[i])
        s = int(w.segment_of_record[r])
        dcd = w.dcd_files[s]
        out["frames"].append(dict(
            rid=w.rid, segment=s, segment_rid=os.path.basename(dcd)[len("WTMetaD_"):-len(".dcd")], dcd=dcd,
            pdb=dcd[:-4] + ".pdb", ref_pdb=w.dcd_files[0][:-4] + ".pdb", frame=int(frame[i]),
            time_ns=float(w.t[r]), cvs={n: float(X[i, j]) for j, n in enumerate(cvs)}, weight=float(Wt[i])))
    if fmap is not None:
        f = _map_value(fmap[1], fmap[0], X[pick])
        ok = np.isfinite(f)
        if ok.any() and Wt[pick][ok].sum() > 0:
            out["mean_F"] = float(np.average(f[ok], weights=Wt[pick][ok]))
    return out


# ---------------------------------------------------------------- aligned PDBs

class _Top:
    """Atom records, TER positions and bonds (OpenMM templates + CONECT) of a topology PDB."""

    def __init__(self, path):
        from openmm.app import PDBFile
        self.path = path
        self.lines, self.ter_after, last = [], {}, -1
        with open(path) as fh:
            for ln in fh:
                if ln.startswith(("ATOM", "HETATM")):
                    self.lines.append(ln.rstrip("\n").ljust(80))
                    last = len(self.lines) - 1
                elif ln.startswith("TER") and last >= 0:
                    self.ter_after[last] = ln.rstrip("\n")
        top = PDBFile(path).topology
        if top.getNumAtoms() != len(self.lines):
            raise ValueError(f"{path}: OpenMM reads {top.getNumAtoms()} atoms, file has {len(self.lines)}: "
                             "not supported")
        self.bonds = np.array([(b[0].index, b[1].index) for b in top.bonds()], int).reshape(-1, 2)
        self.ids = [ln[12:27] for ln in self.lines]                # name, altloc, resname, chain, resid
        self.serial = [ln[6:11] for ln in self.lines]

    def check_same(self, other):
        if self.ids != other.ids:
            raise ValueError(f"{other.path}: atoms differ from {self.path}: not supported")


def _universe(top, traj=None):
    """MDAnalysis Universe of ``top`` (+ ``traj``) carrying the topology's bonds only."""
    import MDAnalysis as mda
    u = mda.Universe(top.path, traj) if traj else mda.Universe(top.path)
    if len(u.atoms) != len(top.lines):
        raise ValueError(f"{top.path}: MDAnalysis reads {len(u.atoms)} atoms, file has {len(top.lines)}")
    if hasattr(u, "bonds"):
        u.delete_bonds(u.bonds)
        u.add_bonds([tuple(b) for b in top.bonds])
    else:
        u.add_TopologyAttr("bonds", [tuple(b) for b in top.bonds])
    return u


def _whole(u, groups):
    """Make each fragment of ``groups`` whole (bond-based)."""
    from MDAnalysis.lib.mdamath import make_whole
    frags = sorted(groups.fragments, key=len, reverse=True)
    for f in frags:
        if len(f) > 1:
            make_whole(f)
    return frags


def _image(u, lig_resname):
    """Make protein and ligand whole, centre the complex in the box, wrap the rest by residue around it."""
    from MDAnalysis.lib.distances import minimize_vectors
    from MDAnalysis.lib.mdamath import triclinic_vectors
    box = u.dimensions
    if box is None or not np.all(box[:3] > 0):
        raise ValueError("frame without a unit cell: imaging not supported")
    cx = u.select_atoms(f"protein or resname {lig_resname}")
    if not len(u.select_atoms("protein")):
        raise ValueError("no protein atoms: imaging not supported")
    frags = _whole(u, cx)
    pos = u.atoms.positions
    anchor = frags[0].center_of_geometry()
    for f in frags[1:]:
        d = f.center_of_geometry() - anchor
        pos[f.indices] += minimize_vectors(d[None], box)[0] - d
    inc = np.unique(np.concatenate([f.indices for f in frags]))
    centre = triclinic_vectors(box).sum(0) / 2
    pos += centre - pos[inc].mean(0)
    rest = np.setdiff1d(np.arange(len(pos)), inc)
    if len(rest):
        ri = u.atoms.resindices[rest]
        first = {}
        for i, r in zip(rest, ri):
            first.setdefault(r, i)
        lead = np.array([first[r] for r in ri])
        pos[rest] = pos[lead] + minimize_vectors(pos[rest] - pos[lead], box)
        d = pos[lead] - centre
        pos[rest] += minimize_vectors(d, box) - d
    u.atoms.positions = pos


def _conect(top, keep, lig_resname):
    """CONECT records of the ligand's topology bonds whose atoms are both written."""
    lig = {i for i, s in enumerate(top.ids) if s[5:8].strip() == lig_resname}
    nb = {}
    for a, b in top.bonds:
        if a in lig and b in lig and keep[a] and keep[b]:
            nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    out = []
    for a in sorted(nb):
        p = sorted(nb[a])
        for k in range(0, len(p), 4):
            out.append("CONECT" + top.serial[a] + "".join(top.serial[x] for x in p[k:k + 4]))
    return out


def _pdb_model(top, pos, keep):
    """ATOM/HETATM (and TER) records of the kept atoms of ``top`` with coordinates ``pos``."""
    out = []
    for i in np.flatnonzero(keep):
        ln = top.lines[i]
        x, y, z = pos[i]
        out.append(f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}".rstrip())
        if i in top.ter_after:
            out.append(top.ter_after[i])
    return out


def write_aligned_pdb(frames, out_path, topology=None, align_sel="protein and name CA", keep_sel="all",
                      ref=None, lig_resname="UNK", water_resnames=WATER_RESNAMES, dcd=False):
    """Multi-model PDB of frames imaged and superposed onto the reference (optionally also a DCD).

    Per frame: protein and ligand are made whole with the topology's bonds, the complex is centred in the
    box, other residues are moved to their minimum image around it, and every atom is moved by the
    rotation and translation that fit ``align_sel`` onto the reference coordinates. Atom records (names,
    residues, chains, serials, elements) are those of the topology. Every model holds the same atoms:
    ``keep_sel`` (evaluated after alignment, without PBC; ``{lig}``/``{water}`` filled from ``lig_resname``,
    ``water_resnames``) selects the union over all frames, so an atom shell keeps its atoms in every model
    at their own positions. CONECT records: the topology's ligand bonds.

    Parameters
    ----------
    frames : list of dict
        With ``dcd``, ``frame`` and (when ``topology``/``ref`` are None) ``pdb``, ``ref_pdb``.
    out_path : str
        ``.pdb``; with ``dcd``, also ``<stem>.dcd`` and ``<stem>_top.pdb`` (first model).
    dcd : bool
        Also write the DCD and its topology PDB.
    topology : str, optional
        Topology for every frame; default each frame's ``pdb``.
    ref : str, optional
        Reference PDB (made whole with its bonds, not moved); default the first frame's ``ref_pdb``.

    Returns
    -------
    dict
        ``n_models``, ``n_atoms``, ``rmsd_fit`` and ``rmsd_nofit`` (Å, ``align_sel`` vs the reference per
        model), ``lig_bonds``, ``lig_bond_max`` (Å, over models), ``ref``, ``files``.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.align import rotation_matrix
    if not frames:
        raise ValueError("no frames to write")
    sel_s = keep_sel.format(lig=lig_resname, water=" ".join(water_resnames))
    ref = ref or frames[0]["ref_pdb"]
    rt = _Top(ref)
    ru = _universe(rt)
    _whole(ru, ru.select_atoms(f"protein or resname {lig_resname}"))
    ridx = ru.select_atoms(align_sel).indices
    if not len(ridx):
        raise ValueError(f"no reference atoms in {align_sel!r}: not supported")
    rpos = ru.atoms.positions[ridx].astype(float)
    rc = rpos.mean(0)
    tops, unis, coords, rfit = {}, {}, [], []
    for fr in frames:
        tp = topology or fr["pdb"]
        if tp not in tops:
            tops[tp] = _Top(tp)
            rt.check_same(tops[tp])
        key = (tp, fr["dcd"])
        if key not in unis:
            unis[key] = _universe(tops[tp], fr["dcd"])
        u = unis[key]
        if not 0 <= fr["frame"] < len(u.trajectory):
            raise ValueError(f"{fr['dcd']}: frame {fr['frame']} not in {len(u.trajectory)} frames: not supported")
        u.trajectory[fr["frame"]]
        _image(u, lig_resname)
        mob = u.atoms.positions[ridx].astype(float)
        mc = mob.mean(0)
        R, r = rotation_matrix(mob - mc, rpos - rc)
        coords.append((u.atoms.positions - mc) @ np.asarray(R).T + rc)
        rfit.append(float(r))
    keep = np.zeros(len(rt.lines), bool)
    u = next(iter(unis.values()))
    for p in coords:
        u.atoms.positions = p
        keep[u.select_atoms(sel_s, periodic=False).indices] = True
    if not keep.any():
        raise ValueError(f"{sel_s!r} selects no atoms")
    lig = np.array([s[5:8].strip() == lig_resname for s in rt.ids])
    lb = np.array([(a, b) for a, b in rt.bonds if lig[a] and lig[b] and keep[a] and keep[b]], int).reshape(-1, 2)
    nofit = [float(np.sqrt(np.mean(np.sum((p[ridx] - rpos) ** 2, 1)))) for p in coords]
    bmax = float(max(np.linalg.norm(p[lb[:, 0]] - p[lb[:, 1]], axis=1).max() for p in coords)) if len(lb) else float("nan")
    con = _conect(rt, keep, lig_resname)
    with open(out_path, "w") as fh:
        fh.write(f"REMARK   1 {len(coords)} frames aligned on {align_sel} to {os.path.basename(ref)}\n")
        for k, p in enumerate(coords, 1):
            fh.write(f"MODEL     {k:4d}\n" + "\n".join(_pdb_model(rt, p, keep)) + "\nENDMDL\n")
        fh.write("\n".join(con + ["END"]) + "\n")
    files = [out_path]
    if dcd:
        stem = out_path[:-4] if out_path.endswith(".pdb") else out_path
        with open(stem + "_top.pdb", "w") as fh:
            fh.write("\n".join(_pdb_model(rt, coords[0], keep) + con + ["END"]) + "\n")
        with mda.Writer(stem + ".dcd", n_atoms=int(keep.sum())) as W:
            sub = mda.Universe(stem + "_top.pdb")
            for p in coords:
                sub.atoms.positions = p[keep]
                sub.dimensions = None
                W.write(sub.atoms)
        files += [stem + ".dcd", stem + "_top.pdb"]
    return dict(n_models=len(coords), n_atoms=int(keep.sum()), rmsd_fit=rfit, rmsd_nofit=nofit,
                lig_bonds=int(len(lb)), lig_bond_max=bmax, ref=ref, files=files)


# ---------------------------------------------------------------- driver

def _pair(run, pair):
    """CV names of the map: ``pair`` (1 or 2 names), else the fig_fes2d default for two biased CVs, else
    the single biased CV."""
    from .plots import _default_pair
    if pair is not None:
        names = [pair] if isinstance(pair, str) else [p for p in pair if p is not None]
    elif len(run.cvs) == 2:
        names = list(_default_pair(run)[0])
    elif len(run.cvs) == 1:
        names = [run.cvs[0].name]
    else:
        raise ValueError(f"{len(run.cvs)} biased CVs without pair=: not supported")
    if len(names) not in (1, 2):
        raise ValueError(f"pair {names}: 1 or 2 CVs supported")
    per = [c.name for c in run.cvs if c.periodic and c.name in names]
    if per:
        raise ValueError(f"periodic CV {per}: minima and basins not supported")
    return names


def _maps(run, m, names, burn, hill_mask_sigmas, fes_kw):
    """``{key: (grids, F kcal/mol, axes in ``names`` order)}``; key "pooled" or the walker id."""
    if m == "bias":
        bn = [c.name for c in run.cvs]
        if set(names) != set(bn):
            raise ValueError("bias-derived map: pair is not the biased CVs")
        fb = fes_bias(run, hill_mask_sigmas)
        items = [("pooled", fb.pooled)] if fb.pooled is not None else list(fb.per_walker.items())
        out = {}
        for k, (grids, F) in items:
            F = np.asarray(F, float) * KJ_TO_KCAL                     # stored (last CV, ..., first CV)
            if F.ndim == 2 and names == bn:
                F = F.T
            out[k] = ([grids[bn.index(n)] for n in names], F)
        return out
    fr = fes_reweighted(run, names[0], names[1] if len(names) > 1 else None, burn=burn, method=m, **fes_kw)
    items = [("pooled", fr.pooled)] if fr.pooled is not None else list(fr.per_walker.items())
    return {k: (list(v[:-1]), np.asarray(v[-1], float) * KJ_TO_KCAL) for k, v in items}


def _fmt(v):
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


def _stem(prefix, k, names, coords, units, mean_f):
    parts = [f"{n}{_fmt(v)}{u or ''}" for n, v, u in zip(names, coords, units)]
    f = f"F{_fmt(mean_f)}kcal" if np.isfinite(mean_f) else "Fna"
    return f"{prefix}{k:02d}_" + "_".join(parts) + f"_{f}"


def _plot_minima(run, bands_, names, maps, rows, m, path, fmax=None):
    import matplotlib
    import matplotlib.pyplot as plt
    from .plots import _band_edges, _cap, _fill_capped, _funnel_outline, _grid, _label, _rxy
    keys = list(maps)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys) + 0.8, 4.2), squeeze=False,
                             constrained_layout=True)
    two = len(names) == 2
    cap = _cap([maps[k][1] for k in keys], fmax) if two else None
    drawn = None
    for a, k in zip(axes[0], keys):
        grids, F = maps[k]
        if two:
            if np.isfinite(F).any():
                drawn = _fill_capped(fig, a, grids[0], grids[1], F.T, None, cap)
            a.set(ylabel=_label(run, names[1]))
            if run.funnel is not None and names[0] == FUNNEL_Z and names[1] == _rxy(run):
                a.plot(*_funnel_outline(run, ang_per_unit(run, names[0])), "k--", lw=1)
        else:
            a.plot(grids[0], F, color="k", lw=1.2)
            a.set(ylabel="F (kcal/mol)")
            _grid(a)
        a.set(xlabel=_label(run, names[0]))
        for v, kind in zip(names, ("v", "h")):
            if v in bands_:
                _band_edges(a, bands_[v], kind)
        for r in rows.get(k, []):
            xy = r["coords"] if two else (r["coords"][0], r["F_min"] if np.isfinite(r["F_min"]) else 0.0)
            a.plot(*xy, ls="none", marker="o", ms=11, mfc="w", mec="k", zorder=5)
            a.text(*xy, str(r["k"]), ha="center", va="center", fontsize=7, zorder=6)
        a.set_title(f"{k if k != 'pooled' else 'all'}: {LABELS[m]} F, numbered minima", fontsize=8, loc="left")
    if drawn is not None:
        fig.colorbar(drawn, ax=list(axes[0]), label="F (kcal/mol)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_csv(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def extract_fe_frames(run, bands_, out_dir, estimators=None, pair=None, n_frames=10, radius_hills=1.0,
                      max_minima=6, f_cut_kcal=6.0, depth_kcal=0.5, seed=0, at=None, burn=0.2,
                      hill_mask_sigmas=2.0, lig_resname="UNK", keep_sel="all", fmax=None, ref=None, **fes_kw):
    """Minima of each estimator's map, their basin frames and one aligned PDB per minimum.

    Maps are built as in ``fig_fes2d``: reweighted maps with ``fes_reweighted`` (same binning, smoothing and
    masks, ``fes_kw``), the bias map as hill-masked -scale·V. The minimum filter spans about one hill width
    (odd, at least 3 bins) and minima closer than one hill width merge (see ``fes_minima``); a CV without a
    hill width uses ``bins_per_hill`` bins. ``at`` gives CV points instead of detected minima (``pt`` files).
    Writes ``out_dir/fe_frames/<estimator>/[<rid>/]`` (rid only for isolated runs) with one multi-model PDB
    per minimum (see ``write_aligned_pdb``; ``keep_sel`` as there), ``minima.csv``, ``frames.csv``, and
    ``out_dir/fe_frames/<estimator>/minima.png``. Every PDB is aligned onto the same reference ``ref``
    (default ``system.pdb`` in the run directory's parent).

    Returns
    -------
    dict
        ``pair``, ``estimators`` (``{method: {key: {"dir", "minima": [row, ...]}}}``, plus ``png``),
        ``skipped`` (list of str).
    """
    fes_kw = {**FES_DEFAULTS, **fes_kw}
    if ref is None:
        ref = os.path.join(os.path.dirname(os.path.abspath(run.run_dir)), "system.pdb")
        if not os.path.exists(ref):
            raise ValueError(f"no reference {ref}: pass ref (system PDB) — not supported without one")
    names = _pair(run, pair)
    units = [length_unit(run, n) for n in names]
    iso = run.layout == "isolated"
    estimators = list(estimators or ESTIMATORS)
    bad = set(estimators) - set(ESTIMATORS)
    if bad:
        raise ValueError(f"unknown estimators {sorted(bad)}: not supported")
    info = {c.name: c for c in run.cvs}
    root = os.path.join(out_dir, "fe_frames")
    summary = dict(pair=names, estimators={}, skipped=[])
    for m in estimators:
        if not run.has_reweight:
            summary["skipped"].append(f"{m}: no REWEIGHT data, no frame weights")
            continue
        try:
            maps = _maps(run, m, names, burn, hill_mask_sigmas, fes_kw)
        except ValueError as e:
            summary["skipped"].append(f"{m}: {e}")
            continue
        est, rows_by = {}, {}
        for key, (grids, F) in maps.items():
            step = np.array([g[1] - g[0] for g in grids])
            widths = np.array([info[n].sigma if n in info and info[n].sigma > 0 else fes_kw["bins_per_hill"] * s
                               for n, s in zip(names, step)])
            size = [max(3, int(round(w / s)) | 1) for w, s in zip(widths, step)]
            if at is None:
                mins, prefix = fes_minima(F, grids, size=size, depth_kcal=depth_kcal, max_minima=max_minima,
                                          min_separation=widths, f_cut_kcal=f_cut_kcal), "min"
            else:
                pts = np.atleast_2d(np.asarray(at, float))
                if pts.shape[1] != len(names):
                    raise ValueError(f"--at points have {pts.shape[1]} values for {len(names)} CVs")
                fv = _map_value(F, grids, pts)
                mins, prefix = [dict(index=None, coords=tuple(map(float, p)), F=float(v)) for p, v in zip(pts, fv)], "pt"
            ws = [w for w in run.reweighted() if w.rid == key] if iso else run.reweighted()
            d = os.path.join(root, m, key) if iso else os.path.join(root, m)
            os.makedirs(d, exist_ok=True)
            rows, frows = [], []
            for k, mn in enumerate(mins, 1):
                bf = basin_frames(run, ws, m, mn, radius_hills=radius_hills, n_frames=n_frames, seed=seed + k,
                                  cvs=names, widths=widths, burn=burn, fmap=(grids, F))
                row = dict(k=k, label=prefix, F_min=mn["F"], mean_F=bf["mean_F"], n_frames=len(bf["frames"]),
                           n_available=bf["n_available"], ess=bf["ess"], weights=bf["weight_method"], pdb="",
                           reason=bf["reason"] or "", coords=mn["coords"],
                           **{n: c for n, c in zip(names, mn["coords"])})
                if bf["frames"]:
                    pdb = os.path.join(d, _stem(prefix, k, names, mn["coords"], units, bf["mean_F"]) + ".pdb")
                    res = write_aligned_pdb(bf["frames"], pdb, keep_sel=keep_sel, lig_resname=lig_resname, ref=ref)
                    row.update(pdb=os.path.basename(pdb), n_atoms=res["n_atoms"], fit_rmsd_max=max(res["rmsd_fit"]),
                               nofit_rmsd_max=max(res["rmsd_nofit"]), lig_bond_max=res["lig_bond_max"])
                    for j, fr in enumerate(bf["frames"], 1):
                        frows.append(dict(k=k, model=j, rid=fr["rid"], segment=fr["segment"],
                                          segment_rid=fr["segment_rid"], dcd=os.path.basename(fr["dcd"]),
                                          frame=fr["frame"], time_ns=fr["time_ns"], weight=fr["weight"],
                                          **fr["cvs"]))
                rows.append(row)
            cols = ["k", "label"] + names + ["F_min", "mean_F", "n_frames", "n_available", "ess", "weights",
                                             "n_atoms", "fit_rmsd_max", "nofit_rmsd_max", "lig_bond_max", "pdb", "reason"]
            _write_csv(os.path.join(d, "minima.csv"), rows, cols)
            _write_csv(os.path.join(d, "frames.csv"), frows,
                       ["k", "model", "rid", "segment", "segment_rid", "dcd", "frame", "time_ns"] + names + ["weight"])
            est[key] = dict(dir=d, minima=rows)
            rows_by[key] = rows
        png = os.path.join(root, m, "minima.png")
        _plot_minima(run, bands_, names, maps, rows_by, m, png, fmax)
        est["png"] = png
        summary["estimators"][m] = est
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--funnel"); ap.add_argument("--learned-cv"); ap.add_argument("--bands")
    ap.add_argument("--estimators", nargs="+", choices=ESTIMATORS)
    ap.add_argument("--pair", help="x[,y] CV names; default as the 2D maps")
    ap.add_argument("--at", action="append", help="CV point 'x[,y]' instead of detected minima (repeatable)")
    ap.add_argument("--n-frames", type=int, default=10); ap.add_argument("--radius-hills", type=float, default=1.0)
    ap.add_argument("--max-minima", type=int, default=6); ap.add_argument("--f-cut", type=float, default=6.0)
    ap.add_argument("--depth", type=float, default=0.5); ap.add_argument("--burn", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lig-resname", default="UNK")
    ap.add_argument("--keep", default="all", help="MDAnalysis selection written ('all'; 'shell' = protein, "
                    "ligand and waters within 5 Å of it in any frame)")
    ap.add_argument("--hill-mask-sigmas", type=float, default=2.0)
    ap.add_argument("--smooth-sigma", type=float, default=2.0); ap.add_argument("--min-density", type=float, default=0.05)
    ap.add_argument("--bins-per-hill", type=float, default=4)
    ap.add_argument("--ref", help="reference PDB every file is aligned onto (default <run_dir>/../system.pdb)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ov = None
    if a.bands:
        with open(a.bands) as fh:
            ov = json.load(fh)
    run = load_run(a.run_dir, funnel=a.funnel, learned_cv=a.learned_cv)
    at = [[float(v) for v in s.split(",")] for s in a.at] if a.at else None
    s = extract_fe_frames(run, bands(run, ov), a.out or a.run_dir, estimators=a.estimators,
                          pair=a.pair.split(",") if a.pair else None, n_frames=a.n_frames,
                          radius_hills=a.radius_hills, max_minima=a.max_minima, f_cut_kcal=a.f_cut,
                          depth_kcal=a.depth, seed=a.seed, at=at, burn=a.burn, hill_mask_sigmas=a.hill_mask_sigmas,
                          lig_resname=a.lig_resname, keep_sel=SHELL_SEL if a.keep == "shell" else a.keep,
                          smooth_sigma=a.smooth_sigma, min_density=a.min_density,
                          bins_per_hill=a.bins_per_hill, ref=a.ref)
    names = s["pair"]
    for m, est in s["estimators"].items():
        for key, e in est.items():
            if key == "png":
                continue
            print(f"\n{m} [{key}] -> {e['dir']}")
            print("  k  " + "  ".join(f"{n:>12s}" for n in names) + "   F_min  mean_F  n  avail    ESS")
            for r in e["minima"]:
                print(f"  {r['k']:<2d} " + "  ".join(f"{c:12.3f}" for c in r["coords"])
                      + f"  {r['F_min']:6.2f}  {r['mean_F']:6.2f} {r['n_frames']:2d} {r['n_available']:6d} "
                        f"{r['ess']:6.1f}  {r['reason']}")
    for x in s["skipped"]:
        print("skipped:", x)


if __name__ == "__main__":
    main()
