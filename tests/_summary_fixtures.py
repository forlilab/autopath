"""Synthetic metadynamics run directories (HILLS v2 / REWEIGHT / COLVAR) for summary tests."""
import os
import numpy as np

GRIDS = {"funnel_z": (-0.4, 2.4, 0.096, 10.0, 0.0), "learned_sw": (-2.0, 4.0, 0.15, np.nan, np.nan),
         "r_xy": (0.0, 1.5, 0.05, np.nan, np.nan), "dist": (0.2, 3.6, 0.01, np.nan, np.nan), "torsion": (-np.pi, np.pi, 0.1, np.nan, np.nan),
         "contacts": (-0.75, 6.75, 0.25, np.nan, np.nan)}


def _traj(rng, lo, hi, n):
    # walker hops between the two ends of the grid
    x = np.empty(n); x[0] = lo + 0.1 * (hi - lo)
    for i in range(1, n):
        x[i] = np.clip(x[i - 1] + rng.normal(0, 0.04 * (hi - lo)), lo, hi)
    return x


def make_run(path, layout="shared", n_walkers=2, cvs=("funnel_z",), tempering="wt", n_rows=400, extend=False,
             truncate=None, reweight=True, monitors=("funnel_z",), seed=0, extra_meta=None, cross=False, bounds=None,
             dcd=False):
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)
    dt, dcd = 0.01, 0.05
    for w in range(n_walkers):
        rid = f"w{w}"
        segs = [rid, rid + "_x2"] if extend else [rid]
        bias_dir = os.path.abspath(path if layout != "isolated" else os.path.join(path, f"bias_{rid}"))
        for s in segs:
            X = np.stack([_traj(rng, GRIDS[c][0], GRIDS[c][1], n_rows) for c in cvs], 1)
            if cross:                                  # sweeps the whole grid every 0.5 ns
                tt = np.arange(n_rows) * dt
                for j, c in enumerate(cvs):
                    lo, hi = GRIDS[c][:2]
                    X[:, j] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sin(2 * np.pi * tt / 0.5 + w)
            np.save(os.path.join(path, f"COLVAR_{s}.npy"), X if len(cvs) > 1 else X[:, 0])
            t = np.arange(n_rows) * dt
            V = 2.0 * np.log1p(t)                      # slowly growing bias
            if reweight:
                mon = {f"monitor_{m}": _traj(rng, *GRIDS[m][:2], n_rows) for m in monitors if m not in cvs}
                np.savez(os.path.join(path, f"REWEIGHT_{s}.npz"), time_ns=t, bias_kj=V + rng.normal(0, 0.1, n_rows),
                         rct_kj=V, record_dt_ns=dt, dcd_interval_ns=dcd, temperature=300.0, bias_factor=20.0,
                         rct_ustride=5, tempering=tempering, **mon)
            nh = n_rows // 2
            hc = X[::2][:nh]
            g = [GRIDS[c] for c in cvs]
            rec = dict(hills_version=2, run_id=s, finished=True, bias_dir=bias_dir, started_at=0.0,
                       time_ns=np.arange(nh) * 2 * dt, wall_s=np.arange(nh, dtype=float), cv=hc.reshape(nh, len(cvs)),
                       height_kj=2.0 * np.exp(-np.arange(nh) / nh),
                       grid_min=np.array([x[0] for x in g]), grid_max=np.array([x[1] for x in g]),
                       grid_points=np.array([101] * len(cvs)), sigma=np.array([x[2] for x in g]),
                       periodic=np.array([c == "torsion" for c in cvs]),
                       bounds_lo=np.array([(bounds or {}).get(c, (-np.inf if c == "funnel_z" else np.nan,))[0]
                                           for c in cvs], float),
                       bounds_hi=np.array([(bounds or {}).get(c, (None, GRIDS[c][1] if c == "funnel_z" else np.nan))[1]
                                           for c in cvs], float),
                       bias_factor=20.0, temperature=300.0, tempering=tempering, cv_names=np.array(cvs),
                       cv_length_per_unit_ang=np.array([x[3] for x in g]), origin=np.array([x[4] for x in g]),
                       record_dt_ns=dt, dcd_interval_ns=dcd, **(extra_meta or {}))
            if "tt" in tempering:
                rec.update(vstar_kj=np.linspace(0, 10, nh), tt_basins=np.zeros((2, len(cvs))), tt_threshold=0.0,
                           tt_bias_factor=20.0)
            np.savez(os.path.join(path, f"HILLS_{s}.npz"), **rec)
            if dcd:
                write_tiny_md(os.path.join(path, f"WTMetaD_{s}.pdb"), os.path.join(path, f"WTMetaD_{s}.dcd"),
                              int(round(n_rows * dt / 0.05)), seed=int(rng.integers(1 << 30)))
    if truncate is not None:
        rid, kind = truncate                          # e.g. ("w1", "reweight-empty") or ("w1", "length")
        if kind == "reweight-empty":
            open(os.path.join(path, f"REWEIGHT_{rid}.npz"), "wb").close()
        elif kind == "length":
            c = np.load(os.path.join(path, f"COLVAR_{rid}.npy")); np.save(os.path.join(path, f"COLVAR_{rid}.npy"), c[:-7])
    return path


TINY_BOX = 24.0


def tiny_system():
    """Universe of a 4-residue backbone chain (ALA), a 3-atom ligand (UNK) and 12 waters in a 24 Å cube."""
    import MDAnalysis as mda
    pos, names, el, resn, resid = [], [], [], [], []
    for i in range(12):                                # N-CA-C zig-zag, O off each C along z
        pos.append((2.0 + 1.4 * i, 6.0 + 0.5 * (-1) ** i, 6.0))
        names.append("N CA C".split()[i % 3]); el.append("N" if i % 3 == 0 else "C")
        resn.append("ALA"); resid.append(i // 3 + 1)
        if i % 3 == 2:
            pos.append((pos[-1][0], pos[-1][1], 7.2)); names.append("O"); el.append("O")
            resn.append("ALA"); resid.append(i // 3 + 1)
    for k in range(3):
        pos.append((7.0 + 1.4 * k, 10.0 + 0.4 * (-1) ** k, 6.0)); names.append(f"C{k + 1}"); el.append("C")
        resn.append("UNK"); resid.append(5)
    for k in range(12):
        o = np.array([2.0 + 4.0 * (k % 4), 14.0 + 3.5 * (k // 4), 6.0 + 3.0 * (k % 2)])
        for nm, d in (("O", (0, 0, 0)), ("H1", (0.96, 0, 0)), ("H2", (-0.24, 0.93, 0))):
            pos.append(tuple(o + d)); names.append(nm); el.append(nm[0]); resn.append("HOH"); resid.append(6 + k)
    n = len(pos)
    rix = np.unique(resid, return_inverse=True)[1]
    u = mda.Universe.empty(n, n_residues=rix.max() + 1, atom_resindex=rix, trajectory=True)
    u.add_TopologyAttr("name", names); u.add_TopologyAttr("type", el); u.add_TopologyAttr("element", el)
    u.add_TopologyAttr("resname", [resn[list(rix).index(r)] for r in range(rix.max() + 1)])
    u.add_TopologyAttr("resid", np.arange(1, rix.max() + 2)); u.add_TopologyAttr("segid", ["A"])
    u.add_TopologyAttr("chainID", ["A"] * n)
    u.atoms.positions = np.array(pos, float)
    u.dimensions = [TINY_BOX] * 3 + [90.0] * 3
    return u


def write_tiny_md(pdb, dcd, n_frames, seed=0):
    """Topology PDB of ``tiny_system`` and a DCD of rigidly rotated, translated copies wrapped atom by atom
    into the box (molecules split across the boundary)."""
    import MDAnalysis as mda
    from scipy.spatial.transform import Rotation
    u = tiny_system()
    u.atoms.write(pdb)
    rng = np.random.default_rng(seed)
    x0 = u.atoms.positions.copy()
    c = x0.mean(0)
    with mda.Writer(dcd, n_atoms=len(x0)) as W:
        for _ in range(n_frames):
            R = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
            x = (x0 - c) @ R.T + c + rng.uniform(-TINY_BOX, TINY_BOX, 3)
            u.atoms.positions = np.mod(x, TINY_BOX)
            u.dimensions = [TINY_BOX] * 3 + [90.0] * 3
            W.write(u.atoms)
    return pdb, dcd


def add_fresh(path, rid, n_hills=0, n_rows=1, like="w0", bias_dir=None):
    """A just-started segment ``rid`` (as written at its first save): ``n_hills`` hills, ``n_rows`` records."""
    h = dict(np.load(os.path.join(path, f"HILLS_{like}.npz")))
    ncv = np.atleast_1d(h["cv_names"]).size
    h.update(run_id=rid, time_ns=np.zeros(n_hills), wall_s=np.zeros(n_hills), cv=np.full((n_hills, ncv), 0.1),
             height_kj=np.ones(n_hills), finished=False)
    if bias_dir is not None:
        h["bias_dir"] = bias_dir
    np.savez(os.path.join(path, f"HILLS_{rid}.npz"), **h)
    np.save(os.path.join(path, f"COLVAR_{rid}.npy"), np.full((n_rows, ncv) if ncv > 1 else n_rows, 0.1))
    r = dict(np.load(os.path.join(path, f"REWEIGHT_{like}.npz")))
    for k in list(r):
        if k in ("time_ns", "bias_kj", "rct_kj") or k.startswith("monitor_"):
            r[k] = r[k][:n_rows]
    np.savez(os.path.join(path, f"REWEIGHT_{rid}.npz"), **r)


def write_walker(path, rid, cvs, X, grid, hill_cv, hill_h, hill_t, *, bias=None, ct=None, wall=None, dt=0.01,
                 tempering="wt", bias_factor=20.0, bias_dir=None, bounds=None, periodic=None, temperature=300.0):
    """One walker segment from explicit arrays: COLVAR ``X`` (n, ncv), REWEIGHT ``bias``/``ct`` (n), hills.

    ``grid`` is ``{cv: (min, max, points, sigma)}``; ``bounds`` ``{cv: (lo, hi)}`` (NaN = none).
    """
    os.makedirs(path, exist_ok=True)
    X = np.asarray(X, float).reshape(len(X), len(cvs))
    nh = len(hill_h)
    np.save(os.path.join(path, f"COLVAR_{rid}.npy"), X if len(cvs) > 1 else X[:, 0])
    if bias is not None:
        np.savez(os.path.join(path, f"REWEIGHT_{rid}.npz"), time_ns=np.arange(len(X)) * dt, bias_kj=bias,
                 rct_kj=ct, record_dt_ns=dt, dcd_interval_ns=dt, temperature=temperature, bias_factor=bias_factor,
                 rct_ustride=1, tempering=tempering)
    g = [grid[c] for c in cvs]
    b = [(bounds or {}).get(c, (np.nan, np.nan)) for c in cvs]
    rec = dict(hills_version=2, run_id=rid, finished=True, bias_dir=os.path.abspath(bias_dir or path),
               started_at=0.0, time_ns=np.asarray(hill_t, float),
               wall_s=np.arange(nh, dtype=float) if wall is None else np.asarray(wall, float),
               cv=np.asarray(hill_cv, float).reshape(nh, len(cvs)), height_kj=np.asarray(hill_h, float),
               grid_min=np.array([x[0] for x in g], float), grid_max=np.array([x[1] for x in g], float),
               grid_points=np.array([x[2] for x in g], int), sigma=np.array([x[3] for x in g], float),
               periodic=np.array(periodic or [False] * len(cvs)),
               bounds_lo=np.array([x[0] for x in b], float), bounds_hi=np.array([x[1] for x in b], float),
               bias_factor=bias_factor, temperature=temperature, tempering=tempering, cv_names=np.array(cvs),
               cv_length_per_unit_ang=np.full(len(cvs), np.nan), origin=np.full(len(cvs), np.nan),
               record_dt_ns=dt, dcd_interval_ns=dt)
    if "tt" in tempering:
        rec.update(vstar_kj=np.zeros(nh), tt_basins=np.zeros((1, len(cvs))), tt_threshold=0.0, tt_bias_factor=20.0)
    np.savez(os.path.join(path, f"HILLS_{rid}.npz"), **rec)


TOY_GRID = {"x": (0.0, 3.0, 151, 0.06)}
TOY_BANDS = {"x": {"bound": [0.35, 0.85], "unbound": [1.8, 2.6]}}


def toy_potential(x):
    """1D well (kJ/mol) on the toy grid: a site well near 0.6 and a flat region beyond 1.5, walled at the ends."""
    x = np.asarray(x, float)
    return (-10.0 * np.exp(-(x - 0.6) ** 2 / (2 * 0.12 ** 2)) + 400.0 * np.clip(0.15 - x, 0, None) ** 2
            + 400.0 * np.clip(x - 2.85, 0, None) ** 2)


def make_wt_toy(path, n_steps=200_000, gamma=8.0, h0=1.0, stride=100, rec=20, seed=0, temperature=300.0, step=0.02,
                n_walkers=1, layout="single"):
    """Well-tempered metadynamics of overdamped Langevin dynamics on ``toy_potential``, written as a run.

    The felt bias is the grid bias (Gaussians on the grid, linear interpolation); c(t) is the Tiwary-Parrinello
    expression on the grid. Shared walkers deposit into one grid in lock step (wall time = step).
    """
    kt = 0.0083144626 * temperature
    lo, hi, npts, sig = TOY_GRID["x"]
    g = np.linspace(lo, hi, npts)
    dx = g[1] - g[0]
    rng = np.random.default_rng(seed)
    ddt = step ** 2 / 2                                # D*dt
    dU = np.gradient(toy_potential(g), dx)
    nb = 1 if layout != "isolated" else n_walkers
    V, dV, C = [np.zeros(npts) for _ in range(nb)], [np.zeros(npts) for _ in range(nb)], [0.0] * nb
    x = np.full(n_walkers, 0.6)
    recs = [dict(X=[], bias=[], ct=[]) for _ in range(n_walkers)]
    hills = [dict(cv=[], h=[], t=[], wall=[]) for _ in range(n_walkers)]
    noise = rng.normal(0.0, step, (n_steps, n_walkers))
    dt_ns = 1e-4
    a = gamma / (gamma - 1)

    def ct(v):
        m = v.max()
        return kt * (np.log(np.exp(a * (v - m) / kt).sum()) - np.log(np.exp((v - m) / ((gamma - 1) * kt)).sum())) \
            + m * (a - 1 / (gamma - 1))
    for i in range(n_steps):
        for k in range(n_walkers):
            j = 0 if nb == 1 else k
            if i % rec == 0:
                recs[k]["X"].append(x[k]); recs[k]["bias"].append(np.interp(x[k], g, V[j])); recs[k]["ct"].append(C[j])
            if i > 0 and i % stride == 0:
                h = h0 * np.exp(-np.interp(x[k], g, V[j]) / ((gamma - 1) * kt))
                V[j] += h * np.exp(-(g - x[k]) ** 2 / (2 * sig ** 2))
                dV[j], C[j] = np.gradient(V[j], dx), ct(V[j])
                hills[k]["cv"].append(x[k]); hills[k]["h"].append(h); hills[k]["t"].append(i * dt_ns)
                hills[k]["wall"].append(float(i))
            f = -(np.interp(x[k], g, dU) + np.interp(x[k], g, dV[j]))
            x[k] = min(max(x[k] + ddt * f / kt + noise[i, k], lo), hi)
    for k in range(n_walkers):
        r, h = recs[k], hills[k]
        bd = os.path.join(path, "bias") if nb == 1 else os.path.join(path, f"bias_w{k}")
        write_walker(path, f"w{k}", ("x",), np.array(r["X"])[:, None], TOY_GRID, np.array(h["cv"]), np.array(h["h"]),
                     np.array(h["t"]), bias=np.array(r["bias"]), ct=np.array(r["ct"]), wall=np.array(h["wall"]),
                     dt=rec * dt_ns, tempering="wt", bias_factor=gamma, bias_dir=bd, temperature=temperature)
    return path
