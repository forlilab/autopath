"""Synthetic metadynamics run directories (HILLS v2 / REWEIGHT / COLVAR) for summary tests."""
import os
import numpy as np

GRIDS = {"funnel_z": (-0.4, 2.4, 0.096, 10.0, 0.0), "learned_sw": (-2.0, 4.0, 0.15, np.nan, np.nan),
         "r_xy": (0.0, 1.5, 0.05, np.nan, np.nan), "dist": (0.2, 3.6, 0.01, np.nan, np.nan), "torsion": (-np.pi, np.pi, 0.1, np.nan, np.nan)}


def _traj(rng, lo, hi, n):
    # walker hops between the two ends of the grid
    x = np.empty(n); x[0] = lo + 0.1 * (hi - lo)
    for i in range(1, n):
        x[i] = np.clip(x[i - 1] + rng.normal(0, 0.04 * (hi - lo)), lo, hi)
    return x


def make_run(path, layout="shared", n_walkers=2, cvs=("funnel_z",), tempering="wt", n_rows=400, extend=False,
             truncate=None, reweight=True, monitors=("funnel_z",), seed=0, extra_meta=None, cross=False):
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
                       bounds_lo=np.array([-np.inf if c == "funnel_z" else np.nan for c in cvs]),
                       bounds_hi=np.array([GRIDS[c][1] if c == "funnel_z" else np.nan for c in cvs]),
                       bias_factor=20.0, temperature=300.0, tempering=tempering, cv_names=np.array(cvs),
                       cv_length_per_unit_ang=np.array([x[3] for x in g]), origin=np.array([x[4] for x in g]),
                       record_dt_ns=dt, dcd_interval_ns=dcd, **(extra_meta or {}))
            if "tt" in tempering:
                rec.update(vstar_kj=np.linspace(0, 10, nh), tt_basins=np.zeros((2, len(cvs))), tt_threshold=0.0,
                           tt_bias_factor=20.0)
            np.savez(os.path.join(path, f"HILLS_{s}.npz"), **rec)
    if truncate is not None:
        rid, kind = truncate                          # e.g. ("w1", "reweight-empty") or ("w1", "length")
        if kind == "reweight-empty":
            open(os.path.join(path, f"REWEIGHT_{rid}.npz"), "wb").close()
        elif kind == "length":
            c = np.load(os.path.join(path, f"COLVAR_{rid}.npy")); np.save(os.path.join(path, f"COLVAR_{rid}.npy"), c[:-7])
    return path
