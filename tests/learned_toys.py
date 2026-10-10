"""Shared toy system and residualised-s_w reference for the learned-CV tests."""
import numpy as np
import openmm as mm

from autopath.metadynamics.learned.features import Feature, evaluate, features_to_dict
from autopath.metadynamics.learned.sw import z_spline_values

FEATS = [Feature("p", "pair", "F1", pairs=[(0, 1)], r0=0.35),
         Feature("c", "contacts", "F2", group_a=[0, 1], group_b=[2, 3, 4], r0=0.3),
         Feature("o", "cos", "F8", groups=[[0], [1], [2], [3]])]
HOST, GUEST = [2, 3], [0, 1]
AXIS = np.array([0.6, -0.3, 0.74])
MASSES = [12.0, 31.0, 5.0, 16.0, 20.0]                     # unequal: z is mass-weighted
_RNG = np.random.default_rng(11)
X0 = _RNG.uniform(1.5, 2.6, size=(5, 3))
W = _RNG.normal(size=3)
MU = _RNG.normal(size=3)
SD = _RNG.uniform(0.5, 2.0, size=3)


def com(X, idx):
    m = np.array(MASSES)[idx]
    return (m[:, None] * X[idx]).sum(0) / m.sum()


def z_of(X):
    return float(AXIS / np.linalg.norm(AXIS) @ (com(X, GUEST) - com(X, HOST)))


Z0 = z_of(X0)
ZMIN, ZMAX = Z0 - 0.3, Z0 + 0.3
_G = np.linspace(ZMIN, ZMAX, 60)
TABLES = np.vstack([np.sin(4 * _G), (_G - Z0) ** 2, 0.5 * np.cos(3 * _G)])


def toy_system(x0=X0, box=10.0, masses=MASSES):
    s = mm.System()
    for m in masses:
        s.addParticle(m)
    s.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.2)
    for _ in masses:
        nb.addParticle(0.0, 0.3, 0.0)
    s.addForce(nb)
    tether = mm.CustomExternalForce("50*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    for p in ("x0", "y0", "z0"):
        tether.addPerParticleParameter(p)
    for i, x in enumerate(x0):
        tether.addParticle(i, list(x))
    s.addForce(tether)
    return s


def s_ref(X):
    f = evaluate(FEATS, X[None])[0]
    g = z_spline_values(TABLES, ZMIN, ZMAX, z_of(X))
    return float(np.sum(W * ((f - MU) / SD - g)))


def s_plain(X):
    return float(np.sum(W * (evaluate(FEATS, X[None])[0] - MU) / SD))


def write_npz_r2(path, residualised=True):
    c = s_ref(X0) if residualised else s_plain(X0)
    np.savez(path, kept=np.arange(3), weights=np.c_[W, -W], mu=MU, sd=SD, component=1, sw_weights=W,
             residualised=residualised, z_min=ZMIN if residualised else np.nan,
             z_max=ZMAX if residualised else np.nan,
             z_tables=TABLES if residualised else np.zeros((0, 0)),
             funnel_host=np.array(HOST), funnel_guest=np.array(GUEST), funnel_axis=AXIS / np.linalg.norm(AXIS),
             sw_lo=c - 2.0, sw_hi=c + 2.0, sw_grid_min=c - 2.6, sw_grid_max=c + 2.6, sw_hill_width=0.2,
             **features_to_dict(FEATS))
    return path


# rupture-class learn_cv toys: 5 candidate features over the 5-atom toy system
RUPTURE_FEATS = [Feature("F1_p", "pair", "F1", pairs=[(0, 1)], r0=0.35),
                 Feature("F7_d", "centroid_distance", "F7", groups=[[0, 1], [2, 3]]),
                 Feature("F2_c", "contacts", "F2", group_a=[0, 1], group_b=[2, 3, 4], r0=0.3),
                 Feature("F6_w", "contacts", "F6", group_a=[0], group_b=[4], r0=0.4),
                 Feature("F8_o", "cos", "F8", groups=[[0], [1], [2], [3]])]
FUNNEL = {"host_index": HOST, "guest_index": GUEST, "unbinding_axis": [1.0, 0.0, 0.0]}


def rupture_data(n_pull=10, n=400, speeds=(0.01, 0.02), seed=0):
    """Per pull: B 0-149, TS 200-249, U 280-399; the last pull excluded.

    Columns: TS signal, z, U signal, 2 noise. Returns (V, frames, labels, z, diag).
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    V, z, lab, r, pull, speed, diag = [], [], [], [], [], [], []
    for p in range(n_pull):
        rr = np.linspace(0.8, 2.6, n)
        lp = np.full(n, "", dtype=object)
        if p < n_pull - 1:
            lp[:150], lp[200:250], lp[280:] = "B", "TS", "U"
            diag.append({"pull": f"p{p}", "reason": ""})
        else:
            diag.append({"pull": f"p{p}", "reason": "no slip after the peak"})
        zz = rr - 0.8 + rng.normal(0, 0.02, n)
        V.append(np.c_[2.0 * (lp == "TS") + rng.normal(0, 0.3, n), zz + rng.normal(0, 0.05, n),
                       1.0 * (lp == "U") + rng.normal(0, 0.3, n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        z.append(zz)
        lab.append(lp)
        r.append(rr)
        pull += [f"p{p}"] * n
        speed += [speeds[p % len(speeds)]] * n
    frames = pd.DataFrame({"r": np.concatenate(r), "W": np.zeros(n * n_pull), "pull": pull, "speed": speed})
    return np.concatenate(V), frames, np.concatenate(lab), np.concatenate(z), diag


def pull_files(d, name="sMD_replica-1_v0.01_forward"):
    """Empty .dat / _aligned.dcd pair in directory d; returns (dat, dcd)."""
    dat, dcd = str(d / f"{name}.dat"), str(d / f"{name}_aligned.dcd")
    for f in (dat, dcd):
        open(f, "w").close()
    return dat, dcd
