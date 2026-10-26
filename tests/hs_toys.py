"""Overdamped-Langevin sMD toys that mimic ``SMDData.raw_data``."""
import os

import numpy as np
import pandas as pd

R_KJ = 0.008314462618


def flat():
    return (lambda r: np.zeros_like(np.asarray(r, float))), (lambda r: np.zeros_like(np.asarray(r, float)))


def harmonic(r0=1.0, kappa=500.0):
    return (lambda r: 0.5 * kappa * (np.asarray(r, float) - r0) ** 2), (lambda r: kappa * (np.asarray(r, float) - r0))


def double_well(r_a=1.0, r_b=1.6, barrier=10.0):
    mid, h = 0.5 * (r_a + r_b), 0.5 * (r_b - r_a)

    def G0(r):
        x = (np.asarray(r, float) - mid) / h
        return barrier * (x ** 2 - 1.0) ** 2

    def dG0(r):
        x = (np.asarray(r, float) - mid) / h
        return barrier * 4.0 * x * (x ** 2 - 1.0) / h

    return G0, dG0


def _sample_initial(G0, lam0, k, beta, centre, rng, n, half_width=0.6, grid=1e-4):
    # Inverse CDF of rho0(r) exp(-beta u(r, lam0)) on a fine grid.
    g = np.arange(centre - half_width, centre + half_width, grid)
    lam0 = np.broadcast_to(np.asarray(lam0, float), (n,)) if np.ndim(lam0) else np.full(n, float(lam0))
    out = np.empty(n)
    for j in range(n):
        lw = -beta * (G0(g) + 0.5 * k * (g - lam0[j]) ** 2)
        p = np.exp(lw - lw.max())
        cdf = np.cumsum(p)
        out[j] = np.interp(rng.random() * cdf[-1], cdf, g)
    return out


def simulate_pulls(G0, dG0, n_pulls, speed, k=3765.6, T=300.0, r_start=1.0, length_nm=1.0,
                   dx=0.001, D=0.01, dt=0.002, lam0="common", r0_jitter_nm=0.0,
                   stop_steps=None, seed=0):
    beta = 1.0 / (R_KJ * T)
    assert beta * D * k * dt < 0.1, "dt too large for the spring"
    rng = np.random.default_rng(seed)
    n_moves = int(round(length_nm / dx))
    n_sub = max(1, int(round(dx / speed / dt)))
    dt_eff = dx / speed / n_sub
    assert beta * D * k * dt_eff < 0.1, "dt too large for the spring"

    if lam0 == "common":
        lam_start = np.full(n_pulls, float(r_start))
        r = _sample_initial(G0, lam_start, k, beta, r_start, rng, n_pulls)
    elif lam0 == "own":
        r = _sample_initial(G0, np.full(n_pulls, float(r_start)), k, beta, r_start, rng, n_pulls)
        r = r + rng.normal(0.0, r0_jitter_nm, n_pulls) if r0_jitter_nm else r
        lam_start = r.copy()
    else:
        raise ValueError(f"unknown lam0 {lam0!r}")

    lam = lam_start.copy()
    r_before = np.empty((n_moves, n_pulls))
    r_after = np.empty_like(r_before)
    dW = np.empty_like(r_before)
    lam_rec = np.empty_like(r_before)
    noise = np.sqrt(2.0 * D * dt_eff)
    for i in range(n_moves):
        new = lam_start + (i + 1) * dx
        dW[i] = 0.5 * k * ((r - new) ** 2 - (r - lam) ** 2)
        lam = new
        r_before[i] = r
        for _ in range(n_sub):
            r = r - beta * D * (dG0(r) + k * (r - lam)) * dt_eff + noise * rng.standard_normal(n_pulls)
        r_after[i] = r
        lam_rec[i] = lam

    work = np.cumsum(dW, axis=0)
    steps = np.arange(n_moves)
    frames = []
    for j in range(n_pulls):
        last = n_moves - 1 if stop_steps is None else int(stop_steps[j])
        sl = slice(0, last + 1)
        frames.append(pd.DataFrame({
            "trajname": f"replica-{j}_v{speed}",
            "speed": float(speed),
            "realized_speed": float(speed),
            "step": steps[sl],
            "time": (steps[sl] + 1) * dx / speed,
            "r_target": lam_rec[sl, j],
            "r_before": r_before[sl, j],
            "r_after": r_after[sl, j],
            "force": k * (lam_rec[sl, j] - r_after[sl, j]),
            "dW_protocol": dW[sl, j],
            "work": work[sl, j],
            "path": 1,
        }))
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(k=k, T=T, beta=beta)
    return out


def protocol_grids(rows):
    grids = {}
    for speed, g in rows.groupby("speed"):
        s0 = g["step"].min()
        r0 = g.loc[g["step"] == s0, "r_target"].median()
        dx = np.median(g.groupby("trajname")["r_target"].diff().dropna().values)
        steps = np.arange(s0, g["step"].max() + 1)
        grids[speed] = pd.DataFrame({"step": steps, "r_target_protocol": r0 + (steps - s0) * dx})
    return grids


def write_dat(rows, out_dir, header_k=True):
    k = rows.attrs.get("k", 3765.6)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (name, g) in enumerate(rows.groupby("trajname", sort=False)):
        speed = g["speed"].iloc[0]
        fn = os.path.join(out_dir, f"sMD_replica-{i}_v{speed}_forward.dat")
        d = pd.DataFrame({
            "step": g["step"], "time": g["time"], "r_target": g["r_target"],
            "r_before": g["r_before"], "r_after": g["r_after"], "force": g["force"],
            "U_cvpack": 0.5 * k * (g["r_after"] - g["r_target"]) ** 2,
            "dW_protocol": g["dW_protocol"],
        })
        with open(fn, "w") as f:
            if header_k:
                f.write(f"# spring_constant_kJ_mol_nm2={k:.10g}\n")
            d.to_csv(f, index=False)
        paths.append(fn)
    return paths
