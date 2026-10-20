#!/usr/bin/env python
"""Compare membrane-permeation metadynamics arms (PMF, symmetry convergence, transits).

    python analyse_permeation.py NAME=ARM_DIR [NAME=ARM_DIR ...] --out DIR

ARM_DIR is what launch_arm.sh made (run dirs r1, r2, ... inside) or a single run dir.
Every independent bias is one PMF estimate (isolated: one per walker; shared: one per run);
an arm's PMF is the mean of its estimates in free-energy space with their SD, and time is
the walker-ns summed over the arm, so arms of different layouts compare at equal cost.

Writes to --out (nothing else is touched):
  pmf.png          final PMF(z) per arm, mean ± SD over estimates; right: symmetrised
  convergence.png  symmetry error, centre barrier, well depth and estimate SD vs walker-ns
  z_traces.png     z(t) of every walker of each arm's first run
  summary.csv      one row per arm (also printed)
Works on runs still in progress (each estimate is cut at the shortest one).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autopath.metadynamics import permeation as P
from autopath.metadynamics.summary.data import load_run

plt.style.use("default")                        # autopath's import switches to fivethirtyeight
KCAL = 4.184
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#1f1f1e", "#6b6a64"


def run_dirs(arm_dir):
    if glob.glob(os.path.join(arm_dir, "HILLS_*.npz")):
        return [arm_dir]
    return sorted(d for d in glob.glob(os.path.join(arm_dir, "r*")) if glob.glob(os.path.join(d, "HILLS_*.npz")))


def analyse_arm(name, arm_dir, a):
    dirs = run_dirs(arm_dir)
    if not dirs:
        raise SystemExit(f"{name}: no runs with HILLS in {arm_dir}")
    est = P.arm_estimates(dirs, tuple(a.bulk), n=a.snapshots, n_t=a.snapshots)
    z, pmf = est["z"], est["pmf"] / KCAL                  # (n_est, n_t, nz)
    vma, vaa = P.symmetry_error(z, pmf, a.bulk[1])         # (n_est, n_t)
    feats = [[P.profile_features(z, P.symmetrise(p), a.membrane) for p in e] for e in pmf]
    centre = np.array([[f["centre"] for f in e] for e in feats])
    well = np.array([[f["well"] for f in e] for e in feats])
    trans = {f"{os.path.basename(d)}/{k}": v for d in dirs
             for k, v in P.walker_transits(d, a.bulk[0], a.inner).items()}
    drift = []
    for d in dirs:
        for w in load_run(d).walkers:
            if "bilayer_z" in w.cvs:
                drift.append(np.nanmax(np.abs(w.cvs["bilayer_z"])))
    firsts = [v["first_full_ns"] for v in trans.values() if v["first_full_ns"] is not None]
    row = dict(arm=name, runs=len(dirs), estimates=pmf.shape[0], walkers=len(trans),
               walker_ns=round(sum(v["ns"] for v in trans.values()), 1),
               ns_per_estimate=round(float(est["t_ns"][-1]), 1),
               centre_kcal=centre[:, -1].mean(), centre_sd=centre[:, -1].std(ddof=1) if len(centre) > 1 else np.nan,
               well_kcal=well[:, -1].mean(), well_sd=well[:, -1].std(ddof=1) if len(well) > 1 else np.nan,
               sym_mean_kcal=vaa[:, -1].mean(), sym_max_kcal=vma[:, -1].mean(),
               full_crossings=sum(v["full"] for v in trans.values()),
               half_transits=sum(v["half"] for v in trans.values()),
               first_crossing_median_ns=np.median(firsts) if firsts else np.nan,
               walkers_never_crossed=sum(v["full"] == 0 for v in trans.values()),
               bilayer_drift_max_nm=max(drift) if drift else np.nan)
    return dict(name=name, dirs=dirs, est=est, z=z, pmf=pmf, vaa=vaa, centre=centre, well=well, row=row)


def _style(ax, xlabel, ylabel):
    ax.grid(color="#e4e3dd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


def _band(ax, x, y, color, label):
    m = y.mean(0)
    ax.plot(x, m, color=color, lw=2, label=label)
    if len(y) > 1:
        sd = y.std(0, ddof=1)
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, lw=0)


def fig_pmf(arms, a, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for k, r in enumerate(arms):
        lab = f"{r['name']} ({r['row']['estimates']} est.)"
        _band(axes[0], r["z"], r["pmf"][:, -1], COLORS[k], lab)
        _band(axes[1], r["z"], P.symmetrise(r["pmf"][:, -1]), COLORS[k], lab)
    for ax, t in zip(axes, ["PMF, mean ± SD over estimates", "symmetrised (F(z) + F(-z)) / 2"]):
        _style(ax, "z, solute - membrane centre (nm)", "PMF (kcal/mol), bulk = 0")
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6)
        for s in (-1, 1):
            ax.axvspan(s * a.bulk[0], s * a.bulk[1], color="#e4e3dd", alpha=0.5, lw=0)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_convergence(arms, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    panels = [("vaa", "mean |F(z) - F(-z)| (kcal/mol)", "symmetry error per estimate (0 = converged)"),
              ("centre", "F(0) (kcal/mol)", "centre barrier, symmetrised"),
              ("well", "min F, |z| < membrane (kcal/mol)", "well depth, symmetrised"),
              ("sd", "SD over estimates at z = 0 (kcal/mol)", "spread between independent estimates")]
    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        for k, r in enumerate(arms):
            x = r["est"]["agg_ns"]
            if key == "sd":
                if len(r["centre"]) > 1:
                    ax.plot(x, r["centre"].std(0, ddof=1), color=COLORS[k], lw=2, label=r["name"])
            else:
                _band(ax, x, r[key], COLORS[k], r["name"])
        _style(ax, "walker-ns summed over the arm", ylab)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_traces(arms, path):
    fig, axes = plt.subplots(len(arms), 1, figsize=(12, 1.2 + 1.6 * len(arms)), squeeze=False, sharex=True)
    for k, (ax, r) in enumerate(zip(axes[:, 0], arms)):
        run = load_run(r["dirs"][0])
        hw = next(c.grid_max for c in run.cvs if c.name == P.Z_NAME)
        for j, w in enumerate(run.walkers):
            z = w.cvs[P.Z_NAME].astype(float)
            z[1:][np.abs(np.diff(z)) > hw] = np.nan          # periodic-image move through water, not a crossing
            ax.plot(w.t, z, color=COLORS[k], lw=0.5, alpha=0.4 + 0.6 * (j == 0))
        _style(ax, "walker time (ns)", "z (nm)")
        ax.set_title(f"{r['name']}: {os.path.basename(r['dirs'][0])}, {len(run.walkers)} walkers",
                     loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=ARM_DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bulk", nargs=2, type=float, default=[2.6, 3.3], help="|z| window of bulk water = 0 (nm)")
    ap.add_argument("--membrane", type=float, default=1.8, help="|z| splitting interior from interface (nm)")
    ap.add_argument("--inner", type=float, default=1.0, help="|z| a full crossing must reach (nm)")
    ap.add_argument("--snapshots", type=int, default=30)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    arms = [analyse_arm(*s.split("=", 1), a) for s in a.arms]
    table = pd.DataFrame([r["row"] for r in arms]).round(3)
    table.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.T.to_string(header=False))
    fig_pmf(arms, a, os.path.join(a.out, "pmf.png"))
    fig_convergence(arms, os.path.join(a.out, "convergence.png"))
    fig_traces(arms, os.path.join(a.out, "z_traces.png"))
    print(f"figures and summary.csv in {a.out}")


if __name__ == "__main__":
    main()
