#!/usr/bin/env python
"""Compare membrane-permeation metadynamics arms (PMF, symmetry convergence, transits).

    python analyse_permeation.py NAME=ARM_DIR [NAME=ARM_DIR ...] --out DIR

ARM_DIR is what launch_arm.sh made (run dirs r1, r2, ... inside) or a single run dir.
Every independent bias is one PMF estimate (isolated: one per walker; shared: one per run);
an arm's PMF is the mean of its estimates in free-energy space with their SD, and time is
the walker-ns summed over the arm, so arms of different layouts compare at equal cost.

Writes to --out (nothing else is touched):
  pmf.png          final PMF(z) per arm, mean ± SD over estimates; right: symmetrised, band = SD over
                   estimates and both sides F(z), F(-z)
  convergence.png  mean and max symmetry error, centre barrier and well depth vs walker-ns
  fes-2d.png       final 2D PMF (z, second CV) per arm, mean over estimates; TT basins as stars
  z_traces.png     z(t) of every walker of each arm's first run
  reports/         with --reports: the autopath summary report of every run
                   (python -m autopath.metadynamics.summary), bound = |z| < --report-centre, unbound = bulk (+z)
  summary.csv      one row per arm (also printed)
Works on runs still in progress (each estimate is cut at the shortest one).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from autopath.metadynamics import permeation as P
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import _cap, _fill_capped
from autopath.metadynamics.summary.report import report

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
    return dict(name=name, dirs=dirs, est=est, z=z, pmf=pmf, vaa=vaa, vma=vma, centre=centre, well=well, row=row)


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
        m, sd = P.symmetric_mean_sd(r["pmf"][:, -1])
        axes[1].plot(r["z"], m, color=COLORS[k], lw=2, label=lab)
        axes[1].fill_between(r["z"], m - sd, m + sd, color=COLORS[k], alpha=0.18, lw=0)
    for ax, t in zip(axes, ["PMF, mean ± SD over estimates",
                            "symmetrised (F(z) + F(-z)) / 2, ± SD over estimates and both sides"]):
        _style(ax, "z, solute - membrane centre (nm)", "PMF (kcal/mol), bulk = 0")
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
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
    panels = [("vaa", "mean |F(z) - F(-z)| (kcal/mol)", "a  mean symmetry error (0 = converged)"),
              ("vma", "max |F(z) - F(-z)| (kcal/mol)", "b  worst-point symmetry error"),
              ("centre", "F(0) (kcal/mol)", "c  centre barrier, symmetrised"),
              ("well", "min F, |z| < membrane (kcal/mol)", "d  well depth, symmetrised")]
    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        for k, r in enumerate(arms):
            _band(ax, r["est"]["agg_ns"], r[key], COLORS[k], r["name"])
        _style(ax, "walker-ns summed over the arm", ylab)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_fes2d(arms, a, path):
    """Final 2D PMF per arm in the summary report's palette (1 kcal/mol levels, shared cap)."""
    maps = []
    for r in arms:
        try:
            est = [m for d in r["dirs"] for m in P.fes2d(d)]
        except ValueError:                                # single-CV arm
            continue
        f = np.mean([m["fes"] for m in est], axis=0) / KCAL
        maps.append((r["name"], est[0], f - f.min()))
    if not maps:
        return
    fmax = _cap([m[2] for m in maps])
    fig, axes = plt.subplots(1, len(maps), figsize=(4.8 * len(maps) + 1.2, 4.2), squeeze=False,
                             gridspec_kw=dict(wspace=0.3))
    for ax, (name, m, f) in zip(axes[0], maps):
        cf = _fill_capped(fig, ax, m["z"], m["y"], f, None, fmax)
        for s in (-1, 1):
            ax.axvline(s * a.bulk[0], color="k", ls=":", lw=0.8)
        if m["basins"] is not None:
            ax.plot(m["basins"][:, 0], m["basins"][:, 1], ls="none", marker="*", ms=14, mfc="w", mec="k", zorder=5)
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("z, solute - membrane centre (nm)", color=INK)
        ax.set_ylabel(m["y_name"], color=INK)
    fig.colorbar(cf, ax=axes[0].tolist(), label="F (kcal/mol), min = 0")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_traces(arms, a, path):
    fig, axes = plt.subplots(len(arms), 1, figsize=(8, 0.6 + 2.6 * len(arms)), squeeze=False, sharex=True)
    for k, (ax, r) in enumerate(zip(axes[:, 0], arms)):
        run = load_run(r["dirs"][0])
        hw = next(c.grid_max for c in run.cvs if c.name == P.Z_NAME)
        ax.axhspan(-a.bilayer, a.bilayer, color="#d9d8d2", alpha=0.6, lw=0)
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


def run_reports(arms, a):
    """Summary report per run dir: bound band = membrane centre, unbound = the +z bulk window."""
    bands = {P.Z_NAME: {"bound": [-a.report_centre, a.report_centre], "unbound": list(a.bulk)}}
    for r in arms:
        for d in r["dirs"]:
            run = os.path.basename(os.path.normpath(d))
            out = os.path.join(a.out, "reports", r["name"], run)
            report(d, bands=bands, out_dir=out, prefix=f"{r['name']}_{run}")
            print(f"report: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=ARM_DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bulk", nargs=2, type=float, default=[2.6, 3.3], help="|z| window of bulk water = 0 (nm)")
    ap.add_argument("--membrane", type=float, default=1.8, help="|z| splitting interior from interface (nm)")
    ap.add_argument("--inner", type=float, default=1.0, help="|z| a full crossing must reach (nm)")
    ap.add_argument("--bilayer", type=float, default=2.0,
                    help="|z| of the headgroup planes, shaded as the membrane on z_traces.png (nm)")
    ap.add_argument("--snapshots", type=int, default=30)
    ap.add_argument("--reports", action="store_true", help="also write the autopath summary report of every run")
    ap.add_argument("--report-centre", type=float, default=0.5,
                    help="|z| (nm) of the report's bound band, the membrane centre")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    arms = [analyse_arm(*s.split("=", 1), a) for s in a.arms]
    table = pd.DataFrame([r["row"] for r in arms]).round(3)
    table.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.T.to_string(header=False))
    fig_pmf(arms, a, os.path.join(a.out, "pmf.png"))
    fig_convergence(arms, os.path.join(a.out, "convergence.png"))
    fig_fes2d(arms, a, os.path.join(a.out, "fes-2d.png"))
    fig_traces(arms, a, os.path.join(a.out, "z_traces.png"))
    print(f"figures and summary.csv in {a.out}")
    if a.reports:
        run_reports(arms, a)


if __name__ == "__main__":
    main()
