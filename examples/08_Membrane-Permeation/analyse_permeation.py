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
  fes_marginal_ARM.png  per two-CV arm: final 2D PMF with the symmetrised PMF(z) above it (with the
                   membrane colour bar) and PMF of the second CV (LMZ or cos θ) to its right
                   (Ligand_Bilayer_MD fancy_plot layout; same data as pmf.png)
  e2e_atoms.png    with an end-to-end arm: 2D drawing of the solute, the two end-to-end atoms highlighted
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
from matplotlib.ticker import MaxNLocator, MultipleLocator
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
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
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


Y_LABELS = {"lm_z": "Lipophilicity Moment Z", "cos_theta": "End-to-End cos θ"}
MEMBRANE_BAR = [(None, -3.0, "lightblue"), (-3.0, -2.0, ("lightblue", "salmon")), (-2.0, -1.5, "salmon"),
                (-1.5, -1.0, ("salmon", "grey")), (-1.0, 1.0, "grey"), (1.0, 1.5, ("grey", "salmon")),
                (1.5, 2.0, "salmon"), (2.0, 3.0, ("salmon", "lightblue")), (3.0, None, "lightblue")]


def _membrane_bar(ax, z_max, y_bottom, height):
    """Water / headgroup / tail colour bar (POPC, nm) along the bottom of ax."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    for lo, hi, color in MEMBRANE_BAR:
        lo, hi = (-z_max if lo is None else lo), (z_max if hi is None else hi)
        if isinstance(color, str):
            ax.add_patch(Rectangle((lo, y_bottom), hi - lo, height, color=color, lw=0))
            continue
        cmap, xs = LinearSegmentedColormap.from_list("grad", list(color)), np.linspace(lo, hi, 100)
        for i in range(len(xs) - 1):
            ax.add_patch(Rectangle((xs[i], y_bottom), xs[i + 1] - xs[i], height, color=cmap(i / 98), lw=0))


def fig_e2e_atoms(arm_dir, path, size=500):
    """2D drawing of the run's solute (its --ligand-sdf) with the two end-to-end atoms highlighted."""
    import json
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from autopath.metadynamics import end_to_end_atoms
    js = sorted(glob.glob(os.path.join(arm_dir, "walker_*.json")))
    sdf = json.load(open(js[0])).get("ligand_sdf") if js else None
    if not sdf or not os.path.exists(sdf):
        return
    mol = Chem.MolFromMolFile(sdf, removeHs=False)
    ends = end_to_end_atoms(mol)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]   # RemoveHs keeps this order
    mol2d = Chem.RemoveHs(mol)
    rdDepictor.Compute2DCoords(mol2d)
    hl = [heavy.index(i) for i in ends]
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    d.DrawMolecule(mol2d, highlightAtoms=hl, highlightAtomColors={i: (1.0, 0.55, 0.0) for i in hl},
                   highlightBonds=[])
    d.FinishDrawing()
    with open(path, "wb") as f:
        f.write(d.GetDrawingText())


def _fes_figure(z, fz, sdz, y, fy, sdy, f2, ylabel, title, path, stars=None):
    """2D FES f2 (len(y), len(z)) with PMF(z) ± sdz above it and PMF(y) ± sdy to its right
    (fancy_plot_no_lmz.py layout, plus the second marginal)."""
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 5], width_ratios=[10, 2.5, 0.5], wspace=0.08, hspace=0.1)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_top.plot(z, fz, color="black")
    ax_top.fill_between(z, fz - sdz, fz + sdz, color="gray", alpha=0.3)
    ax_top.set_ylabel("Free Energy (kcal/mol)", fontsize=14)
    ax_top.tick_params(axis="both", labelsize=14)
    ax_top.tick_params(axis="x", labelbottom=False)
    lo, hi = ax_top.get_ylim()
    _membrane_bar(ax_top, z[-1], lo, 0.06 * (hi - lo))

    ax_right.plot(fy, y, color="black")
    ax_right.fill_betweenx(y, fy - sdy, fy + sdy, color="gray", alpha=0.3)
    ax_right.set_xlabel("Free Energy\n(kcal/mol)", fontsize=14)
    ax_right.tick_params(axis="both", labelsize=14)
    ax_right.tick_params(axis="y", labelleft=False)

    mesh = ax.pcolormesh(z, y, f2, cmap="Spectral", shading="auto")
    ax.contour(z, y, f2, levels=np.linspace(np.nanmin(f2), np.nanmax(f2), 10),
               colors="black", linewidths=0.5, alpha=0.6)
    if stars is not None:
        ax.plot(stars[:, 0], stars[:, 1], ls="none", marker="*", ms=14, mfc="w", mec="k")
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel("Ligand COM Depth in Bilayer (nm)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(mesh, cax=fig.add_subplot(gs[1, 2]), label="Free Energy (kcal/mol)")
    fig.suptitle(title, fontsize=18, y=0.95)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.15)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_fes_marginal(arms, a, out_dir):
    """Per two-CV arm, fes_marginal_ARM.png from the same HILLS reconstruction as pmf.png
    (edge-corrected): the 2D FES with PMF(z) above it (second CV integrated out, bulk window = 0)
    and PMF(second CV) to its right (z integrated out, min = 0). Both 1D profiles symmetrised
    (z -> -z flips the second CV too), band = SD over estimates and both sides.
    Layout and colours from Ligand_Bilayer_MD fancy_plot_no_lmz.py."""
    from scipy.special import logsumexp
    for r in arms:
        try:
            est = [m for d in r["dirs"] for m in P.fes2d(d)]
        except ValueError:                                # single-CV arm
            continue
        m0, z, y = est[0], est[0]["z"], est[0]["y"]
        ylab = Y_LABELS.get(m0["y_name"], m0["y_name"])
        kt = r["est"]["kt"] / KCAL
        maps = np.array([m["fes"] for m in est]) / KCAL                    # (n_est, ny, nz)
        f2 = maps.mean(axis=0)
        marg = -kt * logsumexp(-f2 / kt, axis=0)                           # same marginal as the 1D PMF
        f2 = f2 - marg[P.bulk_mask(z, a.bulk)].mean()                      # bulk = 0, as in the 1D PMF
        title = f"Free Energy of {a.ligand + ' ' if a.ligand else ''}in a 100% POPC Bilayer ({r['name']})"

        fz, sdz = P.symmetric_mean_sd(r["pmf"][:, -1])                     # PMF(z), kcal/mol
        fy = -kt * logsumexp(-maps / kt, axis=2)                          # PMF(y) per estimate (n_est, ny)
        fy, sdy = P.symmetric_mean_sd(fy - fy.min(axis=1, keepdims=True))
        _fes_figure(z, fz, sdz, y, fy - fy.min(), sdy, f2, ylab, title,
                    os.path.join(out_dir, f"fes_marginal_{r['name']}.png"), stars=m0["basins"])
        if m0["y_name"] == "cos_theta":                  # same atoms for every e2e arm of a ligand
            fig_e2e_atoms(r["dirs"][0], os.path.join(out_dir, "e2e_atoms.png"))


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
    ap.add_argument("--ligand", default="", help="ligand name for the fes_marginal_*.png titles")
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
    fig_fes_marginal(arms, a, a.out)
    print(f"figures and summary.csv in {a.out}")
    if a.reports:
        run_reports(arms, a)


if __name__ == "__main__":
    main()
