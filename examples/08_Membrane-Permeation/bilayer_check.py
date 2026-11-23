#!/usr/bin/env python
"""Bilayer sanity check for one metaD arm, from the walkers' saved trajectories.

    python bilayer_check.py metadynamics/wt_lmz --out analysis/bilayer --stride 10

Per walker (WTMetaD_w<k>.dcd + .pdb in ARM_DIR), every --stride-th frame:
  bilayer centre   mean z of the lipid P atoms, relative to DUM (the zero of the membrane_z CV)
  thickness        mean z of upper-leaflet P - mean z of lower-leaflet P (leaflets fixed at frame 0)
  area per lipid   box x * y / lipids per leaflet
  box height       box z
  ligand z         ligand centroid relative to the bilayer centre (not DUM)
  core waters      water O within --core nm of the bilayer centre
  waters at ligand water O within --shell nm of any ligand atom
z values are minimum-image relative to DUM, so a bilayer split across the box edge is handled.

Writes OUT/bilayer_health.png, OUT/water_penetration.png and OUT/bilayer_check.npz (all series).
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def min_image(dz, L):
    return dz - L * np.round(dz / L)


def walker_series(pdb, dcd, a, dcd_ps):
    u = mda.Universe(pdb, dcd)
    P = u.select_atoms(f"resname {a.lipid} and name {a.p_atom}")
    O = u.select_atoms(f"resname {a.water} and name O")
    lig = u.select_atoms(f"resname {a.ligand}")
    dum = u.select_atoms(f"resname {a.center}")
    if not (len(P) and len(O) and len(lig) and len(dum) == 1):
        raise SystemExit(f"{pdb}: selections empty (P {len(P)}, water O {len(O)}, ligand {len(lig)}, DUM {len(dum)})")
    upper = None
    out = {k: [] for k in ("t_ns", "centre", "thick", "apl", "box_z", "lig_z", "core_w", "lig_w",
                           "mid_sd", "thick_sd", "p_near")}
    cell_nm, p_cut = getattr(a, "bin", 1.5), getattr(a, "p_near", 1.2)
    for ts in u.trajectory[::a.stride]:
        Lx, Ly, Lz = ts.dimensions[:3] / 10.0                       # Å -> nm
        z0 = dum.positions[0, 2] / 10.0
        pz = min_image(P.positions[:, 2] / 10.0 - z0, Lz)
        centre = pz.mean()
        if upper is None:
            upper = pz > centre
        lz = lig.positions[:, 2] / 10.0
        lz = lz[0] + min_image(lz - lz[0], Lz)                      # keep the ligand whole across the box edge
        oz = min_image(O.positions[:, 2] / 10.0 - z0 - centre, Lz)
        near = capped_distance(lig.positions, O.positions, a.shell * 10.0, box=ts.dimensions, return_distances=False)
        out["t_ns"].append(ts.frame * dcd_ps / 1000.0)
        out["centre"].append(centre)
        out["thick"].append(pz[upper].mean() - pz[~upper].mean())
        out["apl"].append(Lx * Ly / upper.sum())
        out["box_z"].append(Lz)
        out["lig_z"].append(min_image(lz.mean() - z0 - centre, Lz))
        out["core_w"].append(int((np.abs(oz) < a.core).sum()))
        out["lig_w"].append(len(np.unique(near[:, 1])) if len(near) else 0)
        # height map: per-leaflet mean P z on an x-y grid of ~cell_nm cells
        nb = max(2, int(round(min(Lx, Ly) / cell_nm)))
        xy = (P.positions[:, :2] / 10.0) % [Lx, Ly]
        cell = (xy[:, 0] / Lx * nb).astype(int) % nb * nb + (xy[:, 1] / Ly * nb).astype(int) % nb
        zu = np.array([pz[upper & (cell == c)].mean() if (upper & (cell == c)).any() else np.nan for c in range(nb * nb)])
        zl = np.array([pz[~upper & (cell == c)].mean() if (~upper & (cell == c)).any() else np.nan for c in range(nb * nb)])
        out["mid_sd"].append(np.nanstd((zu + zl) / 2))             # mid-plane roughness: bending / buckling
        out["thick_sd"].append(np.nanstd(zu - zl))                 # local thickness variation: dimples, thinning
        out["p_near"].append(int((np.abs(pz - centre) < p_cut).sum()))   # headgroups pulled toward the core
    return {k: np.asarray(v) for k, v in out.items()}


def _style(ax, ylabel):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def fig_health(S, path, title):
    rows = [("centre", "bilayer centre − DUM (nm)"), ("thick", "P–P thickness (nm)"),
            ("apl", "area per lipid (nm²)"), ("box_z", "box height (nm)"),
            ("mid_sd", "mid-plane roughness (nm)"), ("p_near", "P near the centre plane")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 12), sharex=True)
    for ax, (k, lab) in zip(axes, rows):
        for i, (w, s) in enumerate(S.items()):
            ax.plot(s["t_ns"], s[k], color=COLORS[i % len(COLORS)], lw=1.5, label=w)
        _style(ax, lab)
    axes[0].axhline(0, color=MUTED, lw=0.6)
    axes[0].legend(frameon=False, fontsize=8, ncol=len(S), loc="upper left")
    axes[-1].set_xlabel("walker time (ns)", color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_water(S, path, title, a):
    rows = [("lig_z", "ligand z − bilayer centre (nm)"), ("core_w", f"waters within {a.core} nm of centre"),
            ("lig_w", f"waters within {a.shell} nm of ligand")]
    fig, axes = plt.subplots(len(rows), len(S), figsize=(3.2 * len(S), 7.5), sharex=True, sharey="row", squeeze=False)
    for j, (w, s) in enumerate(S.items()):
        c = COLORS[j % len(COLORS)]
        for i, (k, lab) in enumerate(rows):
            ax = axes[i, j]
            ax.plot(s["t_ns"], s[k], color=c, lw=1.2)
            _style(ax, lab if j == 0 else "")
        axes[0, j].axhspan(-a.core, a.core, color=GRID, alpha=0.6, lw=0)       # the core band counted in row 2
        axes[0, j].set_title(w, loc="left", fontsize=9, color=INK)
        axes[-1, j].set_xlabel("walker time (ns)", color=INK, fontsize=9)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10, help="every n-th saved frame (10 = 1 ns at 100 ps frames)")
    ap.add_argument("--core", type=float, default=1.0, help="half-width of the core band for counting waters (nm)")
    ap.add_argument("--shell", type=float, default=0.5, help="water-ligand cutoff (nm)")
    ap.add_argument("--bin", type=float, default=1.5, help="height-map cell size (nm), ~3-4 lipids per leaflet")
    ap.add_argument("--p-near", type=float, default=1.2, help="|z| from the centre counted as a headgroup in the core")
    ap.add_argument("--lipid", default="POP"); ap.add_argument("--p-atom", default="P")
    ap.add_argument("--water", default="HOH"); ap.add_argument("--ligand", default="UNK")
    ap.add_argument("--center", default="DUM")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    S, keys = {}, {}
    for dcd in sorted(glob.glob(os.path.join(a.arm_dir, "*_w*.dcd"))):
        k = re.search(r"_w(\d+)\.dcd$", dcd).group(1)
        cfg = os.path.join(a.arm_dir, f"walker_{k}.json")
        c = json.load(open(cfg)) if os.path.exists(cfg) else {}
        m = re.search(r"equilibration_(-?[\d.]+)", c.get("state") or c.get("pdb") or "")
        label = f"w{k} · start z {float(m.group(1)):+.1f} nm" if m else f"w{k}"
        S[label] = s = walker_series(dcd[:-4] + ".pdb", dcd, a, c.get("dcd_ps", 100.0))
        keys[label] = f"w{k}"
        print(f"{label}: {len(s['t_ns'])} frames | centre drift max {np.abs(s['centre']).max():.2f} nm | "
              f"thickness {s['thick'].mean():.2f} ± {s['thick'].std():.2f} nm | "
              f"APL {s['apl'].mean():.3f} ± {s['apl'].std():.3f} nm² | core waters max {s['core_w'].max()} | "
              f"mid-plane roughness max {s['mid_sd'].max():.2f} nm | P near centre max {s['p_near'].max()}")
    if not S:
        raise SystemExit(f"no *_w<k>.dcd in {a.arm_dir}")
    title = os.path.basename(os.path.normpath(a.arm_dir))
    fig_health(S, os.path.join(a.out, "bilayer_health.png"), title)
    fig_water(S, os.path.join(a.out, "water_penetration.png"), title, a)
    np.savez(os.path.join(a.out, "bilayer_check.npz"), **{f"{keys[w]}_{k}": v for w, s in S.items() for k, v in s.items()})
    print(f"plots and bilayer_check.npz in {a.out}")


if __name__ == "__main__":
    main()
