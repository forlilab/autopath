# autopath/metadynamics/learned/pipeline.py
"""learn_cv: sMD pulls -> features -> relevance -> HLDA -> learned_cv.npz (+ report.png)."""
import logging
import os
import warnings

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .define import define_features
from .features import evaluate, features_to_dict
from .reduce import HLDA, lda_reference
from .select import find_pulls, labels_from_r, pull_frames, select

logger = logging.getLogger("autopath.metadynamics.learned")
ORDER = ["bound", "vestibule", "bulk"]
SUBSAMPLE = 10


def _speed(dat):
    return float(os.path.basename(dat).split("_v")[-1].split("_")[0])


def _load(u, dcd):
    u.load_new(dcd)
    return u


def _read(u, idx):
    """Coordinates (nm, float32) of the given frame indices."""
    return np.stack([u.atoms.positions.astype(np.float32) / 10.0 for _ in u.trajectory[list(idx)]])


def _frames_table(u, pulls, stride):
    tabs = []
    for dat, dcd in pulls:
        n = len(_load(u, dcd).trajectory)
        fr = pull_frames(dat, n).iloc[::stride].reset_index(drop=True)
        fr["pull"], fr["speed"] = os.path.basename(dat), _speed(dat)
        fr["frame"] = np.arange(n)[::stride][:len(fr)]
        tabs.append(fr)
    return tabs


def _subsample_coords(u, pulls, tabs, mask):
    """Coordinates of the rows selected by mask (concatenated-table order)."""
    out, off = [], 0
    for (_, dcd), fr in zip(pulls, tabs):
        rows = np.where(mask[off:off + len(fr)])[0]
        off += len(fr)
        if len(rows):
            out.append(_read(_load(u, dcd), fr.frame.values[rows]))
    return np.concatenate(out)


def _standardise(V, kept, fams):
    """Column mean/sd of the kept features; zero sd is an error except for forced F3 columns."""
    mu, sd = V[:, kept].mean(0), V[:, kept].std(0)
    for j in np.flatnonzero(sd == 0):
        if fams[kept[j]] != "F3":
            raise ValueError(f"kept feature {kept[j]} ({fams[kept[j]]}) has zero variance")
        logger.warning("forced F3 feature %d has zero variance; using sd=1", kept[j])
        sd[j] = 1.0
    return mu, sd


def learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
             direction="forward", stride=2, n_perm=1000, reducer=None) -> str:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
        return _learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
                         direction, stride, n_perm, reducer)


def _learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
              direction, stride, n_perm, reducer):
    os.makedirs(out_dir, exist_ok=True)
    pulls = find_pulls(traj_dir, direction)
    if not pulls:
        raise FileNotFoundError(f"no sMD pulls with aligned DCDs in {traj_dir}")
    u = mda.Universe(topology_pdb)
    tabs = _frames_table(u, pulls, stride)
    frames = pd.concat(tabs, ignore_index=True)
    first_r = np.array([f.r.iloc[0] for f in tabs])
    labels = labels_from_r(frames.r.values, first_r, float(frames.r.max()))
    for lab in ("bound", "vestibule"):
        if (labels == lab).sum() < 2:
            raise ValueError(f"fewer than 2 '{lab}' frames ({(labels == lab).sum()}); "
                             "cannot learn a CV (check direction/stride/pull data)")
    masks = {}
    for lab in ("bound", "vestibule"):
        m = np.zeros(len(labels), bool)
        m[np.where(labels == lab)[0][::SUBSAMPLE]] = True
        masks[lab] = m
    feats = define_features(topology_pdb, ligand_sdf, ligand_idx, pocket_idx,
                            bound_X=_subsample_coords(u, pulls, tabs, masks["bound"]),
                            exit_X=_subsample_coords(u, pulls, tabs, masks["vestibule"]))
    rows = []
    for (_, dcd), fr in zip(pulls, tabs):
        X = _read(_load(u, dcd), fr.frame.values)
        rows.append(np.asarray(evaluate(feats, X)))
        del X
    V = np.concatenate(rows)
    names = np.array([f.name for f in feats])
    fams = [f.family for f in feats]
    np.savez(os.path.join(out_dir, "features.npz"), V=V, names=names, r=frames.r.values,
             W=frames.W.values, pull=frames.pull.values.astype(str), speed=frames.speed.values,
             labels=labels.astype(str), **features_to_dict(feats))
    sel = select(V, list(names), fams, frames, labels, n_perm=n_perm, rng=np.random.default_rng(0))
    kept = np.array(sel["kept"])
    mu, sd = _standardise(V, kept, fams)
    Z = (V[:, kept] - mu) / sd
    reducer = reducer or HLDA(n_components=2)
    Wt = reducer.fit(Z, labels, ORDER)
    Wl = lda_reference(Z, labels, Wt.shape[1])
    cv = Z @ Wt
    bound = labels == "bound"
    acc = _grouped_accuracy(Z, labels, frames.pull.values)
    f3 = [k for k, i in enumerate(kept) if fams[i] == "F3"]
    acc_f3 = _grouped_accuracy(Z[:, f3], labels, frames.pull.values)
    rb = np.round(frames.r.values / 0.05)
    prof = pd.Series(cv[:, 0]).groupby(rb).mean()
    mono = float(spearmanr(prof.index, prof.values)[0])
    out = os.path.join(out_dir, "learned_cv.npz")
    np.savez(out, kept=kept, weights=Wt, lda_weights=Wl, mu=mu, sd=sd, fisher=sel["fisher"],
             p_hidden=sel["p_hidden"], rho_hidden=sel["rho_hidden"], names=names,
             grid_min=np.floor(cv.min(0) - 1), grid_max=np.ceil(cv.max(0) + 1),
             hill_width=0.5 * cv[bound].std(0), cv_acc=acc, f3_acc=acc_f3, monotonic_rho=mono,
             n_pulls=len(pulls), n_frames=len(frames), **features_to_dict(feats))
    _report(os.path.join(out_dir, "report.png"), frames, labels, cv, names[kept], Wt, Wl)
    logger.info("learned CV: %d candidates -> %d kept (%s); held-out accuracy %.3f (F3 alone %.3f); "
                "CV1 vs r monotonic rho %.2f", len(feats), len(kept), ", ".join(names[kept]), acc, acc_f3, mono)
    return out


def _grouped_accuracy(Z, labels, pulls):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold, cross_val_score
    if Z.shape[1] == 0:
        return float("nan")
    m = labels != "bulk"
    return float(cross_val_score(LinearDiscriminantAnalysis(), Z[m], labels[m], groups=pulls[m],
                                 cv=GroupKFold(5)).mean())


def _report(path, frames, labels, cv, kept_names, Wt, Wl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.style.context("default"):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        rb = np.round(frames.r.values / 0.05) * 0.05
        prof = pd.DataFrame({"rb": rb, "cv": cv[:, 0]}).groupby("rb").cv.agg(["mean", "std"])
        ax[0].fill_between(prof.index, prof["mean"] - prof["std"], prof["mean"] + prof["std"], alpha=0.3)
        ax[0].plot(prof.index, prof["mean"], lw=2)
        ax[0].set_xlabel("r pulling coordinate (nm)")
        ax[0].set_ylabel("learned CV1")
        x = np.arange(len(kept_names))
        ax[1].bar(x - 0.2, Wt[:, 0], 0.4, label="HLDA")
        ax[1].bar(x + 0.2, Wl[:, 0], 0.4, label="LDA (reference)")
        ax[1].set_xticks(x, kept_names, rotation=60, ha="right", fontsize=7)
        ax[1].legend(frameon=False)
        ax[1].set_title("CV1 weights (standardised features)", fontsize=9, loc="left")
        for lab, col in zip(ORDER, ("C3", "C2", "C0")):
            ax[2].hist(cv[labels == lab, 0], bins=50, histtype="step", lw=2, color=col, label=lab, density=True)
        ax[2].legend(frameon=False)
        ax[2].set_xlabel("learned CV1")
        fig.savefig(path, dpi=130)
        plt.close(fig)
