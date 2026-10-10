# autopath/metadynamics/learned/pipeline.py
"""learn_cv: sMD pulls -> features -> relevance -> HLDA -> learned_cv.npz (+ report.png)."""
import hashlib
import json
import logging
import os
import warnings

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .classes import R_ORDER, distance_features, find_milestones, heavy_ligand_idx, milestone_labels
from .define import define_features
from .features import evaluate, features_from_dict, features_to_dict
from .reduce import HLDA, lda_reference
from .select import find_pulls, labels_from_r, pull_frames, select

logger = logging.getLogger("autopath.metadynamics.learned")
ORDER = R_ORDER
SUBSAMPLE = 10


def _speed(dat):
    return float(os.path.basename(dat).split("_v")[-1].split("_")[0])


def _load(u, dcd):
    u.load_new(dcd)
    return u


def _read(u, idx, box=False):
    """Coordinates (nm, float32) of the given frame indices; box=True also returns box dims (A, deg) or None."""
    X, dims = [], []
    for ts in u.trajectory[list(idx)]:
        X.append(u.atoms.positions.astype(np.float32) / 10.0)
        dims.append(None if ts.dimensions is None else ts.dimensions.copy())
    X = np.stack(X)
    if not box:
        return X
    return X, (None if any(d is None or not d[:3].all() for d in dims) else np.stack(dims))


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


def _sha(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _code_hash() -> str:
    """Hash of the feature definition/evaluation sources."""
    from . import classes, define, features
    return hashlib.sha256("".join(_sha(m.__file__) for m in (features, define, classes)).encode()).hexdigest()


def _cache_key(pulls, stride, direction, topology_pdb, ligand_sdf, ligand_idx, pocket_idx,
               code_hash=None, classes="r", milestones=None, milestone_pocket_idx=None) -> str:
    """JSON of everything the extracted features depend on (inputs, contents, feature code)."""
    return json.dumps({
        "pulls": sorted((d, c, os.path.getmtime(d), os.path.getmtime(c)) for d, c in pulls),
        "stride": int(stride), "direction": direction,
        "topology_pdb": os.path.abspath(topology_pdb), "topology_sha": _sha(topology_pdb),
        "ligand_sdf": os.path.abspath(ligand_sdf), "ligand_sha": _sha(ligand_sdf),
        "code": code_hash or _code_hash(), "classes": classes,
        "milestones": sorted(_sha(m) for m in milestones) if classes == "milestones" else None,
        "milestone_pocket_idx": (sorted(int(i) for i in milestone_pocket_idx)
                                 if classes == "milestones" and milestone_pocket_idx is not None else None),
        "ligand_idx": sorted(int(i) for i in ligand_idx),
        "pocket_idx": sorted(int(i) for i in pocket_idx)})


def _cache_valid(npz_path, key) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            return "cache_key" in d.files and str(d["cache_key"]) == key
    except Exception:
        return False


def learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
             direction="forward", stride=2, n_perm=1000, reducer=None, hidden=True,
             use_cache=True, classes="r", milestones=None, milestone_pocket_idx=None) -> str:
    """classes="r": bound/vestibule/bulk cut from r; "milestones": nearest AutoPath milestone
    (milestones = XML list or directory; milestone_pocket_idx = atoms the milestone clustering used,
    required for milestones: the pocket atoms the milestones were clustered on)."""
    if classes not in ("r", "milestones"):
        raise ValueError(f"classes must be 'r' or 'milestones', not {classes!r}")
    if classes == "milestones":
        if milestones is None:
            raise ValueError("classes='milestones' needs milestones (XML list or directory)")
        milestones = find_milestones(milestones)
        if milestone_pocket_idx is None:
            raise ValueError("classes='milestones' needs milestone_pocket_idx: "
                             "the pocket atoms the milestones were clustered on")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
        return _learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
                         direction, stride, n_perm, reducer, hidden, use_cache,
                         classes, milestones, milestone_pocket_idx)


def _extract(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, pulls, stride, classes="r",
             milestones=None, milestone_pocket_idx=None):
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
    rows, drows = [], []
    ms = classes == "milestones"
    lig = heavy_ligand_idx(u, ligand_idx) if ms else None
    for (_, dcd), fr in zip(pulls, tabs):
        if ms:
            X, dims = _read(_load(u, dcd), fr.frame.values, box=True)
            rows.append(np.asarray(evaluate(feats, X)))
            drows.append(distance_features(u, X, lig, milestone_pocket_idx, dims))
        else:
            X = _read(_load(u, dcd), fr.frame.values)
            rows.append(np.asarray(evaluate(feats, X)))
        del X
    D = None
    if ms:
        D = np.concatenate(drows)
        labels, _, _ = milestone_labels(milestones, u, ligand_idx, milestone_pocket_idx, features=D)
    return feats, np.concatenate(rows), frames, labels, D


def _class_order(classes, n_ms):
    return ORDER if classes == "r" else [f"m{k + 1}" for k in range(n_ms)]


def _check_classes(labels, order):
    counts = {c: int((labels == c).sum()) for c in order}
    for c in order[:2] if set(order) == set(ORDER) else order:
        if counts[c] < 2:
            raise ValueError(f"class '{c}' has {counts[c]} frame(s) (need >= 2); counts {counts}; "
                             "check direction/stride/milestones")
    return counts


def _learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
              direction, stride, n_perm, reducer, hidden, use_cache, classes="r", milestones=None,
              milestone_pocket_idx=None):
    os.makedirs(out_dir, exist_ok=True)
    pulls = find_pulls(traj_dir, direction)
    if not pulls:
        raise FileNotFoundError(f"no sMD pulls with aligned DCDs in {traj_dir}")
    npz = os.path.join(out_dir, "features.npz")
    key = _cache_key(pulls, stride, direction, topology_pdb, ligand_sdf, ligand_idx, pocket_idx,
                     classes=classes, milestones=milestones, milestone_pocket_idx=milestone_pocket_idx)
    if use_cache and _cache_valid(npz, key):
        with np.load(npz, allow_pickle=False) as d:
            V, names, labels = d["V"], d["names"], d["labels"].astype(object)
            frames = pd.DataFrame({"r": d["r"], "W": d["W"], "pull": d["pull"], "speed": d["speed"]})
            feats = features_from_dict(d)
        logger.info("learned CV: reusing feature cache %s", npz)
    else:
        feats, V, frames, labels, D = _extract(topology_pdb, ligand_sdf, ligand_idx, pocket_idx,
                                               pulls, stride, classes, milestones, milestone_pocket_idx)
        names = np.array([f.name for f in feats])
        np.savez(npz, V=V, names=names, r=frames.r.values, W=frames.W.values,
                 pull=frames.pull.values.astype(str), speed=frames.speed.values,
                 labels=labels.astype(str), cache_key=np.array(key),
                 D=np.zeros((0, 0)) if D is None else D, **features_to_dict(feats))
        logger.info("learned CV: extracted features (%s) -> %s", "cache stale" if use_cache else "cache off", npz)
    order = _class_order(classes, len(milestones or ()))
    counts = _check_classes(labels, order)
    fams = [f.family for f in feats]
    sel = select(V, list(names), fams, frames, labels, n_perm=n_perm, rng=np.random.default_rng(0),
                 hidden=hidden, order=None if classes == "r" else order)
    kept = np.array(sel["kept"])
    mu, sd = _standardise(V, kept, fams)
    Z = (V[:, kept] - mu) / sd
    reducer = reducer or HLDA(n_components=2)
    Wt = reducer.fit(Z, labels, order)
    Wl = lda_reference(Z, labels, Wt.shape[1], order)
    cv = Z @ Wt
    bound = labels == order[0]
    acc_cls = order if classes == "milestones" else order[:2]
    acc = _grouped_accuracy(Z, labels, frames.pull.values, acc_cls)
    f3 = [k for k, i in enumerate(kept) if fams[i] == "F3"]
    acc_f3 = _grouped_accuracy(Z[:, f3], labels, frames.pull.values, acc_cls)
    rb = np.round(frames.r.values / 0.05)
    prof = pd.Series(cv[:, 0]).groupby(rb).mean()
    mono = float(spearmanr(prof.index, prof.values)[0])
    out = os.path.join(out_dir, "learned_cv.npz")
    np.savez(out, kept=kept, weights=Wt, lda_weights=Wl, mu=mu, sd=sd, fisher=sel["fisher"],
             p_hidden=sel["p_hidden"], rho_hidden=sel["rho_hidden"], names=names,
             grid_min=np.floor(cv.min(0) - 1), grid_max=np.ceil(cv.max(0) + 1),
             hill_width=0.5 * cv[bound].std(0), cv_acc=acc, f3_acc=acc_f3, monotonic_rho=mono,
             n_pulls=len(pulls), n_frames=len(frames), **features_to_dict(feats))
    _report(os.path.join(out_dir, "report.png"), frames, labels, cv, names[kept], Wt, Wl, order)
    for c in order:
        r = frames.r.values[labels == c]
        logger.info("learned CV class %s: %d frames, r %.2f-%.2f nm", c, counts[c],
                    r.min() if len(r) else np.nan, r.max() if len(r) else np.nan)
    logger.info("learned CV: %d candidates -> %d kept (%s); held-out accuracy %.3f (F3 alone %.3f); "
                "CV1 vs r monotonic rho %.2f", len(feats), len(kept), ", ".join(names[kept]), acc, acc_f3, mono)
    return out


def _grouped_accuracy(Z, labels, pulls, classes):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold, cross_val_score
    if Z.shape[1] == 0:
        return float("nan")
    m = np.isin(labels, classes)
    return float(cross_val_score(LinearDiscriminantAnalysis(), Z[m], labels[m], groups=pulls[m],
                                 cv=GroupKFold(5)).mean())


def _report(path, frames, labels, cv, kept_names, Wt, Wl, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.style.context("default"):
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
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
        cols = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(order)))
        for lab, col in zip(order, cols):
            ax[2].hist(cv[labels == lab, 0], bins=50, histtype="step", lw=2, color=col, label=lab, density=True)
            ax[3].boxplot(frames.r.values[labels == lab], positions=[order.index(lab)], widths=0.6,
                          vert=True, showfliers=False)
        ax[2].legend(frameon=False)
        ax[2].set_xlabel("learned CV1")
        ax[3].set_xticks(range(len(order)), [f"{c}\n n={int((labels == c).sum())}" for c in order], fontsize=8)
        ax[3].set_ylabel("r (nm)")
        ax[3].set_title("class vs r", fontsize=9, loc="left")
        fig.savefig(path, dpi=130)
        plt.close(fig)
