# autopath/metadynamics/learned/pipeline.py
"""learn_cv: sMD pulls -> features -> relevance -> HLDA -> learned_cv.npz (+ report.png)."""
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from collections import Counter

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.mdamath import triclinic_vectors
from scipy.stats import spearmanr

from ..Funnel import funnel_frame, funnel_z_ang, load_funnel_params
from .classes import (R_ORDER, RUPTURE_ORDER, distance_features, find_milestones, heavy_ligand_idx,
                      milestone_labels, rupture_labels)
from .define import define_features
from .features import evaluate, features_from_dict, features_to_dict
from .imaging import anchor_atoms, cn_centres, open_box, safe_frames
from .reduce import HLDA, lda_reference
from .modules import choose_resolution
from .sw import choose_component, fit_sw
from .select import find_pulls, labels_from_r, pull_frames, relevance, select
from .water import WATER_RES, define_water_features
from .visualize import write_feature_pse

logger = logging.getLogger("autopath.metadynamics.learned")
ORDER = R_ORDER
SUBSAMPLE = 10
SW_BOUND_SIGMAS = 3.0
NPZ_NAME, REPORT_NAME, RUPTURE_REPORT_NAME = "learned_cv.npz", "report.png", "rupture_report.json"
PSE_NAME = "features.pse"
ANALYSIS_DIRNAME = "analysis_traces"
FEATURE_SETS = ("water", "contacts")


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
        fr["dcd"] = os.path.basename(dcd)
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
    """Hash of the feature definition/evaluation sources and of funnel z."""
    from .. import Funnel
    from . import classes, define, features, imaging, select, water
    mods = (features, define, classes, select, water, imaging, Funnel, sys.modules[__name__])
    return hashlib.sha256("".join(_sha(m.__file__) for m in mods).encode()).hexdigest()


def _cache_key(pulls, stride, direction, topology_pdb, ligand_sdf, ligand_idx, pocket_idx,
               code_hash=None, classes="r", milestones=None, milestone_pocket_idx=None,
               exclude_families=(), funnel=None, masses=None, feature_set="contacts", system_sha=None) -> str:
    """JSON of everything the extracted features depend on (inputs, contents, feature code).

    funnel is the geometry from ``_funnel_geometry``; masses are hashed over its host, guest and frame atoms;
    system_sha is the hash of the serialised System (water features).
    """
    fz = None
    if funnel is not None:
        idx = list(funnel["host"]) + list(funnel["guest"])
        fr = funnel.get("frame")
        if fr is not None:
            idx += list(fr["orient"]) if "orient" in fr else list(fr["b"]) + list(fr["c"])
        fz = {"host": sorted(funnel["host"]), "guest": sorted(funnel["guest"]),
              "axis": [round(float(a), 9) for a in funnel["axis"]],
              "masses_sha": hashlib.sha256(np.asarray(masses, float)[idx].tobytes()).hexdigest()}
        if fr is not None and "orient" in fr:
            fz["frame"] = {"orient": list(fr["orient"]), "offset": [round(float(a), 9) for a in fr["offset"]],
                           "ref_sha": hashlib.sha256(np.round(np.asarray(fr["ref"], float), 9).tobytes()).hexdigest()}
        elif fr is not None:
            fz["frame"] = {"b": list(fr["b"]), "c": list(fr["c"]),
                           "coeffs": [round(float(a), 9) for a in fr["coeffs"]]}
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
        "pocket_idx": sorted(int(i) for i in pocket_idx),
        "exclude_families": sorted(exclude_families), "funnel": fz,
        "feature_set": feature_set, "system_sha": system_sha})


def _cache_valid(npz_path, key) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            return "cache_key" in d.files and str(d["cache_key"]) == key
    except Exception:
        return False


def _funnel_geometry(funnel):
    """{"host", "guest", "axis" (unit), "frame" (protein or orientation frame, or None)} from a funnel_params
    dict or its saved .npz path."""
    if isinstance(funnel, (str, os.PathLike)):
        funnel = load_funnel_params(str(funnel))
    axis = np.asarray(funnel["unbinding_axis"], float)
    return {"host": [int(i) for i in funnel["host_index"]], "guest": [int(i) for i in funnel["guest_index"]],
            "axis": axis / np.linalg.norm(axis), "frame": funnel_frame(funnel)}


def _funnel_z_nm(X, funnel, masses):
    """Funnel z (nm) per frame of X (n, N, 3) nm, as the run-time funnel z computes it.

    An orientation frame measures each frame in its own core orientation (exact at θ = 0).
    """
    return np.array([funnel_z_ang(x, funnel["host"], funnel["guest"], masses, funnel["axis"],
                                  frame=funnel["frame"]) for x in X]) / 10.0


def _load_system(system):
    """(openmm.System, sha256 of its XML) from a System or a system.xml path; (None, None) for None."""
    if system is None:
        return None, None
    import openmm as mm
    if isinstance(system, (str, os.PathLike)):
        with open(system) as fh:
            xml = fh.read()
        system = mm.XmlSerializer.deserialize(xml)
    else:
        xml = mm.XmlSerializer.serialize(system)
    return system, hashlib.sha256(xml.encode()).hexdigest()


def learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
             direction="forward", stride=2, n_perm=1000, reducer=None, hidden=True,
             use_cache=True, classes="r", milestones=None, milestone_pocket_idx=None,
             mosaic_resolution="cv", funnel=None, masses=None, exclude_families=None, temperature=300.0,
             analysis_dir=None, feature_set=None, system=None, milestone_pdbs=None) -> str:
    """classes="r": bound/vestibule/bulk cut from r; "milestones": nearest AutoPath milestone
    (milestones = XML list or directory; milestone_pocket_idx = atoms the milestone clustering used,
    required for milestones: the pocket atoms the milestones were clustered on);
    "rupture": B/TS/U from each pull's force trace, biased component s_w (funnel = funnel_params dict
    or .npz path, masses = system particle masses in Da; both required).
    mosaic_resolution: "cv" picks the MoSAIC resolution by pull-grouped CV on the relevant features; a float fixes it.
    exclude_families: feature families not defined; default ("F3", "F5") for "rupture", none otherwise.
    temperature (K), analysis_dir (default: analysis_traces next to traj_dir): the v->0 barrier check of "rupture".
    feature_set: "water" (W1/W3 cn features; default for "rupture", which alone supports it) or "contacts"
    (round-2 families; default otherwise). Water frames with a cn centre atom closer than CN_DMAX to a face of
    the imaged cell are excluded from training.
    system: openmm.System or system.xml path; required by "water" (exclusions); masses default to its masses.
    milestone_pdbs: milestone_<class>_frame_<n>.pdb paths (list, or {name: path}) shown in features.pse."""
    if not (isinstance(mosaic_resolution, str) and mosaic_resolution == "cv"
            or isinstance(mosaic_resolution, (int, float, np.floating)) and not isinstance(mosaic_resolution, bool)
            and 0 < mosaic_resolution <= 1):
        raise ValueError(f"mosaic_resolution must be 'cv' or a number in (0, 1], not {mosaic_resolution!r}")
    if classes not in ("r", "milestones", "rupture"):
        raise ValueError(f"classes must be 'r', 'milestones' or 'rupture', not {classes!r}")
    if classes == "milestones":
        if milestones is None:
            raise ValueError("classes='milestones' needs milestones (XML list or directory)")
        milestones = find_milestones(milestones)
        if milestone_pocket_idx is None:
            raise ValueError("classes='milestones' needs milestone_pocket_idx: "
                             "the pocket atoms the milestones were clustered on")
    if feature_set is None:
        feature_set = "water" if classes == "rupture" else "contacts"
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, not {feature_set!r}")
    if feature_set == "water" and classes != "rupture":
        raise ValueError(f"feature_set='water' with classes={classes!r} is not supported (rupture only)")
    if classes == "rupture":
        if funnel is None or (masses is None and system is None):
            raise ValueError("classes='rupture' needs funnel (funnel_params dict or .npz path) and masses "
                             "(system particle masses, Da): s_w is checked against funnel z")
        if feature_set == "water" and system is None:
            raise ValueError("feature_set='water' needs the openmm.System (system=, for the cn exclusions); "
                             "without it water features are not supported")
        funnel = _funnel_geometry(funnel)
    system, system_sha = _load_system(system)
    if classes == "rupture":
        if masses is None:
            from openmm import unit
            masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]
        masses = np.asarray(masses, float)
    elif funnel is not None:
        raise ValueError(f"funnel is only used by classes='rupture'; with classes={classes!r} it is not supported")
    if exclude_families is None:
        exclude_families = ("F3", "F5") if classes == "rupture" and feature_set == "contacts" else ()
    exclude_families = tuple(sorted(set(exclude_families)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
        return _learn_cv(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, traj_dir, out_dir,
                         direction, stride, n_perm, reducer, hidden, use_cache,
                         classes, milestones, milestone_pocket_idx, mosaic_resolution,
                         exclude_families, funnel, masses, temperature, analysis_dir, feature_set, system,
                         system_sha, milestone_pdbs)


def _extract(topology_pdb, ligand_sdf, ligand_idx, pocket_idx, pulls, stride, classes="r",
             milestones=None, milestone_pocket_idx=None, exclude_families=(), funnel=None, masses=None,
             feature_set="contacts", system=None):
    """(features, V, frames, labels, D, z, diag, extra); extra holds the water-set info, the per-row
    ``valid`` mask (rows too near the cell faces have NaN features) and the extraction time."""
    u = mda.Universe(topology_pdb)
    tabs = _frames_table(u, pulls, stride)
    frames = pd.concat(tabs, ignore_index=True)
    water = feature_set == "water"
    extra = {}
    if water:
        feats, info = define_water_features(topology_pdb, ligand_idx, system)
        feats = [f for f in feats if f.family not in exclude_families]
        wres = [r.atoms.indices for r in u.residues if r.resname in WATER_RES]
        if not wres:
            raise ValueError("topology has no water molecules; water features are not supported")
        centre_idx, centres = anchor_atoms(topology_pdb), cn_centres(feats)
        if len({len(w) for w in wres}) > 1:
            raise ValueError("water residues with different atom counts are not supported")
        water_mols = np.array(wres, dtype=int).reshape(len(wres), -1)
        extra["water"] = info
    else:
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
        feats = [f for f in feats if f.family not in exclude_families]
    ms, rup = classes == "milestones", classes == "rupture"
    lig = heavy_ligand_idx(u, ligand_idx) if ms else None
    rows, drows, zrows, rlabs, diag, valid = [], [], [], [], [], []
    t0 = time.perf_counter()
    for (dat, dcd), fr in zip(pulls, tabs):
        u = _load(u, dcd)
        if ms or water:
            X, dims = _read(u, fr.frame.values, box=True)
        else:
            X = _read(u, fr.frame.values)
        if ms:
            drows.append(distance_features(u, X, lig, milestone_pocket_idx, dims))
        if water:
            box = None if dims is None else np.stack([triclinic_vectors(dm) for dm in dims]) / 10.0
            ok, _ = safe_frames(X, box, centre_idx, centres, water_mols)
            V = np.full((len(X), len(feats)), np.nan)
            if ok.any():
                # every water within CN_DMAX of a centre atom is inside the cell: plain distances are exact
                V[ok] = evaluate(feats, X[ok], box=open_box(X[ok]))
            rows.append(V)
            valid.append(ok)
        else:
            rows.append(np.asarray(evaluate(feats, X)))
        if rup:
            lab, info = rupture_labels(dat, len(u.trajectory))
            rlabs.append(lab[fr.frame.values])
            diag.append(dict(info, pull=os.path.basename(dat)))
            zrows.append(_funnel_z_nm(X, funnel, masses))
        del X
    extra["s_per_frame"] = (time.perf_counter() - t0) / max(len(frames), 1)
    if water:
        extra["valid"] = np.concatenate(valid)
    D, z = None, None
    if ms:
        D = np.concatenate(drows)
        labels, _, _ = milestone_labels(milestones, u, ligand_idx, milestone_pocket_idx, features=D)
    if rup:
        labels, z = np.concatenate(rlabs), np.concatenate(zrows)
    return feats, np.concatenate(rows), frames, labels, D, z, diag, extra


def _class_order(classes, n_ms):
    if classes == "rupture":
        return list(RUPTURE_ORDER)
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
              milestone_pocket_idx=None, mosaic_resolution="cv", exclude_families=(), funnel=None, masses=None,
              temperature=300.0, analysis_dir=None, feature_set="contacts", system=None, system_sha=None, milestone_pdbs=None):
    os.makedirs(out_dir, exist_ok=True)
    pulls = find_pulls(traj_dir, direction)
    if not pulls:
        raise FileNotFoundError(f"no sMD pulls with aligned DCDs in {traj_dir}")
    npz = os.path.join(out_dir, "features.npz")
    key = _cache_key(pulls, stride, direction, topology_pdb, ligand_sdf, ligand_idx, pocket_idx,
                     classes=classes, milestones=milestones, milestone_pocket_idx=milestone_pocket_idx,
                     exclude_families=exclude_families, funnel=funnel, masses=masses,
                     feature_set=feature_set, system_sha=system_sha)
    if use_cache and _cache_valid(npz, key):
        with np.load(npz, allow_pickle=False) as d:
            V, names, labels = d["V"], d["names"], d["labels"].astype(object)
            frames = pd.DataFrame({"r": d["r"], "W": d["W"], "pull": d["pull"], "speed": d["speed"],
                                   "frame": d["frame"], "dcd": d["dcd"]})
            feats = features_from_dict(d)
            z = d["z"] if d["z"].size else None
            diag = json.loads(str(d["rupture_json"]))
            valid = d["valid"]
            extra = json.loads(str(d["extra_json"]))
        logger.info("learned CV: reusing feature cache %s", npz)
    else:
        feats, V, frames, labels, D, z, diag, extra = _extract(
            topology_pdb, ligand_sdf, ligand_idx, pocket_idx, pulls, stride, classes, milestones,
            milestone_pocket_idx, exclude_families, funnel, masses, feature_set, system)
        valid = extra.pop("valid", np.ones(len(V), bool))
        names = np.array([f.name for f in feats])
        np.savez(npz, V=V, names=names, r=frames.r.values, W=frames.W.values,
                 pull=frames.pull.values.astype(str), speed=frames.speed.values,
                 frame=frames.frame.values.astype(int), dcd=frames.dcd.values.astype(str), valid=valid,
                 labels=labels.astype(str), cache_key=np.array(key),
                 D=np.zeros((0, 0)) if D is None else D, z=np.zeros(0) if z is None else z,
                 rupture_json=np.array(json.dumps(diag)), extra_json=np.array(json.dumps(extra)),
                 **_water_keys(feature_set, extra), **features_to_dict(feats))
        logger.info("learned CV: extracted features (%s) -> %s", "cache stale" if use_cache else "cache off", npz)
    if feature_set == "water":
        extra["safe_radius"] = _safe_radius_summary(valid, labels, frames.pull.values)
        logger.info("learned CV: %d/%d frames with cn centres near the cell faces excluded (per class %s); "
                    "extraction %.3f s/frame", extra["safe_radius"]["n_excluded"], len(valid),
                    extra["safe_radius"]["excluded_per_class"], extra["s_per_frame"])
        V, labels, frames = V[valid], labels[valid], frames[valid].reset_index(drop=True)
        z = None if z is None else z[valid]
    order = _class_order(classes, len(milestones or ()))
    _check_classes(labels, order)
    fams = [f.family for f in feats]
    sel_order = None if classes == "r" else order
    res, cv_tab = _mosaic_resolution(V, fams, frames, labels, sel_order, mosaic_resolution)
    sel = select(V, list(names), fams, frames, labels, n_perm=n_perm, rng=np.random.default_rng(0),
                 hidden=hidden, order=sel_order, resolution=res)
    kept = np.array(sel["kept"])
    mu, sd = _standardise(V, kept, fams)
    Z = (V[:, kept] - mu) / sd
    shared = dict(out_dir=out_dir, feats=feats, names=names, kept=kept, mu=mu, sd=sd, sel=sel, res=res,
                  cv_tab=cv_tab, frames=frames, labels=labels, order=order, n_pulls=len(pulls),
                  feature_set=feature_set, extra=extra, system=system, fams=fams,
                  structures=_viewer_structures(topology_pdb, milestone_pdbs), ligand_idx=ligand_idx)
    if classes == "rupture":
        analysis = analysis_dir or os.path.join(os.path.dirname(os.path.normpath(traj_dir)), ANALYSIS_DIRNAME)
        return _finish_rupture(shared, Z, z, diag, reducer, funnel, analysis, temperature)
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
    out = _write_learned(shared, cv, Wt, Wl, dict(
        weights=Wt, lda_weights=Wl, grid_min=np.floor(cv.min(0) - 1), grid_max=np.ceil(cv.max(0) + 1),
        hill_width=0.5 * cv[bound].std(0), cv_acc=acc, f3_acc=acc_f3, monotonic_rho=mono))
    logger.info("learned CV: %d candidates -> %d kept (%s); held-out accuracy %.3f (F3 alone %.3f); "
                "CV1 vs r monotonic rho %.2f", len(feats), len(kept), ", ".join(names[kept]), acc, acc_f3, mono)
    return out


def _water_keys(feature_set, extra):
    """npz keys feature_set, w_p (NaN without W3) and patches_json."""
    w = extra.get("water", {})
    return dict(feature_set=np.array(feature_set),
                w_p=np.nan if w.get("w_p") is None else float(w["w_p"]),
                patches_json=np.array(json.dumps(w.get("patches", []))))


def _safe_radius_summary(valid, labels, pulls):
    bad = ~np.asarray(valid, bool)
    per_pull = {str(p): int(np.sum(bad & (pulls == p))) for p in np.unique(pulls)}
    return {"n_frames": int(len(bad)), "n_excluded": int(bad.sum()),
            "excluded_per_class": {c: int(np.sum(bad & (labels == c))) for c in RUPTURE_ORDER},
            "excluded_unlabelled": int(np.sum(bad & (labels == ""))),
            "excluded_per_pull": {p: n for p, n in per_pull.items() if n}}


def _viewer_structures(topology_pdb, milestone_pdbs):
    """Ordered {name: pdb} for features.pse: the bound topology first, then the milestones."""
    out = {"bound": topology_pdb}
    if isinstance(milestone_pdbs, dict):
        out.update(milestone_pdbs)
    else:
        for p in milestone_pdbs or ():
            stem = os.path.splitext(os.path.basename(p))[0]
            out[stem[len("milestone_"):] if stem.startswith("milestone_") else stem] = p
    return out


def _write_learned(shared, cv, Wt, Wl, extra):
    """Write learned_cv.npz (shared keys + extra) and report.png, log the class sizes; returns the npz path.

    cv, Wt, Wl are the projections and the HLDA / LDA weights shown in the report.
    """
    sh = shared
    names, kept, frames, labels, order = sh["names"], sh["kept"], sh["frames"], sh["labels"], sh["order"]
    out = os.path.join(sh["out_dir"], NPZ_NAME)
    np.savez(out, kept=kept, mu=sh["mu"], sd=sh["sd"], fisher=sh["sel"]["fisher"],
             p_hidden=sh["sel"]["p_hidden"], rho_hidden=sh["sel"]["rho_hidden"], names=names,
             n_pulls=sh["n_pulls"], n_frames=len(frames), mosaic_resolution=sh["res"],
             modules=sh["sel"]["modules"], mosaic_cv_resolution=sh["cv_tab"].resolution.values,
             mosaic_cv_mean=sh["cv_tab"]["mean"].values, mosaic_cv_sd=sh["cv_tab"]["sd"].values,
             mosaic_cv_n_modules=sh["cv_tab"].n_modules.values, **_water_keys(sh["feature_set"], sh["extra"]),
             **features_to_dict(sh["feats"]), **extra)
    _report(os.path.join(sh["out_dir"], REPORT_NAME), frames, labels, cv, names[kept], Wt, Wl, order,
            sh["cv_tab"], sh["res"])
    write_feature_pse(out, sh["structures"], os.path.join(sh["out_dir"], PSE_NAME), ligand_idx=sh["ligand_idx"])
    for c in order:
        r = frames.r.values[labels == c]
        logger.info("learned CV class %s: %d frames, r %.2f-%.2f nm", c, len(r),
                    r.min() if len(r) else np.nan, r.max() if len(r) else np.nan)
    return out


def _mosaic_resolution(V, fams, frames, labels, order, mosaic_resolution):
    """(resolution, CV table); the table is empty when the resolution is fixed."""
    if mosaic_resolution != "cv":
        return float(mosaic_resolution), pd.DataFrame(columns=["resolution", "mean", "sd", "n_modules"], dtype=float)
    _, relevant = relevance(V, fams, labels, order)
    idx = np.flatnonzero(relevant)
    res, tab = choose_resolution(V[:, idx], frames.r.values, frames.pull.values, frames.speed.values)
    logger.info("learned CV: MoSAIC resolution %.2f by pull-grouped CV on %d relevant features\n%s",
                res, len(idx), tab.to_string(index=False, float_format="%.3f"))
    return res, tab


def _grouped_accuracy(Z, labels, pulls, classes):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold, cross_val_score
    if Z.shape[1] == 0:
        return float("nan")
    m = np.isin(labels, classes)
    return float(cross_val_score(LinearDiscriminantAnalysis(), Z[m], labels[m], groups=pulls[m],
                                 cv=GroupKFold(5)).mean())


def multimodal(x, min_n=10):
    """(flag, [BIC 1 component, BIC 2 components]); flag: the 2-component Gaussian mixture has the lower BIC.

    (None, [NaN, NaN]) below min_n values.
    """
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, float).reshape(-1, 1)
    if len(x) < min_n:
        return None, [float("nan"), float("nan")]
    bic = [float(GaussianMixture(k, random_state=0).fit(x).bic(x)) for k in (1, 2)]
    return bool(bic[1] < bic[0]), bic


def _v0_barrier(ts_target, ts_r_after, analysis_dir, temperature):
    """r_TS of the v->0 PMF of each estimator in analysis_dir/mixture_pmfs.csv against the TS frames.

    The PMF coordinate is the spring-centre position, so the comparison uses ts_target (r_target of the
    TS frames, None when unavailable); ts_r_after (ligand r) is reported for reference only.

    Returns {"status": "skipped", "reason"} when the PMFs are absent, hold no v->0 profile, or ts_target is None.
    """
    path = None if analysis_dir is None else os.path.join(analysis_dir, "mixture_pmfs.csv")
    if path is None or not os.path.exists(path):
        return {"status": "skipped", "reason": f"no mixture_pmfs.csv in {analysis_dir or 'an analysis_traces directory'}"}
    from ...pulling.Estimators import KramersEstimator
    if ts_target is None:
        return {"status": "skipped", "reason": "pull .dat files lack r_target (the PMF coordinate)"}
    pmf = pd.read_csv(path, usecols=["speed", "estimator", "r_coord", "dG"])
    pmf = pmf[pmf.speed == 0.0]
    if pmf.empty:
        return {"status": "skipped", "reason": f"{path} has no v->0 (speed 0) profile"}
    kbt = KramersEstimator.R_GAS_KJ_MOL_K * temperature
    q = np.percentile(ts_target, [5, 25, 50, 75, 95])
    out = {"status": "ok", "coordinate": "r_target", "temperature_K": float(temperature),
           "ts_quantiles": [float(v) for v in q],
           "ts_r_after_quantiles": [float(v) for v in np.percentile(ts_r_after, [5, 25, 50, 75, 95])],
           "estimators": {}}
    for est, g in pmf.groupby("estimator"):
        g = g.dropna(subset=["r_coord", "dG"]).sort_values("r_coord")
        r_ts = KramersEstimator.detect_ts(g.r_coord.to_numpy(float), g.dG.to_numpy(float), kbt)
        out["estimators"][str(est)] = (
            {"r_ts": None, "ts_percentile": None, "within_ts_iqr": None} if r_ts is None else
            {"r_ts": float(r_ts), "ts_percentile": float(np.mean(ts_target <= r_ts)),
             "within_ts_iqr": bool(q[1] <= r_ts <= q[3])})
    return out


def rupture_report(frames, labels, z, Z, kept_names, diag, order=RUPTURE_ORDER, analysis_dir=None,
                   temperature=300.0, kept_families=None) -> dict:
    """Class sizes, pull exclusions, TS r and z median/IQR/multimodality, per-speed TS feature means and
    their max standardised mean difference across speeds (NaN with one speed), and the v->0 barrier check.

    Parameters
    ----------
    frames : DataFrame with r, pull, speed per frame
    labels : (n,) object array of class names ("" = unlabelled)
    z : (n,) funnel z per frame (nm)
    Z : (n, k) standardised kept features
    kept_names : k names
    diag : per-pull dicts with ``reason`` ("" when labelled)
    order : [B, TS, U]
    frames : also r_target (spring-centre coordinate) for the barrier check; without it the check is skipped
    analysis_dir : directory holding ``mixture_pmfs.csv``; None skips the barrier check
    temperature : K, for the PMF-peak prominence threshold
    kept_families : k families of the kept features; adds family_composition and corr_W1_W3

    Returns
    -------
    dict
        class_counts, class_frames_per_pull_median (labelled pulls), n_pulls, n_pulls_labelled, excluded,
        ts_{r,z}_{median,iqr,multimodal,bic}, {B,U}_r_median, ts_speed_means, ts_speed_max_smd, v0_barrier;
        with kept_families: family_composition (kept count per family) and corr_W1_W3 ({"W1", "W3": names,
        "r": Pearson matrix W1 x W3 over the labelled frames}).
    """
    ts = labels == order[1]
    pulls = frames.pull.values
    out = {"class_counts": {c: int((labels == c).sum()) for c in order},
           "class_frames_per_pull_median": {
               c: float(np.median([np.sum((pulls == p) & (labels == c)) for p in np.unique(pulls[labels != ""])]))
               for c in order},
           "n_pulls": len(diag), "n_pulls_labelled": int(sum(not d["reason"] for d in diag)),
           "excluded": dict(Counter(d["reason"] for d in diag if d["reason"]))}
    for key, x in (("r", frames.r.values[ts]), ("z", np.asarray(z)[ts])):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        flag, bic = multimodal(x)
        out[f"ts_{key}_median"], out[f"ts_{key}_iqr"] = float(med), [float(q1), float(q3)]
        out[f"ts_{key}_multimodal"], out[f"ts_{key}_bic"] = flag, bic
    for c in (order[0], order[2]):
        out[f"{c}_r_median"] = float(np.median(frames.r.values[labels == c]))
    speed = frames.speed.values
    sps = [float(s) for s in np.unique(speed[ts])]
    means = np.array([Z[ts & (speed == s)].mean(0) for s in sps])
    sd = Z[ts].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out["ts_speed_means"] = {str(s): dict(zip(map(str, kept_names), np.round(m, 4).tolist()))
                             for s, m in zip(sps, means)}
    out["ts_speed_max_smd"] = (float(np.max(np.abs(means[:, None] - means[None]) / sd))
                               if len(sps) > 1 else float("nan"))
    tgt = frames["r_target"].values[ts] if "r_target" in frames else None
    if tgt is not None and not np.isfinite(tgt).all():
        tgt = None
    out["v0_barrier"] = _v0_barrier(tgt, frames.r.values[ts], analysis_dir, temperature)
    if kept_families is not None:
        fam = [str(f) for f in kept_families]
        out["family_composition"] = dict(Counter(fam))
        i1 = [k for k, f in enumerate(fam) if f == "W1"]
        i3 = [k for k, f in enumerate(fam) if f == "W3"]
        r = np.zeros((len(i1), len(i3)))
        if i1 and i3:
            m = np.isin(labels, order)
            r = np.corrcoef(np.asarray(Z)[m][:, i1 + i3].T)[:len(i1), len(i1):]
        out["corr_W1_W3"] = {"W1": [str(kept_names[k]) for k in i1], "W3": [str(kept_names[k]) for k in i3],
                             "r": r.tolist()}
    return out


def _strict(x):
    """x with every non-finite float replaced by None (strict JSON)."""
    if isinstance(x, dict):
        return {k: _strict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict(v) for v in x]
    return None if isinstance(x, float) and not np.isfinite(x) else x


def _fused_cn_stats(feats, w, mu, sd, system):
    """Classes, interaction groups and build wall time (s) of the fused s_w's cn force."""
    from .forces import fused_sw_forces
    t0 = time.perf_counter()
    forces, _ = fused_sw_forces(feats, w, mu, sd, system)
    dt = time.perf_counter() - t0
    cn = next(f for f in forces if f.getName() == "fused_cn")
    return {"n_classes": int(cn.getTabulatedFunction(0).getFunctionParameters()[0]),
            "n_groups": int(cn.getNumInteractionGroups()), "build_s": float(dt)}


def _finish_rupture(shared, Z, z, diag, reducer, funnel, analysis_dir, temperature=300.0):
    """Fit s_w on the standardised kept features; write learned_cv.npz, rupture_report.json and report.png."""
    sh = shared
    names, kept, frames, labels, order = sh["names"], sh["kept"], sh["frames"], sh["labels"], sh["order"]
    fit = fit_sw(Z, z, labels, order, reducer=reducer)
    s = fit["X"] @ fit["w"]
    m = np.isin(labels, order)
    # training B frames under-sample the bound basin, so widen the range by SW_BOUND_SIGMAS sigma_B
    sig_b = float(s[labels == order[0]].std())
    sw_lo = float(s[m].min()) - SW_BOUND_SIGMAS * sig_b
    sw_hi = float(s[m].max()) + SW_BOUND_SIGMAS * sig_b
    hw = 0.5 * sig_b
    if not np.isfinite(hw) or hw <= 1e-9 * (sw_hi - sw_lo):
        raise ValueError(f"s_w hill width is {hw} (s_w is constant or undefined over the {order[0]} frames)")
    Wl = lda_reference(fit["X"][m], labels[m], 2, order)
    kl, _ = choose_component(fit["X"] @ Wl, labels, order)
    acc = _grouped_accuracy(fit["X"], labels, frames.pull.values, order)
    rep = rupture_report(frames, labels, z, Z, names[kept], diag, order, analysis_dir=analysis_dir,
                         temperature=temperature, kept_families=[sh["fams"][i] for i in kept])
    ex = sh["extra"]
    if sh["feature_set"] == "water":
        rep.update(water=ex.get("water"), safe_radius=ex["safe_radius"], extraction={"s_per_frame": ex["s_per_frame"]})
    kept_feats = [sh["feats"][i] for i in kept]
    if sh["system"] is not None and any(f.kind == "cn" for f in kept_feats):
        rep["fused_cn"] = _fused_cn_stats(kept_feats, fit["w"], sh["mu"], sh["sd"], sh["system"])
        logger.info("learned CV (rupture): fused cn %d classes, %d interaction groups, built in %.2f s",
                    rep["fused_cn"]["n_classes"], rep["fused_cn"]["n_groups"], rep["fused_cn"]["build_s"])
    rep.update(kept=[str(n) for n in names[kept]], component=int(fit["component"]),
               corr_sz=float(fit["corr_plain"]), corr_sz_resid=float(fit["corr_resid"]),
               residualised=bool(fit["residualised"]), cv_acc=acc, sw_lo=sw_lo, sw_hi=sw_hi, sw_hill_width=hw)
    with open(os.path.join(sh["out_dir"], RUPTURE_REPORT_NAME), "w") as fh:
        json.dump(_strict(rep), fh, indent=1, allow_nan=False)
    basin = {c: [float(np.median(z[labels == c])), float(np.median(s[labels == c]))] for c in (order[0], order[2])}
    out = _write_learned(sh, s[:, None], fit["w"][:, None], Wl[:, [kl]], dict(
        weights=fit["weights"], lda_weights=Wl, component=fit["component"], sw_weights=fit["w"],
        fisher_ts=fit["fisher_ts"], plain_weights=fit["plain_weights"], plain_component=fit["plain_component"],
        corr_sz=fit["corr_plain"], corr_sz_resid=fit["corr_resid"], residualised=fit["residualised"],
        z_min=fit["z_min"], z_max=fit["z_max"], z_tables=fit["tables"],
        funnel_host=np.array(funnel["host"]), funnel_guest=np.array(funnel["guest"]),
        funnel_axis=np.asarray(funnel["axis"], float),
        **({} if funnel["frame"] is None else dict(
            funnel_orient_index=np.array(funnel["frame"]["orient"]),
            funnel_orient_offset=np.asarray(funnel["frame"]["offset"], float),
            funnel_orient_ref=np.asarray(funnel["frame"]["ref"], float)) if "orient" in funnel["frame"] else dict(
            funnel_frame_b=np.array(funnel["frame"]["b"]), funnel_frame_c=np.array(funnel["frame"]["c"]),
            funnel_axis_coeffs=np.asarray(funnel["frame"]["coeffs"], float))),
        sw_lo=sw_lo, sw_hi=sw_hi, sw_grid_min=sw_lo - 3 * hw, sw_grid_max=sw_hi + 3 * hw, sw_hill_width=hw,
        basin_B=np.array(basin[order[0]]), basin_U=np.array(basin[order[2]]), cv_acc=acc,
        n_pulls_labelled=rep["n_pulls_labelled"], class_order=np.array(order)))
    logger.info("learned CV (rupture): classes %s; %d/%d pulls labelled; excluded %s",
                rep["class_counts"], rep["n_pulls_labelled"], rep["n_pulls"], rep["excluded"])
    logger.info("learned CV (rupture): TS r median %.2f nm (IQR %.2f-%.2f, multimodal %s), TS z median %.2f nm "
                "(multimodal %s); per-speed TS max SMD %.2f", rep["ts_r_median"], *rep["ts_r_iqr"],
                rep["ts_r_multimodal"], rep["ts_z_median"], rep["ts_z_multimodal"], rep["ts_speed_max_smd"])
    v0 = rep["v0_barrier"]
    logger.info("learned CV (rupture): v->0 barrier check %s", v0["estimators"] if v0["status"] == "ok"
                else f"skipped ({v0['reason']})")
    logger.info("learned CV (rupture): %d kept (%s); s_w = HLDA component %d; corr(s_w, z) %.3f%s",
                len(kept), ", ".join(names[kept]), fit["component"] + 1, fit["corr_plain"],
                f" -> z-residualised, corr {fit['corr_resid']:.3f}" if fit["residualised"] else "")
    return out


def _report(path, frames, labels, cv, kept_names, Wt, Wl, order, cv_tab=None, res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.style.context("default"):
        fig, ax = plt.subplots(1, 5, figsize=(26, 4.8), constrained_layout=True)
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
        if cv_tab is not None and len(cv_tab):
            ax[4].errorbar(cv_tab.resolution, cv_tab["mean"], cv_tab["sd"], marker="o", capsize=3)
            ax[4].axvline(res, c="k", ls=":")
            ax[4].set_xlabel("MoSAIC resolution")
            ax[4].set_ylabel("held-out silhouette (pull-grouped CV)")
        else:
            ax[4].text(0.5, 0.5, f"MoSAIC resolution {res} (fixed)", ha="center", transform=ax[4].transAxes)
            ax[4].set_axis_off()
        fig.savefig(path, dpi=130)
        plt.close(fig)
