"""Load a metadynamics run directory (HILLS v2, REWEIGHT, COLVAR) into per-walker arrays."""
import glob
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..Analysis import dcd_frame_of_rows
from ..Funnel import load_funnel_params
from ..Hills import HILLS_VERSION

logger = logging.getLogger("autopath")
_SEG = re.compile(r"^(?P<base>.+?)(?:_x(?P<k>\d+))?$")


@dataclass
class CVInfo:
    name: str
    grid_min: float
    grid_max: float
    periodic: bool
    bounds: Optional[tuple]
    sigma: float
    ang_per_unit: Optional[float]


@dataclass
class Walker:
    rid: str
    t: np.ndarray
    cvs: dict
    bias: Optional[np.ndarray]
    ct: Optional[np.ndarray]
    hills_raw: list
    hill_t: np.ndarray
    hill_cv: np.ndarray
    hill_h: np.ndarray
    segments: list
    finished: bool
    bias_dir: str
    record_frame: np.ndarray
    segment_of_record: np.ndarray
    dcd_files: list


@dataclass
class RunData:
    run_dir: str
    walkers: list
    cvs: list
    monitored: list
    layout: str
    tempering: str
    temperature: float
    bias_factor: float
    funnel: Optional[dict]
    learned_cv: Optional[dict]
    skipped: list = field(default_factory=list)
    has_reweight: bool = True


def _npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def _segment(run_dir, rid, names):
    """One segment's arrays, cut to the common length minus the last (possibly partial) row."""
    h = _npz(os.path.join(run_dir, f"HILLS_{rid}.npz"))
    if int(h.get("hills_version", 0)) < HILLS_VERSION:
        raise ValueError(f"HILLS_{rid} is not v{HILLS_VERSION}")
    col = np.load(os.path.join(run_dir, f"COLVAR_{rid}.npy"))
    col = col.reshape(len(col), -1)
    rw_path = os.path.join(run_dir, f"REWEIGHT_{rid}.npz")
    rw = _npz(rw_path) if os.path.exists(rw_path) else None
    n = len(col) if rw is None else min(len(col), len(rw["bias_kj"]))
    n = max(n - 1, 0)
    cvs = {nm: col[:n, j] for j, nm in enumerate(names)}
    mon = {}
    if rw is not None:
        mon = {k[len("monitor_"):]: np.asarray(v)[:n] for k, v in rw.items() if k.startswith("monitor_")}
    dt = float(rw["record_dt_ns"]) if rw is not None else float(h["record_dt_ns"])
    t = np.arange(n) * dt
    frames = dcd_frame_of_rows(n, dt, float(h["dcd_interval_ns"]))
    return dict(h=h, t=t, cvs=cvs, mon=mon, bias=None if rw is None else np.asarray(rw["bias_kj"])[:n],
                ct=None if rw is None else np.asarray(rw["rct_kj"])[:n], frames=frames)


def _layout(walkers):
    if len(walkers) == 1:
        return "single"
    return "shared" if len({w.bias_dir for w in walkers}) == 1 else "isolated"


def load_run(run_dir, funnel=None, learned_cv=None) -> RunData:
    """Per-walker data of a metadynamics run directory; extension segments ``<rid>_x2, ...`` are appended in time.

    Parameters
    ----------
    run_dir : str
    funnel : str or dict, optional
        funnel_params ``.npz`` path or loaded dict; default from HILLS metadata ``funnel_file`` when present.
    learned_cv : str, optional
        learned_cv ``.npz`` path; default from HILLS metadata ``learned_cv_file`` when present.

    Returns
    -------
    RunData
    """
    files = sorted(glob.glob(os.path.join(run_dir, "HILLS_*.npz")))
    if not files:
        raise ValueError(f"no HILLS files in {run_dir}")
    groups = {}
    for f in files:
        rid = os.path.basename(f)[len("HILLS_"):-4]
        m = _SEG.match(rid)
        groups.setdefault(m["base"], []).append((int(m["k"] or 1), rid))
    walkers, skipped, specs, meta = [], [], None, None
    for base, segs in sorted(groups.items()):
        parts = []
        for _, rid in sorted(segs):
            try:
                if meta is None:
                    meta = _npz(os.path.join(run_dir, f"HILLS_{rid}.npz"))
                names = [str(x) for x in np.atleast_1d(meta["cv_names"])]
                parts.append((rid, _segment(run_dir, rid, names)))
            except Exception as e:                      # unreadable / truncated segment
                skipped.append(f"{rid}: {e}")
                logger.warning("summary: skipping %s (%s)", rid, e)
                break
        if not parts:
            continue
        off, t, bias, ct, frames, segi, hills, ht, hc, hh, starts = 0.0, [], [], [], [], [], [], [], [], [], []
        cvs = {}
        for k, (rid, p) in enumerate(parts):
            if k:
                starts.append(off)
            t.append(p["t"] + off); frames.append(p["frames"]); segi.append(np.full(len(p["t"]), k))
            for nm, v in {**p["mon"], **p["cvs"]}.items():
                cvs.setdefault(nm, []).append(v)
            if p["bias"] is not None:
                bias.append(p["bias"]); ct.append(p["ct"])
            h = p["h"]; hills.append(h)
            ht.append(np.asarray(h["time_ns"]) + off); hc.append(np.asarray(h["cv"]).reshape(len(h["time_ns"]), -1))
            hh.append(np.asarray(h["height_kj"]))
            off += float(p["t"][-1] + (p["t"][1] - p["t"][0])) if len(p["t"]) > 1 else 0.0
        has_rw = len(bias) == len(parts)
        walkers.append(Walker(
            rid=base, t=np.concatenate(t), cvs={k: np.concatenate(v) for k, v in cvs.items() if len(v) == len(parts)},
            bias=np.concatenate(bias) if has_rw else None, ct=np.concatenate(ct) if has_rw else None,
            hills_raw=hills, hill_t=np.concatenate(ht), hill_cv=np.concatenate(hc), hill_h=np.concatenate(hh),
            segments=starts, finished=bool(parts[-1][1]["h"].get("finished", False)),
            bias_dir=str(parts[0][1]["h"]["bias_dir"]), record_frame=np.concatenate(frames),
            segment_of_record=np.concatenate(segi),
            dcd_files=[os.path.join(run_dir, f"WTMetaD_{rid}.dcd") for rid, _ in parts]))
    if not walkers:
        raise ValueError(f"no readable walkers in {run_dir}: {skipped}")
    names = [str(x) for x in np.atleast_1d(meta["cv_names"])]
    lo, hi = np.atleast_1d(meta["bounds_lo"]), np.atleast_1d(meta["bounds_hi"])
    specs = [CVInfo(name=nm, grid_min=float(np.atleast_1d(meta["grid_min"])[j]),
                    grid_max=float(np.atleast_1d(meta["grid_max"])[j]),
                    periodic=bool(np.atleast_1d(meta["periodic"])[j]),
                    bounds=None if np.isnan(lo[j]) else (float(lo[j]), float(hi[j])),
                    sigma=float(np.atleast_1d(meta["sigma"])[j]),
                    ang_per_unit=None if np.isnan(np.atleast_1d(meta["cv_length_per_unit_ang"])[j])
                    else float(np.atleast_1d(meta["cv_length_per_unit_ang"])[j]))
             for j, nm in enumerate(names)]
    monitored = sorted(set(walkers[0].cvs) - set(names))
    if funnel is None and meta.get("funnel_file"):
        funnel = str(meta["funnel_file"])
    if isinstance(funnel, str):
        funnel = load_funnel_params(funnel)
    if learned_cv is None and meta.get("learned_cv_file"):
        learned_cv = str(meta["learned_cv_file"])
    lcv = _npz(learned_cv) if isinstance(learned_cv, str) else learned_cv
    return RunData(run_dir=os.path.abspath(run_dir), walkers=walkers, cvs=specs, monitored=monitored,
                   layout=_layout(walkers), tempering=str(meta["tempering"]), temperature=float(meta["temperature"]),
                   bias_factor=float(meta["bias_factor"]), funnel=funnel, learned_cv=lcv, skipped=skipped,
                   has_reweight=all(w.bias is not None for w in walkers))
