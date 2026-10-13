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
    kind: Optional[str] = None


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

    def reweighted(self, walkers=None):
        """Walkers with REWEIGHT data (of ``walkers``, default all)."""
        return [w for w in (self.walkers if walkers is None else walkers) if w.bias is not None]


class NoRecords(Exception):
    """A segment with no hills or fewer than two usable rows (freshly started)."""


def _npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def _names(h):
    return [str(x) for x in np.atleast_1d(h["cv_names"])]


def _segment(run_dir, rid):
    """One segment's arrays, cut to the common length minus the last (possibly partial) row.

    Raises
    ------
    NoRecords
        No hills yet, or fewer than two usable rows.
    """
    h = _npz(os.path.join(run_dir, f"HILLS_{rid}.npz"))
    if int(h.get("hills_version", 0)) < HILLS_VERSION:
        raise ValueError(f"HILLS_{rid} is not v{HILLS_VERSION}")
    names = _names(h)
    ht = np.atleast_1d(np.asarray(h["time_ns"], float))
    if len(ht) == 0:
        raise NoRecords("no hills")
    hc = np.asarray(h["cv"], float).reshape(len(ht), len(names))
    col = np.load(os.path.join(run_dir, f"COLVAR_{rid}.npy"))
    col = col.reshape(len(col), -1)
    rw_path = os.path.join(run_dir, f"REWEIGHT_{rid}.npz")
    rw = _npz(rw_path) if os.path.exists(rw_path) else None
    n = len(col) if rw is None else min(len(col), len(np.atleast_1d(rw["bias_kj"])))
    n = max(n - 1, 0)
    if n < 2:
        raise NoRecords(f"{n} usable rows")
    cvs = {nm: col[:n, j] for j, nm in enumerate(names)}
    mon = {}
    if rw is not None:
        mon = {k[len("monitor_"):]: np.asarray(v)[:n] for k, v in rw.items() if k.startswith("monitor_")}
    dt = float(rw["record_dt_ns"]) if rw is not None else float(h["record_dt_ns"])
    t = np.arange(n) * dt
    frames = dcd_frame_of_rows(n, dt, float(h["dcd_interval_ns"]))
    return dict(h=h, t=t, dt=dt, cvs=cvs, mon=mon, hill_t=ht, hill_cv=hc,
                bias=None if rw is None else np.asarray(rw["bias_kj"])[:n],
                ct=None if rw is None else np.asarray(rw["rct_kj"])[:n], frames=frames)


def _layout(walkers):
    if len(walkers) == 1:
        return "single"
    n = len({w.bias_dir for w in walkers})
    if n == 1:
        return "shared"
    if n == len(walkers):
        return "isolated"
    raise ValueError("mixed shared/private bias dirs: not supported")


_SAME = ("cv_names", "grid_min", "grid_max", "grid_points", "periodic", "tempering")


def _check_same(ref, h, rid):
    """Raise if a segment's CVs, grid or tempering differ from the first used segment."""
    for k in _SAME:
        a, b = np.atleast_1d(ref[k]), np.atleast_1d(h[k])
        same = (a.shape == b.shape and (np.array_equal(a, b) if a.dtype.kind in "USbO" or b.dtype.kind in "USbO"
                                        else np.allclose(a, b, rtol=1e-9, atol=0.0)))
        if not same:
            raise ValueError(f"{rid}: {k} differs from the other walkers: not supported")


def _resolve(given, from_meta, loader, key, skipped):
    """Explicit path/dict (errors raise) or the HILLS-metadata path (errors are recorded and give None)."""
    if given is not None:
        return loader(os.fspath(given)) if isinstance(given, (str, os.PathLike)) else given
    if not from_meta:
        return None
    path = str(from_meta)
    try:
        return loader(path)
    except Exception as e:
        reason = "not found" if isinstance(e, FileNotFoundError) else f"unreadable ({e})"
        skipped.append(f"{key} {path}: {reason}")
        logger.warning("summary: %s %s: %s", key, path, reason)
        return None


def _skip(skipped, msg):
    skipped.append(msg)
    logger.warning("summary: skipping %s", msg)


def _load_parts(run_dir, base, segs, skipped):
    """Readable consecutive segments of one walker; later ones after a failure or gap are listed in ``skipped``."""
    segs = sorted(segs)
    parts = []
    for order, (k, rid) in enumerate(segs):
        if k != order + 1:
            _skip(skipped, f"{rid}: segment {order + 1} of {base} missing" if order else
                  f"{rid}: base segment {base} missing")
            skipped += [f"{r}: not loaded, earlier segment missing" for _, r in segs[order + 1:]]
            break
        try:
            parts.append((rid, _segment(run_dir, rid)))
        except NoRecords as e:
            _skip(skipped, f"{rid}: no records yet ({e})")
            skipped += [f"{r}: not loaded, earlier segment has no records" for _, r in segs[order + 1:]]
            break
        except Exception as e:                      # unreadable / truncated segment
            _skip(skipped, f"{rid}: {e}")
            skipped += [f"{r}: not loaded, earlier segment failed" for _, r in segs[order + 1:]]
            break
    return parts


def _join(run_dir, base, parts, skipped):
    """One Walker from consecutive segments; each segment starts where the previous one's kept rows end."""
    off, t, bias, ct, frames, segi, ht, hc, hh, starts = 0.0, [], [], [], [], [], [], [], [], []
    cvs = {}
    for k, (rid, p) in enumerate(parts):
        if k:
            starts.append(off)
        t.append(p["t"] + off); frames.append(p["frames"]); segi.append(np.full(len(p["t"]), k))
        for nm, v in {**p["mon"], **p["cvs"]}.items():
            cvs.setdefault(nm, []).append(v)
        if p["bias"] is not None:
            bias.append(p["bias"]); ct.append(p["ct"])
        ht.append(p["hill_t"] + off); hc.append(p["hill_cv"]); hh.append(np.asarray(p["h"]["height_kj"], float))
        off += len(p["t"]) * p["dt"]
    for nm, v in cvs.items():
        if len(v) != len(parts):
            _skip(skipped, f"{base}: monitor {nm} missing in {len(parts) - len(v)} segment(s), dropped")
    has_rw = len(bias) == len(parts)
    if not has_rw:
        _skip(skipped, f"{base}: no REWEIGHT data in {len(parts) - len(bias)} segment(s), "
                       "excluded from reweighted estimates")
    return Walker(
        rid=base, t=np.concatenate(t), cvs={k: np.concatenate(v) for k, v in cvs.items() if len(v) == len(parts)},
        bias=np.concatenate(bias) if has_rw else None, ct=np.concatenate(ct) if has_rw else None,
        hills_raw=[p["h"] for _, p in parts], hill_t=np.concatenate(ht), hill_cv=np.concatenate(hc),
        hill_h=np.concatenate(hh), segments=starts, finished=bool(parts[-1][1]["h"].get("finished", False)),
        bias_dir=str(parts[0][1]["h"]["bias_dir"]), record_frame=np.concatenate(frames),
        segment_of_record=np.concatenate(segi),
        dcd_files=[os.path.join(run_dir, f"WTMetaD_{rid}.dcd") for rid, _ in parts])


def load_run(run_dir, funnel=None, learned_cv=None) -> RunData:
    """Per-walker data of a metadynamics run directory; extension segments ``<rid>_x2, ...`` are appended in time.

    Mid-run safe: a segment with no hills or fewer than two usable rows, an unreadable segment, and an
    extension without its earlier segments are skipped and listed in ``RunData.skipped``. Metadata come from
    the first used segment; CV names, grids and tempering must agree across all used segments.

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

    Raises
    ------
    ValueError
        No HILLS files; every walker without usable records ("no usable records yet: not supported");
        mismatched CVs/grids/tempering or mixed shared/private bias dirs (not supported).
    """
    files = sorted(glob.glob(os.path.join(run_dir, "HILLS_*.npz")))
    if not files:
        raise ValueError(f"no HILLS files in {run_dir}")
    groups = {}
    for f in files:
        rid = os.path.basename(f)[len("HILLS_"):-4]
        m = _SEG.match(rid)
        groups.setdefault(m["base"], []).append((int(m["k"] or 1), rid))
    walkers, skipped, meta = [], [], None
    for base, segs in sorted(groups.items()):
        parts = _load_parts(run_dir, base, segs, skipped)
        if not parts:
            continue
        if meta is None:
            meta = parts[0][1]["h"]
        for rid, p in parts:
            _check_same(meta, p["h"], rid)
        walkers.append(_join(run_dir, base, parts, skipped))
    if not walkers:
        raise ValueError(f"no usable records yet: not supported ({'; '.join(skipped)})")
    names = _names(meta)
    lo, hi = np.atleast_1d(meta["bounds_lo"]), np.atleast_1d(meta["bounds_hi"])
    kinds = np.atleast_1d(meta["cv_types"]) if "cv_types" in meta else [None] * len(names)
    specs = [CVInfo(name=nm, grid_min=float(np.atleast_1d(meta["grid_min"])[j]),
                    grid_max=float(np.atleast_1d(meta["grid_max"])[j]),
                    periodic=bool(np.atleast_1d(meta["periodic"])[j]),
                    bounds=None if np.isnan(lo[j]) else (float(lo[j]), float(hi[j])),
                    sigma=float(np.atleast_1d(meta["sigma"])[j]),
                    ang_per_unit=None if np.isnan(np.atleast_1d(meta["cv_length_per_unit_ang"])[j])
                    else float(np.atleast_1d(meta["cv_length_per_unit_ang"])[j]),
                    kind=None if kinds[j] is None else str(kinds[j]))
             for j, nm in enumerate(names)]
    logged = [set(w.cvs) - set(names) for w in walkers]
    monitored = sorted(set.intersection(*logged))
    for nm in sorted(set.union(*logged) - set(monitored)):
        _skip(skipped, f"monitor {nm}: not logged for every walker, dropped")
    funnel = _resolve(funnel, meta.get("funnel_file"), load_funnel_params, "funnel_file", skipped)
    lcv = _resolve(learned_cv, meta.get("learned_cv_file"), _npz, "learned_cv_file", skipped)
    return RunData(run_dir=os.path.abspath(run_dir), walkers=walkers, cvs=specs, monitored=monitored,
                   layout=_layout(walkers), tempering=str(meta["tempering"]), temperature=float(meta["temperature"]),
                   bias_factor=float(meta["bias_factor"]), funnel=funnel, learned_cv=lcv, skipped=skipped,
                   has_reweight=any(w.bias is not None for w in walkers))
