"""Per-pull table and per-step traces written by the sMD analysis."""
import logging
import os

import numpy as np
import pandas as pd

from autopath.pulling.SMDData import SMDData

logger = logging.getLogger(__name__)

PULL_COLUMNS = ["speed", "repid", "realized_speed", "path", "path_ok"]
_TRAILER_NUM = ["final_displacement_nm", "n_moves", "final_move", "wall_minutes"]


def _pull_order(smd: SMDData, names: dict) -> dict:
    """{trajname: 1-based rank}, from one ordering source for the whole set."""
    fns = list(names.values())
    starts = {n: smd.log_metadata[n].get("start_time") for n in names}
    if all(isinstance(s, str) for s in starts.values()):
        key, src = starts, "start_time headers"
    else:
        try:
            key = {n: SMDData._replica_start_datetime(fn) for n, fn in names.items()}
            src = "file times"
        except ValueError:
            key = {n: os.path.getmtime(fn) for n, fn in names.items()}
            src = "file mtimes"
    logger.info(f"pull order from {src}")
    ranked = sorted(names, key=lambda n: (key[n], n))
    return {n: i + 1 for i, n in enumerate(ranked)}


def pull_table(smd: SMDData, raw_data: pd.DataFrame) -> pd.DataFrame:
    """One row per loaded sMD log, with stop bookkeeping and pathway.

    Parameters
    ----------
    smd : SMDData
        Loaded pulls.
    raw_data : pandas.DataFrame
        Per-step traces; supplies ``path`` and ``path_ok``.

    Returns
    -------
    pandas.DataFrame
        Columns: trajname, speed, realized_speed, repid, pull_index,
        start_time, stop_reason, final_displacement_nm, n_moves, final_move,
        wall_minutes, path, path_ok.
    """
    names = {os.path.basename(fn)[:-4]: fn for fn in smd.log_files}
    names = {n: fn for n, fn in names.items() if n in smd.log_metadata}
    order = _pull_order(smd, names) if names else {}
    pathinfo = {}
    if len(raw_data) and "path" in raw_data:
        cols = [c for c in ("path", "path_ok") if c in raw_data]
        pathinfo = raw_data.groupby("trajname")[cols].first().to_dict("index")
    rows = []
    for n, fn in names.items():
        meta, tr = smd.log_metadata[n], SMDData.parse_log_trailer(fn)
        info = pathinfo.get(n, {})
        speed = SMDData._speed_from_log(fn)
        row = dict(
            trajname=n,
            speed=speed,
            realized_speed=float(meta.get("realized_speed_nm_per_ps", speed)),
            repid=SMDData._replica_idx_from_log(fn),
            pull_index=order[n],
            start_time=meta.get("start_time", np.nan),
            stop_reason=str(tr.get("stop_reason", "unknown")),
        )
        row.update({c: float(tr.get(c, np.nan)) for c in _TRAILER_NUM})
        row.update(path=info.get("path", np.nan), path_ok=info.get("path_ok", np.nan))
        rows.append(row)
    cols = ["trajname", "speed", "realized_speed", "repid", "pull_index", "start_time",
            "stop_reason", *_TRAILER_NUM, "path", "path_ok"]
    return pd.DataFrame(rows, columns=cols)


def write_traces(raw_data: pd.DataFrame, pulls: pd.DataFrame, outdir: str, legacy: bool = False) -> None:
    """Write ``pulls.csv`` and ``traces.csv.gz`` (plus the legacy single CSV if asked)."""
    pulls.to_csv(f"{outdir}/pulls.csv", index=False)
    raw_data.drop(columns=PULL_COLUMNS, errors="ignore").to_csv(
        f"{outdir}/traces.csv.gz", index=False, compression="gzip")
    if legacy:
        raw_data.to_csv(f"{outdir}/sMD_processed_data.csv", index=False)


def load_traces(outdir: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Per-step traces with the per-pull columns merged back in.

    Falls back to ``sMD_processed_data.csv`` for older analysis dirs.
    """
    tr, pu, old = (f"{outdir}/{f}" for f in ("traces.csv.gz", "pulls.csv", "sMD_processed_data.csv"))
    if os.path.isfile(tr) and os.path.isfile(pu):
        df = pd.read_csv(tr)
        p = pd.read_csv(pu)
        p = p[["trajname", *[c for c in PULL_COLUMNS if c in p]]]
        df = df.merge(p, on="trajname", how="left", validate="many_to_one")
        if "path_ok" in df and df["path_ok"].notna().all():
            df["path_ok"] = df["path_ok"].astype(bool)
    elif os.path.isfile(old):
        df = pd.read_csv(old)
    else:
        raise FileNotFoundError(f"no traces.csv.gz or sMD_processed_data.csv in {outdir}")
    return df if columns is None else df[columns]
