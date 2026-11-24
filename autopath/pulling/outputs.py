"""Per-pull table and per-step traces written by the sMD analysis."""
import json
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


KB_KJ = 0.008314462618
_META_KEYS = ["autopath_version", "system", "ligand", "temperature_K", "spring_k_kJ_mol_nm2",
              "path_model", "n_paths", "analysis_warnings"]


def jarzynski_convergence(traces: pd.DataFrame, pulls: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Jarzynski estimator health versus pull count, at the full-set best step.

    Parameters
    ----------
    traces : pandas.DataFrame
        Per-step ``trajname``, ``speed``, ``step``, ``work``, ``r_coord``.
    pulls : pandas.DataFrame
        ``trajname`` and ``pull_index`` (acquisition order).
    temperature : float
        Kelvin.

    Returns
    -------
    pandas.DataFrame
        One row per (speed, n >= 2): ``top_pull_weight``, ``ess_over_n``,
        ``tail_position``, ``beta_sigma_W`` and ``r_nm`` (the step used).
    """
    from scipy.special import logsumexp

    kT = KB_KJ * temperature
    idx = pulls.set_index("trajname")["pull_index"]
    rows = []
    for sp, g in traces.groupby("speed"):
        W = g.pivot_table(index="step", columns="trajname", values="work").dropna()
        N = W.shape[1]
        if N < 2 or W.empty:
            continue
        W = W[sorted(W.columns, key=lambda t: idx[t])]
        r = g.groupby("step")["r_coord"].median().reindex(W.index).to_numpy()
        w = W.to_numpy()
        i = int(np.argmax(-kT * (logsumexp(-w / kT, axis=1) - np.log(N))))
        for n in range(2, N + 1):
            x = w[i, :n]
            lw = -x / kT
            p = np.exp(lw - logsumexp(lw))
            dG = -kT * (logsumexp(lw) - np.log(n))
            rows.append(dict(speed=sp, n=n, r_nm=r[i], top_pull_weight=p.max(),
                             ess_over_n=1 / (p ** 2).sum() / n,
                             tail_position=(dG - x.min()) / (kT * np.log(n)),
                             beta_sigma_W=x.std(ddof=1) / kT))
    return pd.DataFrame(rows, columns=["speed", "n", "r_nm", "top_pull_weight", "ess_over_n",
                                       "tail_position", "beta_sigma_W"])


def _num(x, nd):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _read(outdir: str, name: str):
    f = f"{outdir}/{name}"
    return pd.read_csv(f) if os.path.isfile(f) else None


def write_summary(outdir: str, meta: dict | None = None) -> dict:
    """Write ``summary.json`` and return it.

    Parameters
    ----------
    outdir : str
        Analysis directory.
    meta : dict, optional
        Run-level keys (version, system, ligand, temperature, spring constant,
        path model, path count, warnings). ``None`` reloads them from an
        existing ``summary.json`` and refreshes only the disk-derived keys.
    """
    from autopath.pulling.readouts import profile_readouts

    path = f"{outdir}/summary.json"
    if meta is None:
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            meta = {}
    out = {k: meta.get(k) for k in _META_KEYS}
    T = out["temperature_K"] or 300.0
    kT = KB_KJ * T
    pulls, vall = _read(outdir, "pulls.csv"), _read(outdir, "sMD_conv_vALL_metrics.csv")
    pq, kr, pmfs = (_read(outdir, f) for f in ("path_quality.csv", "koff_kramers.csv", "mixture_pmfs.csv"))

    speeds = {}
    if pulls is not None:
        for sp, g in pulls.groupby("speed"):
            conv = None
            if vall is not None and {"speed", "n_replicas", "converged"} <= set(vall.columns):
                c = vall[np.isclose(vall["speed"], sp)]
                if len(c):
                    v = c.loc[c["n_replicas"].idxmax(), "converged"]
                    conv = None if pd.isna(v) else bool(v)
            pw = {}
            if pq is not None and {"estimator", "speed", "path", "weight"} <= set(pq.columns):
                c = pq[np.isclose(pq["speed"], sp)]
                if len(c):
                    c = c[c["estimator"] == c["estimator"].iloc[0]]
                    pw = {str(k): _num(v, 3) for k, v in zip(c["path"], c["weight"])}
            speeds[f"{sp:g}"] = dict(pulls=int(len(g)), stop=g["stop_reason"].value_counts().to_dict(),
                                     converged=conv, path_weights=pw)
    out["speeds"] = speeds

    ro = {}
    if pmfs is not None and len(pmfs):
        for r in profile_readouts(pmfs, kBT=kT).itertuples():
            ro[(r.estimator, float(r.speed))] = dict(dG_kJ_mol=r.dG_profile)
    if kr is not None:
        for r in kr.itertuples():
            d = ro.setdefault((r.estimator, float(r.speed_nm_per_ps)), {})
            d["pkoff"] = -np.log10(r.k_off_per_s) if r.k_off_per_s > 0 else None
            d["barrier_kJ_mol"] = r.barrier_kjmol
    out["readouts"] = [dict(estimator=e, speed=sp, dG_kJ_mol=_num(d.get("dG_kJ_mol"), 1),
                            pkoff=_num(d.get("pkoff"), 2), barrier_kJ_mol=_num(d.get("barrier_kJ_mol"), 1))
                       for (e, sp), d in sorted(ro.items())]

    if pulls is not None:
        try:
            jc = jarzynski_convergence(load_traces(outdir, ["trajname", "speed", "step", "work", "r_coord"]),
                                       pulls, T)
        except (FileNotFoundError, KeyError):
            jc = pd.DataFrame()
        if len(jc):
            last = jc.sort_values("n").groupby("speed").tail(1)
            out["jarzynski_diagnostics"] = {
                f"{r.speed:g}": {k: _num(getattr(r, k), 3) for k in
                                 ("r_nm", "top_pull_weight", "ess_over_n", "beta_sigma_W")}
                for r in last.itertuples()}
    for k, nd in (("temperature_K", 1), ("spring_k_kJ_mol_nm2", 1)):
        out[k] = _num(out[k], nd)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    return out
