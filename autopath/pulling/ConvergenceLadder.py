"""Multi-speed convergence ladder for the force free-energy estimator.

``SMDAnalysis.check_convergence`` grows one pulling speed's replica count at a
time and compares each PMF against a windowed reference over the preceding
rungs. The ``force`` estimator instead needs at
least two speeds simultaneously, because its free energy comes from
extrapolating per-speed weighted PMFs to v->0 (``extrapolate_to_v0``); there
is no single-speed "force PMF" to grow a per-speed loop over.

This module implements that alternative growth axis: at rung k, every speed
contributes its first k replicas, each speed's mixture PMF is rebuilt from
running estimator statistics, the per-speed PMFs are extrapolated to v->0,
and the extrapolated PMF is compared against a reference averaged over the
last ``conv_window`` rungs (``conv_window=1`` reduces to the previous rung)
with deployment's three convergence criteria (RMSD, barrier-height delta,
TS-position delta) via the exact same comparison arithmetic
``check_convergence`` uses.

Promoted from ``scratch/paper_figures/wdr5_conv_lib.py`` (validated against
the deployed API by that scratch package's test suite); WDR5-specific
constants (speed list, system/mode lookups, data paths) have been replaced by
parameters / attributes of the caller-supplied ``SMDAnalysis`` instance.
"""
import inspect
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from autopath.pulling.AnalysisSMD import SMDAnalysis
from autopath.pulling.Convergence import window_mean_pmf, window_reference_scalar
from autopath.pulling.Estimators import KramersEstimator, extrapolate_to_v0
from autopath.pulling.PathModel import DTWPathModel
from autopath.pulling.SMDData import SMDData
from autopath.pulling.support import SupportPolicy


def _ladder_defaults() -> dict:
    """Introspect ``check_convergence``'s own defaults as the single source
    of truth for values the ladder cannot receive as explicit parameters
    (tolerances, plateau fraction, boundary buffer, per-rung support policy).

    ``check_convergence``'s tolerance/boundary defaults are local arguments
    (not module-level constants) and therefore not otherwise importable; its
    own docstring/comments state that any deployment override must match
    these defaults, so the signature is the usable single source of truth.
    """
    sig = inspect.signature(SMDAnalysis.check_convergence)
    return {
        "tol_rmsd": float(sig.parameters["tol_rmsd"].default),
        "tol_barrier": float(sig.parameters["tol_barrier"].default),
        "tol_r_ts": float(sig.parameters["tol_r_ts"].default),
        "plateau_frac": float(sig.parameters["plateau_frac"].default),
        "boundary_buffer_frac": float(sig.parameters["boundary_buffer_frac"].default),
        "min_samples_per_step_conv": int(sig.parameters["min_samples_per_step_conv"].default),
        "min_trajs_per_path_conv": int(sig.parameters["min_trajs_per_path_conv"].default),
    }


def _speed_state(sa, smd, speed, plateau_frac, boundary_buffer_frac, trim_fraction=0.1):
    """Per-speed setup mirroring check_convergence: boundary, clustering.

    Returns
    -------
    (protocol_grid, boundary_cap, speed_data)
        ``protocol_grid`` : Series step -> r_target_protocol
        ``boundary_cap``  : float | None, force-plateau boundary + buffer
        ``speed_data``    : raw_data rows for this speed, with 'path' assigned
    """
    protocol_grid = smd.protocol_grids[speed].set_index("step")["r_target_protocol"]

    boundary_cap = None
    rd = smd.raw_data[smd.raw_data["speed"] == speed]
    if {"r_coord", "force", "speed"}.issubset(rd.columns):
        r_lo, r_hi = float(rd["r_coord"].min()), float(rd["r_coord"].max())
        r_ts = KramersEstimator.force_plateau_boundary(
            rd, speed, r_lo, r_hi, plateau_frac)
        if r_ts is not None:
            boundary_cap = min(r_ts * (1.0 + boundary_buffer_frac), r_hi)

    # group_A/group_B are None: this route (geom_features=True,
    # merge_features=False) clusters on trace + inline-geom features only, so
    # the pocket-ligand distance selection is never consulted regardless of
    # what it's set to.
    feat_df, _ = sa._build_cluster_feature_df(
        smd, group_A=None, group_B=None, features=None,
        merge_features=False, ligand_sdf=None,
        geom_features=True, geom_merge="aligned", recompute_distances=False,
    )
    feat_df_cl, fit_r_range = feat_df, 1 - trim_fraction
    if boundary_cap is not None:
        r_of_step = protocol_grid.reindex(feat_df["step"]).to_numpy(dtype=float)
        restricted = feat_df[r_of_step <= boundary_cap]
        if len(restricted) >= 2:
            feat_df_cl, fit_r_range = restricted, None

    clusterer = DTWPathModel(seed=sa.seed, do_plots=False, outdir=sa.outdir)
    mappings = clusterer.fit_transform(feat_df_cl, r_range=fit_r_range)
    smd.raw_data["path"] = smd.raw_data["trajname"].map(mappings)

    speed_data = smd.raw_data[smd.raw_data["speed"] == speed].copy()
    return protocol_grid, boundary_cap, speed_data


def _support_trim_common(common, results_df, trim_fraction):
    """Deployment's second filter on common_r (check_convergence, ~L1051-1063).

    Keep only steps where the best-supported path's replica count is within
    ``trim_fraction`` of the best-supported step overall. Adaptive: it drops
    low-coverage tails wherever they fall in r, independent of the boundary
    cap already applied. Shared by the force ladder and the single-speed
    validation harness so both match deployment's filtering exactly.
    """
    if (trim_fraction <= 0.0 or results_df is None or results_df.empty
            or "n_samples" not in results_df.columns or len(common) == 0):
        return common
    step_support = results_df.groupby("step")["n_samples"].max()
    n_ref = float(step_support.max())
    if n_ref <= 0:
        return common
    supported_steps = step_support[
        step_support / n_ref >= (1.0 - trim_fraction)].index
    return common[common.isin(supported_steps)]


def _compare_rung(k, pmf_k, prev_pmf, barrier, prev_barrier, r_ts, prev_r_ts,
                  common, tol, min_common_points, extra):
    """Deployment's three-criterion rung comparison.

    Shared by the force ladder and by the single-speed validation harness, so
    the gate in validate_against_api exercises exactly this code.

    The ``prev_*`` arguments are the *window reference* (see
    ``window_mean_pmf`` / ``window_reference_scalar``); with ``conv_window=1``
    that reference is literally the previous rung.

    NaN handling mirrors check_convergence: both NaN waives the criterion,
    one NaN marks non-convergence (inf). Those NaN branches are defensive and
    are currently unreachable: ``_compute_barrier_rts`` always returns finite
    values (``force_plateau`` interpolates; ``pmf_peak`` falls back to
    ``nanargmax(dG)``), and the extrapolated PMF is NaN-free, so
    ``window_reference_scalar`` never sees an all-NaN window either. They are
    kept in case a future TS detector is allowed to report "no peak".
    """
    if np.isnan(barrier) and np.isnan(prev_barrier):
        barrier_delta = np.nan
    elif np.isnan(barrier) or np.isnan(prev_barrier):
        barrier_delta = np.inf
    else:
        barrier_delta = abs(barrier - prev_barrier)

    if np.isnan(r_ts) and np.isnan(prev_r_ts):
        r_ts_delta = np.nan
    elif np.isnan(r_ts) or np.isnan(prev_r_ts):
        r_ts_delta = np.inf
    else:
        r_ts_delta = abs(r_ts - prev_r_ts)

    if len(common) < min_common_points:
        rmsd, converged, reason = np.nan, False, "insufficient_overlap"
    else:
        rmsd = float(np.sqrt(np.mean(
            (pmf_k.loc[common].to_numpy() - prev_pmf.loc[common].to_numpy()) ** 2)))
        converged = bool(
            (rmsd < tol["dG_weighted-rmsd"]) and
            (np.isnan(barrier_delta) or barrier_delta < tol["barrier_delta"]) and
            (np.isnan(r_ts_delta) or r_ts_delta < tol["r_ts_delta"]))
        reason = ""

    row = {"path": "mixture", "n_replicas": k,
           "dG_weighted-rmsd": rmsd, "barrier_delta": barrier_delta,
           "r_ts_delta": r_ts_delta, "barrier_height": barrier, "r_ts": r_ts,
           "converged": converged, "reason": reason,
           "n_common_points": len(common)}
    row.update(extra)
    return row


def force_convergence_ladder(sa, logs, speeds,
                             trace_min_replicas: int = 3,
                             trim_fraction: float = 0.1,
                             min_common_points: int = 5,
                             conv_window: int = 5) -> pd.DataFrame:
    """Multi-speed convergence ladder for the force estimator.

    At rung k, each speed contributes its first k replicas; per-speed weighted
    PMFs are extrapolated to v->0 and the extrapolated PMF is compared against
    the mean of the last ``conv_window`` rungs with deployment's three
    criteria.

    Parameters
    ----------
    sa : SMDAnalysis
        Already-built analysis object; supplies sysname/temperature/seed/
        outdir/reference_pdb and the shared PMF/clustering helpers.
    logs : list[str]
        Forward sMD log paths spanning all ``speeds``.
    speeds : list[float]
        Pulling speeds to ladder together (>=2, for the v->0 extrapolation).
    conv_window : int
        Rungs averaged to form the comparison reference, exactly as in
        ``check_convergence``. ``conv_window=1`` reproduces the pairwise
        previous-rung comparison this ladder originally used.

    Returns
    -------
    pd.DataFrame
        Same columns as ``check_convergence``'s convergence frame, with
        ``speed`` set to the string ``"ALL"`` (the row mixes every speed).
    """
    defaults = _ladder_defaults()
    tol = {
        "dG_weighted-rmsd": defaults["tol_rmsd"],
        "barrier_delta": defaults["tol_barrier"],
        "r_ts_delta": defaults["tol_r_ts"],
    }
    plateau_frac = defaults["plateau_frac"]
    boundary_buffer_frac = defaults["boundary_buffer_frac"]

    conv_policy = SupportPolicy(
        min_samples_per_step=defaults["min_samples_per_step_conv"],
        min_trajs_per_path=defaults["min_trajs_per_path_conv"],
    )

    # Per-speed state: clustering, ordering, protocol grid, boundary.
    per_speed = {}
    for speed in speeds:
        # Chronological, not by filename ID: the ID is a bare HHMMSS with no
        # date, so it would order multi-day cells by time of day and rung k
        # would not be "the first k replicas run".
        speed_logs = sorted([f for f in logs if SMDData._speed_from_log(f) == speed],
                            key=SMDData._replica_start_datetime)
        smd = SMDData(speed_logs, sysname=sa.sysname, temperature=sa.temperature,
                      reference_pdb=sa.reference_pdb)
        protocol_grid, boundary_cap, speed_data = _speed_state(
            sa, smd, speed, plateau_frac, boundary_buffer_frac, trim_fraction)
        traj_order = [os.path.basename(f)[:-4] for f in speed_logs]
        per_speed[speed] = {
            "smd": smd,
            "grid": protocol_grid,
            "cap": boundary_cap,
            "data": speed_data,
            "order": traj_order,
            "stats": defaultdict(lambda: {"n": 0, "sum_w": 0.0,
                                          "sum_w2": 0.0, "sum_exp": 0.0}),
            "samples": defaultdict(list),
            "counts": defaultdict(int),
        }

    K = min(len(per_speed[s]["order"]) for s in speeds)
    # RMSD cap for the extrapolated PMF: use the slowest speed's boundary,
    # because the extrapolated v->0 PMF has no force profile of its own to
    # detect a rupture boundary from -- the slowest (least-dissipative) speed
    # is the closest available proxy to v->0.
    slow_cap = per_speed[min(speeds)]["cap"]
    slow_grid = per_speed[min(speeds)]["grid"]

    rows = []
    # Bounded rung histories forming the comparison reference, mirroring
    # check_convergence exactly (deque(maxlen=w) + window_mean_pmf /
    # window_reference_scalar). w=1 keeps the original pairwise behaviour.
    _w = max(1, int(conv_window))
    hist_pmf = deque(maxlen=_w)
    hist_barrier = deque(maxlen=_w)
    hist_r_ts = deque(maxlen=_w)

    for k in range(1, K + 1):
        frames = []
        res_frames = []
        for speed in speeds:
            st = per_speed[speed]
            trajname = st["order"][k - 1]
            tdf = st["data"][st["data"]["trajname"] == trajname]
            if tdf.empty:
                continue
            paths = tdf["path"].dropna()
            if paths.empty:
                continue
            st["counts"][paths.iloc[0]] += 1
            for _, row in tdf.iterrows():
                key = (int(row["step"]), row["path"])
                s = st["stats"][key]
                w = float(row["work"])
                s["n"] += 1
                s["sum_w"] += w
                s["sum_w2"] += w * w
                s["sum_exp"] += np.exp(-st["smd"].beta * w)
                st["samples"][key].append(w)

            if k < trace_min_replicas:
                continue

            res = sa._results_from_running_stats(
                running_stats=st["stats"], running_samples=st["samples"],
                speed=speed, protocol_grid=st["grid"],
                estimator_name="force", beta=st["smd"].beta, policy=conv_policy,
            )
            if res.empty:
                continue
            res_frames.append(res)
            series = sa._weighted_series_from_results(
                results_df=res, path_traj_counts=st["counts"],
                value_col="dG", beta=st["smd"].beta,
                trim_fraction=trim_fraction, policy=conv_policy,
            )
            if series.empty:
                continue
            frames.append(pd.DataFrame({
                "step": series.index.astype(int),
                "r_coord": st["grid"].reindex(series.index).to_numpy(),
                "speed": speed,
                "estimator": "force",
                "dG_weighted": series.to_numpy(),
            }))

        if k < trace_min_replicas or len(frames) < 2:
            continue

        extrap = extrapolate_to_v0(pd.concat(frames, ignore_index=True),
                                   param="dG_weighted", speeds=speeds,
                                   min_speeds=2)
        if extrap is None or extrap.empty:
            continue
        pmf_k = (extrap.set_index("step")["dG_weighted"]
                 .astype(float).sort_index())
        if pmf_k.empty:
            continue

        # barrier/r_TS for the extrapolated PMF: boundary_method="pmf_peak"
        # (force_df=None) needs no force profile, unlike deployment's default
        # "force_plateau". When no peak is found, _compute_barrier_rts falls
        # back to argmax(dG): r_ts becomes the position of the PMF maximum and
        # barrier becomes dG[argmax] - dG[0] (not NaN, and not r_max -- for a
        # monotonic profile this makes r_ts the profile endpoint and the
        # barrier the total rise). The NaN-aware delta logic in _compare_rung
        # (both NaN waives the criterion, one NaN -> inf) guards other NaN
        # sources and is not exercised by this fallback.
        barrier, r_ts = sa._compute_barrier_rts(
            pmf_k, 1.0 / per_speed[min(speeds)]["smd"].beta, slow_grid,
            force_df=None, speed=0.0,
            boundary_method="pmf_peak", plateau_frac=plateau_frac,
        )

        # First usable rung: seed the history, no comparison yet.
        if not hist_pmf:
            hist_pmf.append(pmf_k)
            hist_barrier.append(barrier)
            hist_r_ts.append(r_ts)
            continue

        ref_pmf = window_mean_pmf(hist_pmf)
        ref_barrier = window_reference_scalar(hist_barrier)
        ref_r_ts = window_reference_scalar(hist_r_ts)

        common = pmf_k.index.intersection(ref_pmf.index)
        if slow_cap is not None and len(common):
            r_of_step = slow_grid.reindex(common).to_numpy(dtype=float)
            common = common[r_of_step <= slow_cap]
        combined_res = (pd.concat(res_frames, ignore_index=True)
                       if res_frames else pd.DataFrame())
        common = _support_trim_common(common, combined_res, trim_fraction)

        rows.append(_compare_rung(
            k=k, pmf_k=pmf_k, prev_pmf=ref_pmf,
            barrier=barrier, prev_barrier=ref_barrier,
            r_ts=r_ts, prev_r_ts=ref_r_ts,
            common=common, tol=tol, min_common_points=min_common_points,
            extra={"estimator": "force", "speed": "ALL"},
        ))
        # append to the bounded history (reference for the next comparison)
        hist_pmf.append(pmf_k)
        hist_barrier.append(barrier)
        hist_r_ts.append(r_ts)

    return pd.DataFrame(rows)
