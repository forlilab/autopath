"""Pure decision logic for sMD convergence.

Separated from AnalysisSMD so the stopping rule can be unit-tested without
building PMFs, and so the deployment loop and offline calibration share one
implementation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def first_streak(df: pd.DataFrame, k_consec: int = 3) -> float:
    """Smallest ``n_replicas`` ending a run of ``k_consec`` consecutive passes.

    Returns NaN when the criterion is never met. That is a *censored*
    observation: callers must report it, never silently drop it.
    """
    d = df.sort_values("n_replicas")
    flags = d["converged"].astype(bool).to_numpy()
    ks = d["n_replicas"].to_numpy()
    run = 0
    for flag, k in zip(flags, ks):
        run = run + 1 if flag else 0
        if run >= k_consec:
            return float(k)
    return float("nan")


def tail_converged(df: pd.DataFrame, k_consec: int) -> bool:
    """True when the LAST ``k_consec`` rungs, in replica order, all passed.

    This is the *deployment* predicate: "are we converged right now?".  It is
    deliberately not ``first_streak``, which answers the offline-calibration
    question "at which rung would we first have stopped?".  ``first_streak``
    finds a streak anywhere in the history, so it would stop a restarted
    campaign whose latest rung failed (rungs 46/47 pass, 48 fails), and it
    would fire whenever re-clustering retroactively flips an older rung —
    ``conv_df`` is rebuilt from scratch on every call and is not append-only.

    Returns False when there are fewer than ``k_consec`` rows, i.e. the
    criterion cannot yet be evaluated.
    """
    if df is None or k_consec < 1 or len(df) < k_consec:
        return False
    d = df.sort_values("n_replicas")
    return bool(d["converged"].astype(bool).to_numpy()[-k_consec:].all())


def window_mean_pmf(pmfs: Sequence[pd.Series]) -> pd.Series:
    """Elementwise mean of PMF series over the index they share.

    A single-element window returns that series unchanged, which is what makes
    ``conv_window=1`` identical to comparing against the previous rung.
    """
    pmfs = list(pmfs)
    if len(pmfs) == 1:
        return pmfs[0]
    frame = pd.concat(pmfs, axis=1, join="inner")
    return frame.mean(axis=1)


def window_reference_scalar(values: Sequence[float]) -> float:
    """Median of the finite values in the window; NaN when none are finite.

    Median rather than mean because one replica carrying an extreme work value
    displaces a mean. NaN entries mean "no peak detected at that rung" and are
    ignored rather than poisoning the reference; a window of only NaNs yields
    NaN, which the caller treats as "criterion waived".
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


AUTOSTOP_ESTIMATORS = ("cumulant", "jarzynski", "force")


def validate_autostop_options(estimator: str, alternate_speeds: bool,
                              speeds: Sequence[float],
                              conv_window: int = 5,
                              conv_streak: int = 3) -> None:
    """Reject illegal autostop option combinations. Raises, never repairs.

    ``force`` has no single-speed v->0 intercept, so it needs at least two
    speeds and needs them advancing together — which only happens when speeds
    are alternated rather than run to completion one at a time.

    The numeric knobs are validated too: ``conv_streak=0`` makes the stopping
    predicate vacuously true (a zero-length tail is trivially all-True), which
    would stop the campaign after a single rung; ``conv_window=0`` would leave
    the comparison reference undefined.
    """
    if estimator not in AUTOSTOP_ESTIMATORS:
        raise ValueError(
            f"Unknown autostop estimator '{estimator}'. "
            f"Allowed: {sorted(AUTOSTOP_ESTIMATORS)}"
        )
    if estimator == "force":
        if not alternate_speeds:
            raise ValueError(
                "sMD_autostop_estimator='force' requires sMD_alternate_speeds=True: "
                "the force estimator needs >=2 speeds advancing together to form a "
                "v->0 intercept, which the one-speed-at-a-time loop cannot provide."
            )
        if len(speeds) < 2:
            raise ValueError(
                "sMD_autostop_estimator='force' requires at least two speeds in "
                f"sMD_pulling_speeds; got {len(speeds)}."
            )
    if int(conv_window) < 1:
        raise ValueError(
            f"sMD_conv_window must be >= 1 (1 == compare against the previous "
            f"rung); got {conv_window}."
        )
    if int(conv_streak) < 1:
        raise ValueError(
            f"sMD_conv_streak must be >= 1; got {conv_streak}. A streak of 0 "
            f"would make the stopping predicate pass immediately."
        )


def round_robin_order(live: dict) -> list:
    """Speeds still accepting replicas, slowest first, retired ones dropped."""
    return [s for s in sorted(live) if live[s]]
