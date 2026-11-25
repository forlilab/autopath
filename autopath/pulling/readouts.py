"""Scalar ΔG readouts of sMD free-energy profiles for ranking several ligands.

Two readouts, both label-free:

- profile readout: ΔG at the most prominent PMF peak for the cumulant (argmax if
  none), the profile maximum for Jarzynski and force.
- rupture readout: ΔG at d*, the median over the ligand set of each ligand's
  force-peak displacement (argmax of the running mean of the mean pulling force).
"""
import numpy as np
import pandas as pd

from .Estimators import _find_pmf_peak

KBT = 0.0083145 * 300.0          # kJ/mol at 300 K
RUPTURE_SMOOTH = 25              # running-mean window (steps) of the mean force
READOUT_ROWS = (("cumulant", "per_speed"), ("jarzynski", "per_speed"), ("force", "v0"))


def _readout_profiles(pmfs: pd.DataFrame):
    """Yield (estimator, speed, d, dG): cumulant/Jarzynski per speed, force at v->0 only."""
    for (est, sp), g in pmfs.groupby(["estimator", "speed"]):
        sp = float(sp)
        if (est == "force") != (sp == 0.0) or est not in ("cumulant", "jarzynski", "force"):
            continue
        g = g.dropna(subset=["r_coord", "dG"]).sort_values("r_coord")
        if len(g) > 3:
            r = g["r_coord"].to_numpy(float)
            yield est, sp, r - r[0], g["dG"].to_numpy(float)


def profile_readouts(pmfs: pd.DataFrame, kBT: float = KBT) -> pd.DataFrame:
    """Profile readout of one system's ``mixture_pmfs.csv``.

    Parameters
    ----------
    pmfs : pd.DataFrame
        Columns ``estimator``, ``speed``, ``r_coord``, ``dG``.
    kBT : float
        Thermal energy (kJ/mol) for the peak prominence threshold.

    Returns
    -------
    pd.DataFrame
        ``estimator``, ``speed``, ``dG_profile``, ``peak_fell_back`` (cumulant without a prominent peak).
    """
    rows = []
    for est, sp, d, dG in _readout_profiles(pmfs):
        fell = False
        if est == "cumulant":
            pk = _find_pmf_peak(d, dG, kBT)[0]
            fell = pk is None
            val = float(np.interp(pk, d, dG)) if not fell else float(np.nanmax(dG))
        else:
            val = float(np.nanmax(dG))
        rows.append(dict(estimator=est, speed=sp, dG_profile=val, peak_fell_back=fell))
    return pd.DataFrame(rows, columns=["estimator", "speed", "dG_profile", "peak_fell_back"])


def rupture_displacement(processed: pd.DataFrame, smooth: int = RUPTURE_SMOOTH) -> dict:
    """{speed: force-peak displacement (nm)} from one system's traces (``autopath.pulling.outputs.load_traces``).

    Steps not reached by every pull are dropped, so the profile ends with the shortest pull.
    """
    out = {}
    for sp, g in processed.groupby("speed"):
        F = g.pivot_table(index="step", columns="trajname", values="force").dropna()
        if F.empty:
            continue
        r = g.groupby("step")["r_coord"].median().reindex(F.index).to_numpy()
        fm = pd.Series(F.mean(axis=1).to_numpy()).rolling(smooth, center=True, min_periods=1).mean()
        out[float(sp)] = float(r[int(np.argmax(fm.to_numpy()))] - r[0])
    return out


def rupture_readouts(pmfs: dict, ruptures: dict) -> pd.DataFrame:
    """Rupture readout of a ligand set.

    Parameters
    ----------
    pmfs : dict
        ``{system: mixture_pmfs DataFrame}``.
    ruptures : dict
        ``{system: rupture_displacement(...)}``.

    Returns
    -------
    pd.DataFrame
        ``system``, ``estimator``, ``speed``, ``d_star`` (nm), ``dG_rupture``; NaN where
        the profile ends before d*. d* is the per-speed median over systems, and the
        median over all speeds for force at v->0.
    """
    rp = pd.DataFrame([dict(system=s, speed=sp, d=v) for s, r in ruptures.items() for sp, v in r.items()])
    dstar = rp.groupby("speed")["d"].median().to_dict() if len(rp) else {}
    dstar[0.0] = float(rp["d"].median()) if len(rp) else np.nan
    rows = []
    for s, pm in pmfs.items():
        for est, sp, d, dG in _readout_profiles(pm):
            ds = dstar.get(sp, np.nan)
            val = float(np.interp(ds, d, dG)) if np.isfinite(ds) and d[-1] >= ds else np.nan
            rows.append(dict(system=s, estimator=est, speed=sp, d_star=ds, dG_rupture=val))
    return pd.DataFrame(rows, columns=["system", "estimator", "speed", "d_star", "dG_rupture"])
