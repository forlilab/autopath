"""report(run_dir): figures + summary.json for a metadynamics run directory."""
import json
import math
import os
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import (LABELS, PRIMARY, bands as _bands, bias_fes_scale, clamp_fractions, fe_numbers, hill_mask,
                       length_unit, recrossings)
from .data import load_run
from .plots import fig_bias, fig_cvs, fig_estimators, fig_fe, fig_fes2d


def _plain(o):
    """Recursively convert to JSON-safe types; NaN/inf become None."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    return str(o)


def _unsampled(run, nsig):
    """Grid points left blank by the hill mask: ``{rid or "pooled": n}``."""
    groups = {w.rid: [w] for w in run.walkers} if run.layout == "isolated" else {"pooled": run.walkers}
    return {k: int((~hill_mask([h for w in g for h in w.hills_raw], nsig)).sum()) for k, g in groups.items()}


def _version():
    try:
        from autopath import __version__
        return __version__
    except (ImportError, AttributeError):
        return None


def report(run_dir, funnel=None, learned_cv=None, bands=None, burn=0.2, t_rej=0.2, step_ns=1.0, out_dir=None,
           prefix="metad", min_points=12, min_count=0, min_bin_ess=0.0,
           hill_mask_sigmas=2.0, smooth_sigma=2.0, fmax=None, bins=None, bins_per_hill=4, bins_fallback=60,
           min_density=0.05, show_numbers=False, primary=PRIMARY) -> dict:
    """Write ``<prefix>_{cvs,bias,fe,estimators,fes-2d}.png`` and ``<prefix>_summary.json``; return the summary.

    Reports numbers only; convergence decisions are made by ``check_convergence``.

    Parameters
    ----------
    run_dir : str
    funnel, learned_cv : str, optional
        Passed to ``load_run``.
    bands : dict, optional
        ``{cv: {"bound": [lo, hi], "unbound": [lo, hi], "bound_end": "low"|"high"}}`` overrides.
    burn, t_rej : float
        Burn-in fraction for reweighting; rejection fraction for block analysis.
    step_ns : float
        ΔG°(t) time step.
    min_points : int
        Fewest ΔG°(t) points for a block analysis.
    bins : int or dict, optional
        Explicit bin count (or ``{cv: count}``) of the reweighted FES; overrides the two rules below.
    bins_per_hill : float
        Reweighted-FES bin width of a biased CV is its hill width / ``bins_per_hill``.
    bins_fallback : int
        Bin count over the padded sampled range for a CV without a hill width (monitored only).
    min_density : float
        Display mask of the reweighted FES: bins whose smoothed frame count is below this are blank.
    min_count, min_bin_ess : float
        Optional extra masks (0 = off): bins whose smoothed neighbourhood has fewer frames or a lower Kish ESS.
    smooth_sigma : float
        Gaussian smoothing (bins) of the reweighted FES histogram; 0 disables it.
    fmax : float, optional
        Colour cap (kcal/mol) of the 2D maps; default the 99th percentile of the unmasked values.
    show_numbers : bool
        Draw the ΔG° numbers as a text block on the FE figure (always written to the summary).
    hill_mask_sigmas : float
        Display mask of bias-derived maps: points farther than this many hill widths from every hill are blank.
    out_dir : str, optional
        Output directory; default ``run_dir``.
    primary : str
        Reweighting estimator (``rw_ct``, ``rw_final``, ``rw_balanced``) that drives the ΔG°(t) blocks, the
        2D colour cap and the convergence checks.

    Returns
    -------
    dict
        Same content as the JSON file; non-finite numbers stay NaN here and are null in the file.
        Reweighted numbers carry their Kish ESS: ``binding`` (per value), ``series["ess"]`` (per point),
        ``blocks["block_ess"]`` and ``fes_ess``. Bias-derived (final bias, F = -fes_scale·V; ``fes_scale`` is
        recorded with ``tempering``): ``binding_bias``, ``dg_bias_series`` and ``estimator_gap`` (reweighted
        minus bias-derived ΔG°, pooled or per walker). All four estimators (``rw_ct``, ``rw_final``,
        ``rw_balanced``, ``bias``): ``estimators`` (``{key: {"label", "binding", "series", "blocks"}}``),
        ``estimator_spread`` (max - min binding value, pooled or per walker), ``estimator_gaps`` (pairwise
        differences) and ``primary``.
    """
    run = load_run(run_dir, funnel=funnel, learned_cv=learned_cv)
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    b = _bands(run, bands)
    p = lambda k: os.path.join(out_dir, f"{prefix}_{k}")
    f1, _, s1 = fig_cvs(run, b, out_path=p("cvs.png")); plt.close(f1)
    f2, _, s2 = fig_bias(run, out_path=p("bias.png"), hill_mask_sigmas=hill_mask_sigmas); plt.close(f2)
    numbers = fe_numbers(run, b, burn=burn, t_rej=t_rej, step_ns=step_ns, min_points=min_points,
                         hill_mask_sigmas=hill_mask_sigmas, primary=primary)
    kw = dict(min_count=min_count, min_bin_ess=min_bin_ess, smooth_sigma=smooth_sigma, bins=bins,
              bins_per_hill=bins_per_hill, bins_fallback=bins_fallback, min_density=min_density)
    f3, _, s3, nums = fig_fe(run, b, burn=burn, t_rej=t_rej, step_ns=step_ns, out_path=p("fe.png"), numbers=numbers,
                             show_numbers=show_numbers, **kw)
    plt.close(f3)
    f5, _, s5 = fig_estimators(run, numbers, out_path=p("estimators.png"), t_rej=t_rej)
    plt.close(f5)
    f4, _, s4, n2 = fig_fes2d(run, b, burn=burn, out_path=p("fes-2d.png"), fmax=fmax,
                              hill_mask_sigmas=hill_mask_sigmas, primary=primary, **kw)
    if f4 is not None:
        plt.close(f4)
    for k in ("fes_ess", "fes_masked", "fes_kept", "fes_bins"):
        if n2[k] is not None:
            nums[k] = {**n2[k], **(nums[k] or {})}
    walkers = {}
    for w in run.walkers:
        rc = {nm: recrossings(w, nm, b[nm]) for nm in b}
        walkers[w.rid] = dict(
            time_ns=float(w.t[-1]), finished=w.finished, segments=list(w.segments),
            recrossings={k: v["n_cycles"] for k, v in rc.items()},
            band_fractions={k: dict(bound=v["frac_bound"], unbound=v["frac_unbound"]) for k, v in rc.items()})
    summary = dict(
        run_dir=run.run_dir, version=_version(), layout=run.layout, tempering=run.tempering,
        temperature=run.temperature, bias_factor=run.bias_factor, funnel=run.funnel is not None,
        cvs=[dict(name=c.name, grid_min=c.grid_min, grid_max=c.grid_max, periodic=c.periodic, bounds=c.bounds,
                  sigma=c.sigma, ang_per_unit=c.ang_per_unit) for c in run.cvs],
        monitored=list(run.monitored),
        bands={k: dict(asdict(v), unit=length_unit(run, k) or "CV units") for k, v in b.items()},
        params=dict(burn=burn, t_rej=t_rej, step_ns=step_ns, prefix=prefix, min_points=min_points,
                    min_count=min_count, min_bin_ess=min_bin_ess, hill_mask_sigmas=hill_mask_sigmas,
                    smooth_sigma=smooth_sigma, fmax=fmax, bins=bins, bins_per_hill=bins_per_hill,
                    bins_fallback=bins_fallback, min_density=min_density, primary=primary),
        walkers=walkers, clamp=clamp_fractions(run), binding=nums["binding"], series=nums["series"],
        blocks=nums["blocks"], binding_bias=nums["binding_bias"], dg_bias_series=nums["dg_bias_series"],
        estimator_gap=nums["estimator_gap"], primary=numbers["primary"],
        estimators={m: dict(label=LABELS[m], binding=asdict(e["binding"]), series=asdict(e["series"]),
                            blocks=e["blocks"]) for m, e in numbers["estimators"].items()},
        estimator_spread=numbers["estimator_spread"], estimator_gaps=numbers["estimator_gaps"], fes_scale=bias_fes_scale(run), fes_ess=nums["fes_ess"],
        fes_masked=nums["fes_masked"], fes_kept=nums["fes_kept"], fes_bins=nums["fes_bins"], fmax_kcal=n2["fmax"],
        bias_unsampled=_unsampled(run, hill_mask_sigmas), unit="kJ/mol",
        skipped_walkers=list(run.skipped), skipped_panels=list(dict.fromkeys(s1 + s2 + s3 + s5 + s4)),
        files=dict(cvs=p("cvs.png"), bias=p("bias.png"), fe=p("fe.png"), estimators=p("estimators.png"),
                   fes2d=p("fes-2d.png") if f4 is not None else None, summary=p("summary.json")))
    with open(p("summary.json"), "w") as fh:
        json.dump(_plain(summary), fh, indent=1, allow_nan=False)
    return summary
