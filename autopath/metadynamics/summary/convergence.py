"""Pluggable convergence criteria over a report() summary; report() itself never decides."""
import math
from dataclasses import dataclass, field

KJ_PER_KCAL = 4.184


@dataclass
class Verdict:
    needs_more_sampling: bool
    passed: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


def check_convergence(summary, criteria) -> Verdict:
    """Apply each criterion ``(summary) -> (ok, reason)``; more sampling is needed if any fails."""
    passed, reasons = {}, []
    for c in criteria:
        ok, reason = c(summary)
        passed[c.__name__] = bool(ok)
        if not ok:
            reasons.append(reason)
    return Verdict(needs_more_sampling=bool(reasons), passed=passed, reasons=reasons)


def _named(name, fn):
    fn.__name__ = name
    return fn


def _kcal(v):
    """kJ/mol (None/NaN allowed) to kcal/mol; NaN if missing."""
    return float("nan") if v is None else float(v) / KJ_PER_KCAL


def _fmt(v):
    return "undefined" if not math.isfinite(v) else f"{v:.2f}"


def _arg(*a):
    return ", ".join(str(x) for x in a if x is not None)


def min_recrossings(n, cv=None):
    """Bound-unbound-bound cycles >= n: per walker if isolated, summed over walkers otherwise.

    ``cv=None`` checks the funnel z when the summary has a funnel, else every biased CV. A CV without
    recrossings in the summary fails as unknown.
    """
    def f(s):
        names = [cv] if cv else (["funnel_z"] if s.get("funnel") else [c["name"] for c in s["cvs"]])
        ws = s["walkers"]
        units = ws if s["layout"] == "isolated" else {"all": None}
        bad, unknown = [], set()
        for k in names:
            for rid in units:
                have = ws.values() if rid == "all" else [ws[rid]]
                if any(k not in w["recrossings"] for w in have):
                    unknown.add(k)
                    continue
                tot = sum(w["recrossings"][k] for w in have)
                if tot < n:
                    bad.append(f"{rid}/{k}={tot}")
        msg = [f"unknown CV {k}" for k in sorted(unknown)] + ([f"recrossings below {n}: {', '.join(bad)}"] if bad else [])
        return (not msg, "; ".join(msg) if msg else f"recrossings >= {n}")
    return _named(f"min_recrossings({_arg(n, cv)})", f)


def _blocks(s):
    """{unit: block dict or None}; one unit per walker if isolated, else "all"."""
    b = s.get("blocks")
    if s["layout"] == "isolated":
        return {rid: (b or {}).get(rid) for rid in s["walkers"]}
    return {"all": b}


def _kinds(s):
    """Kinds of the binding values in a summary ("dG0", "band_dF")."""
    b = s.get("binding") or {}
    vals = [b["pooled"]] if b.get("pooled") else list((b.get("per_walker") or {}).values())
    return {v.get("kind") for v in vals if v}


def _band_fail(s, allow_band):
    """Failure reason when ΔG° thresholds meet a band ΔF and ``allow_band`` is False."""
    if not allow_band and "band_dF" in _kinds(s):
        return "kind is band ΔF, not ΔG°"
    return None


def _block_check(key, tol_kcal, strict, label, allow_band):
    def f(s):
        why = _band_fail(s, allow_band)
        if why:
            return False, f"{label}: {why}"
        bad = []
        for k, v in _blocks(s).items():
            if v is None:
                bad.append(f"{k}: blocks undefined (ΔG°(t) too short)")
                continue
            x = _kcal(v.get(key))
            if not math.isfinite(x):
                bad.append(f"{k}: {label} undefined")
            elif not (x < tol_kcal if strict else x <= tol_kcal):
                bad.append(f"{k}: {label} {x:.2f} kcal/mol")
        op = "<" if strict else "<="
        if bad:
            return False, f"{label} not {op} {tol_kcal} kcal/mol: " + "; ".join(bad)
        return True, f"{label} {op} {tol_kcal} kcal/mol"
    return f


def last_blocks_within(tol_kcal, allow_band=False):
    """|last block - previous block| <= tol_kcal (kcal/mol); undefined fails; band ΔF fails unless
    ``allow_band``."""
    return _named(f"last_blocks_within({tol_kcal})",
                  _block_check("last_blocks_diff", tol_kcal, False, "last blocks", allow_band))


def bootstrap_sd_below(tol_kcal, allow_band=False):
    """Block-bootstrap SD < tol_kcal (kcal/mol); undefined fails; band ΔF fails unless ``allow_band``."""
    return _named(f"bootstrap_sd_below({tol_kcal})", _block_check("boot_sd", tol_kcal, True, "bootstrap SD",
                                                                    allow_band))


def walker_spread_within(tol_kcal, allow_band=False):
    """Max-min of per-walker binding free energies <= tol_kcal (kcal/mol); isolated runs only.

    Fails when any walker's value is undefined, with fewer than two defined walkers, and for band ΔF
    unless ``allow_band``.
    """
    def f(s):
        if s["layout"] != "isolated":
            return True, "walker spread: not applicable (shared)"
        why = _band_fail(s, allow_band)
        if why:
            return False, f"walker spread: {why}"
        b = s.get("binding") or {}
        n, k = b.get("n_walkers"), b.get("n_defined")
        if n is not None and k is not None and k < n:
            return False, f"walker spread: {n - k} of {n} walkers undefined"
        if k is not None and k < 2:
            return False, "walker spread undefined (fewer than 2 defined walkers)"
        sp = _kcal(b.get("spread"))
        if not math.isfinite(sp):
            return False, "walker spread undefined"
        if sp <= tol_kcal:
            return True, f"walker spread {sp:.2f} kcal/mol <= {tol_kcal}"
        return False, f"walker spread {sp:.2f} kcal/mol > {tol_kcal}"
    return _named(f"walker_spread_within({tol_kcal})", f)


def clamp_fraction_below(frac):
    """Fraction of hills at a CV bound < frac, for every bounded CV and walker (or pooled)."""
    def f(s):
        items = [(f"{k}/{r}", v) for k, d in (s.get("clamp") or {}).items() for r, v in d.items()]
        if not items:
            return True, "clamp fraction: not applicable (no bounded CV)"
        und = [k for k, v in items if v is None or not math.isfinite(v)]
        bad = [f"{k}={v:.2f}" for k, v in items if k not in und and v >= frac]
        if und or bad:
            return False, "; ".join(([f"clamp fraction undefined: {', '.join(und)}"] if und else [])
                                    + ([f"hills at a clamp >= {frac}: {', '.join(bad)}"] if bad else []))
        return True, f"clamp fraction < {frac}"
    return _named(f"clamp_fraction_below({frac})", f)


def min_ess(n):
    """Kish ESS of the binding estimate >= n: pooled, or every walker if isolated; undefined fails."""
    def f(s):
        b = s.get("binding") or {}
        vals = {"all": b["pooled"]} if b.get("pooled") else dict(b.get("per_walker") or {})
        if s["layout"] == "isolated":
            vals.update({r: None for r in s["walkers"] if r not in vals})
        if not vals:
            return False, "ESS undefined (no binding estimate)"
        bad = []
        for k, v in vals.items():
            e = (v or {}).get("ess")
            if e is None or not math.isfinite(e):
                bad.append(f"{k}: ESS undefined")
            elif e < n:
                bad.append(f"{k}: ESS {e:.0f}")
        return (False, f"ESS below {n}: " + "; ".join(bad)) if bad else (True, f"ESS >= {n}")
    return _named(f"min_ess({n})", f)


def estimators_agree(tol_kcal):
    """|reweighted - bias-derived binding value| <= tol_kcal (kcal/mol): pooled if shared, every walker if
    isolated; an undefined gap fails."""
    def f(s):
        g = s.get("estimator_gap") or {}
        vals = {"all": g.get("pooled")} if s["layout"] != "isolated" else \
            {r: (g.get("per_walker") or {}).get(r) for r in s["walkers"]}
        bad = []
        for k, v in vals.items():
            x = _kcal(v)
            if not math.isfinite(x):
                bad.append(f"{k}: undefined")
            elif abs(x) > tol_kcal:
                bad.append(f"{k}: gap {x:.2f} kcal/mol")
        if bad:
            return False, f"estimator gap not within {tol_kcal} kcal/mol: " + "; ".join(bad)
        return True, f"estimator gap within {tol_kcal} kcal/mol"
    return _named(f"estimators_agree({tol_kcal})", f)
