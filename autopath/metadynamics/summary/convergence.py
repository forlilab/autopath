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
    return "n/a" if not math.isfinite(v) else f"{v:.2f}"


def min_recrossings(n, cv=None):
    """Bound-unbound-bound cycles >= n: per walker if isolated, summed over walkers otherwise.

    ``cv=None`` checks the funnel z when the summary has a funnel, else every biased CV.
    """
    def f(s):
        names = [cv] if cv else (["funnel_z"] if s.get("funnel") else [c["name"] for c in s["cvs"]])
        ws = s["walkers"]
        units = ws if s["layout"] == "isolated" else {"all": None}
        bad = []
        for k in names:
            for rid in units:
                if rid == "all":
                    tot = sum(w["recrossings"].get(k, 0) for w in ws.values())
                else:
                    tot = ws[rid]["recrossings"].get(k, 0)
                if tot < n:
                    bad.append(f"{rid}/{k}={tot}")
        return (not bad, f"recrossings below {n}: {', '.join(bad)}" if bad else f"recrossings >= {n}")
    return _named(f"min_recrossings({n})", f)


def _blocks(s):
    b = s.get("blocks")
    if not b:
        return {}
    return b if s["layout"] == "isolated" else {"all": b}


def _block_check(key, tol_kcal, strict, label):
    def f(s):
        bl = _blocks(s)
        if not bl:
            return False, f"{label}: no block analysis"
        vals = {k: _kcal(v.get(key)) for k, v in bl.items()}
        bad = {k: x for k, x in vals.items() if not (x <= tol_kcal if not strict else x < tol_kcal)}
        if bad:
            return False, f"{label} not within {tol_kcal} kcal/mol: " + ", ".join(f"{k}={_fmt(x)}" for k, x in bad.items())
        return True, f"{label} ok (<= {tol_kcal} kcal/mol)"
    return f


def last_blocks_within(tol_kcal):
    """|last block - previous block| <= tol_kcal (kcal/mol); NaN fails."""
    return _named(f"last_blocks_within({tol_kcal})", _block_check("last_blocks_diff", tol_kcal, False, "last blocks"))


def bootstrap_sd_below(tol_kcal):
    """Block-bootstrap SD < tol_kcal (kcal/mol); NaN fails."""
    return _named(f"bootstrap_sd_below({tol_kcal})", _block_check("boot_sd", tol_kcal, True, "bootstrap SD"))


def walker_spread_within(tol_kcal):
    """Max-min of per-walker binding free energies <= tol_kcal (kcal/mol); isolated runs only."""
    def f(s):
        if s["layout"] != "isolated":
            return True, "walker spread: not applicable (shared)"
        sp = _kcal((s.get("binding") or {}).get("spread"))
        if sp <= tol_kcal:
            return True, f"walker spread {_fmt(sp)} kcal/mol ok"
        return False, f"walker spread {_fmt(sp)} kcal/mol > {tol_kcal}"
    return _named(f"walker_spread_within({tol_kcal})", f)


def clamp_fraction_below(frac):
    """Fraction of hills at a CV bound < frac, for every bounded CV and walker (or pooled)."""
    def f(s):
        bad = [f"{k}/{r}={v:.2f}" for k, d in (s.get("clamp") or {}).items() for r, v in d.items() if v >= frac]
        return (not bad, f"hills at a clamp >= {frac}: {', '.join(bad)}" if bad else f"clamp fraction < {frac}")
    return _named(f"clamp_fraction_below({frac})", f)
