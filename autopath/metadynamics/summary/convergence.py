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
    return _named(f"min_recrossings({_arg(n, cv)})", f)


def _blocks(s):
    """{unit: block dict or None}; one unit per walker if isolated, else "all"."""
    b = s.get("blocks")
    if s["layout"] == "isolated":
        return {rid: (b or {}).get(rid) for rid in s["walkers"]}
    return {"all": b}


def _block_check(key, tol_kcal, strict, label):
    def f(s):
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


def last_blocks_within(tol_kcal):
    """|last block - previous block| <= tol_kcal (kcal/mol); undefined fails."""
    return _named(f"last_blocks_within({tol_kcal})", _block_check("last_blocks_diff", tol_kcal, False, "last blocks"))


def bootstrap_sd_below(tol_kcal):
    """Block-bootstrap SD < tol_kcal (kcal/mol); undefined fails."""
    return _named(f"bootstrap_sd_below({tol_kcal})", _block_check("boot_sd", tol_kcal, True, "bootstrap SD"))


def walker_spread_within(tol_kcal):
    """Max-min of per-walker binding free energies <= tol_kcal (kcal/mol); isolated runs only."""
    def f(s):
        if s["layout"] != "isolated":
            return True, "walker spread: not applicable (shared)"
        sp = _kcal((s.get("binding") or {}).get("spread"))
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
