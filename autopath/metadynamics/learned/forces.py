# autopath/metadynamics/learned/forces.py
"""OpenMM/cvpack builders for learned-CV features; same formulas as features.evaluate."""
import cvpack
import numpy as np
import openmm as mm
from openmm.app.internal.unitcell import reducePeriodicBoxVectors
from openmm import unit

from .features import (CONTACT_CUTOFF_FACTOR, CONTACT_SWITCH_FACTOR, COUNT_CUTOFF_FACTOR,
                       COUNT_SWITCH_FACTOR)


def _nonbonded(system):
    return next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))


def _centroid(expr, groups, n_bond_groups=None, bonds=None):
    f = mm.CustomCentroidBondForce(n_bond_groups or len(groups), expr)
    for g in groups:
        f.addGroup([int(i) for i in g], [1.0] * len(g))
    for b in (bonds or [list(range(len(groups)))]):
        f.addBond(b, [])
    f.setUsesPeriodicBoundaryConditions(True)
    return f


def _wrap(force, name):
    return cvpack.OpenMMForceWrapper(force, unit.dimensionless, name=name)


def feature_cv(f, system, name):
    if f.kind == "pair":
        b = mm.CustomBondForce(f"1/(1+(r/{f.r0})^6)")
        for i, j in f.pairs:
            b.addBond(int(i), int(j), [])
        b.setUsesPeriodicBoundaryConditions(True)
        return _wrap(b, name)
    if f.kind == "contacts":
        return cvpack.NumberOfContacts(f.group_a, f.group_b, _nonbonded(system), reference=1.0,
                                       thresholdDistance=f.r0 * unit.nanometer,
                                       cutoffFactor=CONTACT_CUTOFF_FACTOR,
                                       switchFactor=CONTACT_SWITCH_FACTOR, name=name)
    if f.kind == "centroid_distance":
        return _wrap(_centroid("distance(g1,g2)", f.groups), name)
    if f.kind == "centroid_count":
        groups = [f.groups[0]] + [[i] for i in f.group_b]
        bonds = [[0, k] for k in range(1, len(groups))]
        box = system.getDefaultPeriodicBoxVectors()
        a, b, c = reducePeriodicBoxVectors(box)
        half = min(a.x, b.y, c.z).value_in_unit(unit.nanometer) / 2
        if COUNT_CUTOFF_FACTOR * f.r0 >= half:
            raise ValueError(f"{name}: count cutoff {COUNT_CUTOFF_FACTOR * f.r0:.3g} nm >= half box {half:.3g} nm")
        rs, rc = COUNT_SWITCH_FACTOR * f.r0, COUNT_CUTOFF_FACTOR * f.r0
        expr = (f"1/(1+(d/{f.r0})^8)*(1-10*x^3+15*x^4-6*x^5); "
                f"x=min(1,max(0,(d-{rs})/({rc}-{rs}))); d=distance(g1,g2)")
        return _wrap(_centroid(expr, groups, 2, bonds), name)
    if f.kind == "cos":
        # a·b from centroid distances; a = c2-c1, b = c4-c3
        expr = ("(distance(g2,g3)^2 + distance(g1,g4)^2 - distance(g2,g4)^2 - distance(g1,g3)^2)"
                "/(2*distance(g1,g2)*distance(g3,g4))")
        return _wrap(_centroid(expr, f.groups), name)
    raise ValueError(f"unknown feature kind {f.kind!r}")


def _fused_contacts(feats, coeffs, system):
    """One CustomNonbondedForce for every contacts feature: sum_k c_k * contacts_k."""
    n = system.getNumParticles()
    sides = {}                                      # atom -> set of (feature, side)
    for k, f in enumerate(feats):
        if set(f.group_a) & set(f.group_b):
            raise ValueError(f"contacts feature {f.name!r} has atoms in both groups: "
                             "not supported in the fused s_w")
        for side, g in ((0, f.group_a), (1, f.group_b)):
            for i in g:
                sides.setdefault(int(i), set()).add((k, side))
    # atoms with the same feature sides form a class; class pairs are disjoint interaction groups,
    # so every atom pair is computed at most once
    sig = {}
    for i in sorted(sides):
        sig.setdefault(frozenset(sides[i]), []).append(i)
    sigs, members = list(sig), list(sig.values())
    cls = np.zeros(n, dtype=int)
    for c, m in enumerate(members):
        cls[m] = c
    r0s = sorted({f.r0 for f in feats})
    nc = len(members)
    tables = {r0: np.zeros((nc, nc)) for r0 in r0s}
    links = set()
    for c1 in range(nc):
        for c2 in range(c1 + 1, nc):
            for (k, s1) in sigs[c1]:
                if (k, 1 - s1) in sigs[c2]:
                    t = tables[feats[k].r0]
                    t[c1, c2] += coeffs[k]
                    t[c2, c1] = t[c1, c2]
                    links.add((c1, c2))
    terms, defs = [], []
    for g, r0 in enumerate(r0s):
        rs, rc = CONTACT_SWITCH_FACTOR * r0, CONTACT_CUTOFF_FACTOR * r0
        terms.append(f"t{g}(cls1, cls2)*sw{g}/(1+(r/{r0!r})^6)")
        defs.append(f"sw{g} = 1-10*x{g}^3+15*x{g}^4-6*x{g}^5")
        defs.append(f"x{g} = min(1, max(0, (r-{rs!r})/({rc!r}-{rs!r})))")
    force = mm.CustomNonbondedForce("; ".join([" + ".join(terms)] + defs))
    force.addPerParticleParameter("cls")
    for g, r0 in enumerate(r0s):
        force.addTabulatedFunction(f"t{g}", mm.Discrete2DFunction(nc, nc, tables[r0].ravel(order="F").tolist()))
    for i in range(n):
        force.addParticle([float(cls[i])])
    nb = _nonbonded(system)
    for e in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(e)
        force.addExclusion(i, j)
    pbc = nb.usesPeriodicBoundaryConditions()
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic if pbc
                             else mm.CustomNonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(CONTACT_CUTOFF_FACTOR * max(r0s))
    force.setUseSwitchingFunction(False)            # per-r0 switching is in the expression
    force.setUseLongRangeCorrection(False)
    for c1, c2 in sorted(links):
        force.addInteractionGroup(members[c1], members[c2])
    return force


def fused_sw_forces(features, weights, mu, sd, system):
    """Few forces whose energies sum to sum_k c_k f_k, c_k = weights_k / sd_k.

    Parameters
    ----------
    features : list of Feature (kinds pair, contacts, centroid_distance, cos)
    weights, mu, sd : (n_features,) arrays
    system : openmm.System with a NonbondedForce (contacts exclusions and PBC)

    Returns
    -------
    forces : list of cvpack CVs, one per kind present
    const : float, -sum_k c_k mu_k; s = sum(forces) + const
    """
    w, mu, sd = (np.asarray(a, dtype=float).ravel() for a in (weights, mu, sd))
    if not (len(w) == len(mu) == len(sd) == len(features)):
        raise ValueError("fused s_w needs one weight, mu and sd per feature")
    c = w / sd
    by_kind = {}
    for k, f in enumerate(features):
        if f.kind not in ("pair", "contacts", "centroid_distance", "cos"):
            raise ValueError(f"feature {f.name!r} of kind {f.kind!r} is not supported in the fused s_w")
        by_kind.setdefault(f.kind, []).append(k)
    forces = []
    if "pair" in by_kind:
        b = mm.CustomBondForce("c/(1+(r/r0)^6)")
        b.addPerBondParameter("c")
        b.addPerBondParameter("r0")
        for k in by_kind["pair"]:
            for i, j in features[k].pairs:
                b.addBond(int(i), int(j), [float(c[k]), features[k].r0])
        b.setUsesPeriodicBoundaryConditions(True)
        forces.append(_wrap(b, "fused_pair"))
    if "contacts" in by_kind:
        ks = by_kind["contacts"]
        forces.append(_wrap(_fused_contacts([features[k] for k in ks], c[ks], system), "fused_contacts"))
    for kind, n_groups, expr in (
            ("centroid_distance", 2, "distance(g1,g2)"),
            ("cos", 4, "(distance(g2,g3)^2 + distance(g1,g4)^2 - distance(g2,g4)^2 - distance(g1,g3)^2)"
                       "/(2*distance(g1,g2)*distance(g3,g4))")):
        if kind not in by_kind:
            continue
        cb = mm.CustomCentroidBondForce(n_groups, f"c*{expr}")
        cb.addPerBondParameter("c")
        for k in by_kind[kind]:
            first = cb.getNumGroups()
            for g in features[k].groups:
                cb.addGroup([int(i) for i in g], [1.0] * len(g))
            cb.addBond(list(range(first, first + n_groups)), [float(c[k])])
        cb.setUsesPeriodicBoundaryConditions(True)
        forces.append(_wrap(cb, f"fused_{kind}"))
    return forces, float(-np.sum(c * mu))


def fused_sw_composite(features, weights, mu, sd, system, key=None):
    """LinearComposite over the fused forces with the same value as ``composite_cv``."""
    from ..BoundaryCorrection import LinearComposite
    forces, const = fused_sw_forces(features, weights, mu, sd, system)
    shift = np.zeros(len(forces))
    shift[0] = -const
    return LinearComposite(forces, np.ones(len(forces)), shift, np.ones(len(forces)), key=key)


def combined_cv(features, weights, mu, sd, system, name="learned"):
    if len(features) > 32:
        raise ValueError(f"{len(features)} features exceed OpenMM CustomCVForce's limit of 32 collective variables")
    names = [f"f{k}" for k in range(len(features))]
    # global parameters: prefixed so several components can share one Context
    p = f"{name}_"
    expr = " + ".join(f"{p}w{k}*({n} - {p}mu{k})/{p}sd{k}" for k, n in enumerate(names))
    params = {}
    for k in range(len(features)):
        params[f"{p}w{k}"], params[f"{p}mu{k}"], params[f"{p}sd{k}"] = float(weights[k]), float(mu[k]), float(sd[k])
    inner = [feature_cv(f, system, n) for f, n in zip(features, names)]
    return cvpack.MetaCollectiveVariable(expr, inner, unit.dimensionless, name=name, **params)


def composite_cv(features, weights, mu, sd, system, key=None):
    """Flat learned CV: the feature forces plus their standardised weights, for the bias force."""
    from ..BoundaryCorrection import LinearComposite
    if len(features) > 32:
        raise ValueError(f"{len(features)} features exceed OpenMM CustomCVForce's limit of 32 collective variables")
    forces = [feature_cv(f, system, f"f{k}") for k, f in enumerate(features)]
    return LinearComposite(forces, weights, mu, sd, key=key)


def funnel_z_force(host, guest, axis, name="funnel_z_lc"):
    """Unclamped funnel z (nm): n . (COM_guest - COM_host), mass-weighted, literal axis.

    Parameters
    ----------
    host, guest : atom index lists
    axis : (3,) array, normalised here
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    expr = " + ".join(f"({float(c)!r})*({x}2-{x}1)" for c, x in zip(n, "xyz"))
    return cvpack.CentroidFunction(expr, unit.nanometer, [[int(i) for i in host], [int(i) for i in guest]],
                                   pbc=False, weighByMass=True, name=name)


def is_rupture_cv(npz_path) -> bool:
    """True for a learned_cv.npz from a rupture-class learn (one biased component, s_w)."""
    with np.load(npz_path, allow_pickle=False) as d:
        return "component" in d.files


def check_funnel_match(npz_path, funnel):
    """Raise ValueError unless the npz funnel host/guest/axis equal those of ``funnel``.

    Parameters
    ----------
    npz_path : learned_cv.npz from learn_cv
    funnel : funnel_params dict (``host_index``, ``guest_index``, ``unbinding_axis``)
    """
    with np.load(npz_path, allow_pickle=False) as d:
        host, guest, axis = d["funnel_host"], d["funnel_guest"], np.asarray(d["funnel_axis"], float)
    f_axis = np.asarray(funnel["unbinding_axis"], float)
    same = (list(host) == [int(i) for i in funnel["host_index"]]
            and list(guest) == [int(i) for i in funnel["guest_index"]]
            and np.allclose(axis / np.linalg.norm(axis), f_axis / np.linalg.norm(f_axis)))
    if not same:
        raise ValueError("learned CV funnel differs from --funnel-from: not supported")


def _sw_spec(d, system, kept, n_plain_cvs=0, fused=True):
    """[CVSpec] of s_w from a rupture-class npz (fused forces, or residual composite when residualised)."""
    from ..BoundaryCorrection import MAX_CVS, LinearComposite, ResidualComposite
    from ..CV import CVSpec
    w, mu, sd = (np.array(d[k], dtype=float) for k in ("sw_weights", "mu", "sd"))
    if fused and bool(d["residualised"]):
        raise ValueError("fused s_w does not support residualised features; pass fused=False")
    n_feat = len(fused_sw_forces(kept, w, mu, sd, system)[0]) if fused else len(kept)
    n_cvs = n_feat + bool(d["residualised"]) + n_plain_cvs
    if n_cvs > MAX_CVS:
        raise ValueError(f"{n_cvs} collective variables ({n_feat} feature forces, the residual z, "
                         f"{n_plain_cvs} plain) exceed OpenMM's limit of {MAX_CVS}: not supported")
    lo, hi, hw = (float(d[k]) for k in ("sw_grid_min", "sw_grid_max", "sw_hill_width"))
    edges = (float(d["sw_lo"]), float(d["sw_hi"]))       # training range: boundary-corrected edges
    resid = bool(d["residualised"])
    if resid:
        host, guest = [int(i) for i in d["funnel_host"]], [int(i) for i in d["funnel_guest"]]
        axis = np.array(d["funnel_axis"], dtype=float)
        z_min, z_max, tables = float(d["z_min"]), float(d["z_max"]), np.array(d["z_tables"], dtype=float)

    def factory(pos, n_atoms, top):
        if fused:
            return fused_sw_composite(kept, w, mu, sd, system)
        forces = [feature_cv(f, system, f"f{k}") for k, f in enumerate(kept)]
        if not resid:
            return LinearComposite(forces, w, mu, sd)
        return ResidualComposite(forces, w, mu, sd, z_force=funnel_z_force(host, guest, axis),
                                 z_min=z_min, z_max=z_max, tables=tables)

    return [CVSpec(cv=None, grid_min=lo, grid_max=hi, hill_width=hw,
                   grid_points=int(round((hi - lo) / (hw / 5))) + 1,
                   name="learned_sw", bounds=edges, _factory=factory)]


def learned_cv_spec(npz_path, system, n_components=1, nested=False, n_plain_cvs=0, fused=True):
    """One CVSpec per HLDA component. Flat LinearComposite by default (components share
    their features); ``nested=True`` gives the old MetaCollectiveVariable (parity tests only).
    A rupture-class npz (key "component") gives the single s_w spec "learned_sw",
    boundary-corrected on (sw_lo, sw_hi); ``n_plain_cvs`` counts the other biased variables'
    CVs against the CustomCVForce limit. ``fused`` builds s_w from one force per feature kind
    (``fused_sw_forces``); a residualised npz needs ``fused=False``.
    """
    from ..CV import CVSpec
    from .features import features_from_dict
    d = np.load(npz_path, allow_pickle=False)
    feats = features_from_dict(d)
    kept = [feats[int(i)] for i in d["kept"]]
    if "component" in d.files:
        if n_components != 1 or nested:
            raise ValueError("a rupture-class learned CV has one biased component (s_w); "
                             f"n_components={n_components}, nested={nested} is not supported")
        return _sw_spec(d, system, kept, n_plain_cvs, fused)
    key = object()                                  # shared by this call's components
    specs = []
    for k in range(n_components):
        w, mu, sd = d["weights"][:, k], d["mu"], d["sd"]
        lo, hi, hw = float(d["grid_min"][k]), float(d["grid_max"][k]), float(d["hill_width"][k])

        if nested:
            def factory(pos, n_atoms, top, w=w, name=f"learned{k + 1}"):
                return combined_cv(kept, w, mu, sd, system, name=name)
        else:
            def factory(pos, n_atoms, top, w=w):
                return composite_cv(kept, w, mu, sd, system, key=key)

        specs.append(CVSpec(cv=None, grid_min=lo, grid_max=hi, hill_width=hw,
                            grid_points=int(round((hi - lo) / (hw / 5))) + 1,
                            name=f"learned{k + 1}", bounds=(-np.inf, np.inf), _factory=factory))
    return specs


def tt_basins_from_classes(npz_path, specs):
    """Transition-tempering basins [B, U] from a rupture-class npz, as (funnel z nm, s_w) points.

    ``specs`` is the bias-variable order; only [funnel_z, learned_sw] is supported.
    """
    names = [sp.name for sp in specs]
    if names != ["funnel_z", "learned_sw"]:
        raise ValueError(f"class basins are (funnel z, s_w) points; bias variables {names} not supported")
    with np.load(npz_path, allow_pickle=False) as d:
        if "basin_B" not in d.files or "basin_U" not in d.files:
            raise ValueError(f"{npz_path} has no class basins: not supported")
        return [[float(x) for x in d["basin_B"]], [float(x) for x in d["basin_U"]]]
