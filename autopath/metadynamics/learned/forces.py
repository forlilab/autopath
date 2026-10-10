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


def _sw_spec(d, system, kept):
    """[CVSpec] of s_w from a rupture-class npz (residual composite when residualised)."""
    from ..BoundaryCorrection import LinearComposite, ResidualComposite
    from ..CV import CVSpec
    w, mu, sd = (np.array(d[k], dtype=float) for k in ("sw_weights", "mu", "sd"))
    lo, hi, hw = (float(d[k]) for k in ("sw_grid_min", "sw_grid_max", "sw_hill_width"))
    edges = (float(d["sw_lo"]), float(d["sw_hi"]))       # training range: boundary-corrected edges
    resid = bool(d["residualised"])
    if resid:
        host, guest = [int(i) for i in d["funnel_host"]], [int(i) for i in d["funnel_guest"]]
        axis = np.array(d["funnel_axis"], dtype=float)
        z_min, z_max, tables = float(d["z_min"]), float(d["z_max"]), np.array(d["z_tables"], dtype=float)

    def factory(pos, n_atoms, top):
        forces = [feature_cv(f, system, f"f{k}") for k, f in enumerate(kept)]
        if not resid:
            return LinearComposite(forces, w, mu, sd)
        return ResidualComposite(forces, w, mu, sd, z_force=funnel_z_force(host, guest, axis),
                                 z_min=z_min, z_max=z_max, tables=tables)

    return [CVSpec(cv=None, grid_min=lo, grid_max=hi, hill_width=hw,
                   grid_points=int(round((hi - lo) / (hw / 5))) + 1,
                   name="learned_sw", bounds=edges, _factory=factory)]


def learned_cv_spec(npz_path, system, n_components=1, nested=False):
    """One CVSpec per HLDA component. Flat LinearComposite by default (components share
    their features); ``nested=True`` gives the old MetaCollectiveVariable (parity tests only).
    A rupture-class npz (key "component") gives the single s_w spec "learned_sw",
    boundary-corrected on (sw_lo, sw_hi).
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
        return _sw_spec(d, system, kept)
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
