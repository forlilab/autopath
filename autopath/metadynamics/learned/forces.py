# autopath/metadynamics/learned/forces.py
"""OpenMM/cvpack builders for learned-CV features; same formulas as features.evaluate."""
import cvpack
import numpy as np
import openmm as mm
from openmm import unit

from .features import CONTACT_CUTOFF_FACTOR, CONTACT_SWITCH_FACTOR


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
        return _wrap(_centroid(f"1/(1+(distance(g1,g2)/{f.r0})^8)", groups, 2, bonds), name)
    if f.kind == "cos":
        # a·b from centroid distances; a = c2-c1, b = c4-c3
        expr = ("(distance(g2,g3)^2 + distance(g1,g4)^2 - distance(g2,g4)^2 - distance(g1,g3)^2)"
                "/(2*distance(g1,g2)*distance(g3,g4))")
        return _wrap(_centroid(expr, f.groups), name)
    raise ValueError(f"unknown feature kind {f.kind!r}")


def combined_cv(features, weights, mu, sd, system, name="learned"):
    names = [f"f{k}" for k in range(len(features))]
    expr = " + ".join(f"w{k}*({n} - mu{k})/sd{k}" for k, n in enumerate(names))
    params = {}
    for k in range(len(features)):
        params[f"w{k}"], params[f"mu{k}"], params[f"sd{k}"] = float(weights[k]), float(mu[k]), float(sd[k])
    inner = [feature_cv(f, system, n) for f, n in zip(features, names)]
    return cvpack.MetaCollectiveVariable(expr, inner, unit.dimensionless, name=name, **params)
