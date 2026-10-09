# tests/test_learned_forces.py
import os

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest

from autopath.metadynamics.learned.features import Feature, evaluate
from autopath.metadynamics.learned.forces import combined_cv, feature_cv


def _toy_system(n, box=5.0):
    s = mm.System()
    for _ in range(n):
        s.addParticle(1.0)
    s.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.2)
    for _ in range(n):
        nb.addParticle(0.0, 0.3, 0.0)
    s.addForce(nb)
    return s


def _value(cv, system, X):
    probe = mm.System()
    for i in range(system.getNumParticles()):
        probe.addParticle(system.getParticleMass(i))
    probe.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    probe.addForce(cv)
    ctx = mm.Context(probe, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(X)
    return cv.getValue(ctx)._value


TOY = [Feature("p", "pair", "F1", pairs=[(0, 1)], r0=0.35),
       Feature("c", "contacts", "F2", group_a=[0, 1], group_b=[2, 3, 4], r0=0.3),
       Feature("d", "centroid_distance", "F3", groups=[[0, 1], [2, 3]]),
       Feature("w", "centroid_count", "F5", groups=[[0, 1]], group_b=[2, 3, 4], r0=0.5),
       Feature("o", "cos", "F8", groups=[[0], [1], [2], [3]])]


@pytest.mark.parametrize("f", TOY, ids=lambda f: f.kind)
def test_each_kind_matches_numpy(f):
    rng = np.random.default_rng(0)
    X = rng.uniform(1.5, 2.6, size=(5, 3))
    system = _toy_system(5)
    want = evaluate([f], X[None])[0, 0]
    got = _value(feature_cv(f, system, name=f.name), system, X)
    assert got == pytest.approx(want, rel=1e-5, abs=1e-6)


def test_centroid_count_rejects_cutoff_beyond_minimum_image():
    f = Feature("w", "centroid_count", "F5", groups=[[0, 1]], group_b=[2, 3, 4], r0=1.1)
    with pytest.raises(ValueError, match="half"):
        feature_cv(f, _toy_system(5, box=5.0), name="w")


def test_combined_cv_is_the_standardised_weighted_sum():
    rng = np.random.default_rng(1)
    X = rng.uniform(1.5, 2.6, size=(5, 3))
    system = _toy_system(5)
    v = evaluate(TOY, X[None])[0]
    w, mu, sd = rng.normal(size=5), rng.normal(size=5), rng.uniform(0.5, 2, size=5)
    got = _value(combined_cv(TOY, w, mu, sd, system), system, X)
    assert got == pytest.approx(float(np.sum(w * (v - mu) / sd)), rel=1e-5)


PDB = "examples/04_Full-Pipeline/3ptb/system.pdb"
XML = "examples/04_Full-Pipeline/3ptb/system.xml"
DCD = "examples/04_Full-Pipeline/3ptb/sMD/trajectories/sMD_replica-164131_v0.01_forward_aligned.dcd"


@pytest.mark.skipif(not all(os.path.exists(p) for p in (PDB, XML, DCD)), reason="3ptb data not on disk")
def test_parity_on_3ptb_frames():
    import MDAnalysis as mda
    from autopath.metadynamics.learned.define import define_features
    uni = mda.Universe(PDB, DCD)
    lig = list(uni.select_atoms("resname UNK").indices)
    pocket = list(uni.select_atoms("protein and around 6 resname UNK").residues.atoms.indices)
    uni.trajectory[0]
    X0 = uni.atoms.positions[None] / 10.0
    feats = define_features(PDB, "examples/data/3ptb.sdf", lig, pocket, bound_X=X0, exit_X=X0)
    system = mm.XmlSerializer.deserialize(open(XML).read())
    for k in (0, 150, 250):
        uni.trajectory[k]
        X = uni.atoms.positions / 10.0
        want = evaluate(feats, X[None])[0]
        for f, wv in zip(feats, want):
            got = _value(feature_cv(f, system, name=f.name), system, X)
            tol = dict(rel=2e-3, abs=2e-3) if f.family in ("F4", "F5", "F6") else dict(rel=1e-5, abs=1e-6)
            assert got == pytest.approx(wv, **tol), (f.name, k)
