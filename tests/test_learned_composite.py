"""Flattened learned CV (LinearComposite) vs the nested MetaCollectiveVariable bias."""
import os
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, PDBFile, Topology
from openmm.app.metadynamics import BiasVariable

from autopath.metadynamics import (BoundaryCorrectedMetadynamics, CVSpec, LinearComposite, MetadynamicsMD,
                                   TransitionTemperedMetadynamics)
from autopath.metadynamics.learned.features import Feature, evaluate, features_to_dict
from autopath.metadynamics.learned.forces import combined_cv, composite_cv, learned_cv_spec

from .learned_toys import toy_system

T = 300.0
TOY = [Feature("p", "pair", "F1", pairs=[(0, 1)], r0=0.35),
       Feature("c", "contacts", "F2", group_a=[0, 1], group_b=[2, 3, 4], r0=0.3),
       Feature("d", "centroid_distance", "F3", groups=[[0, 1], [2, 3]]),
       Feature("w", "centroid_count", "F5", groups=[[0, 1]], group_b=[2, 3, 4], r0=0.5),
       Feature("o", "cos", "F8", groups=[[0], [1], [2], [3]])]
RNG = np.random.default_rng(7)
X0 = RNG.uniform(1.5, 2.6, size=(5, 3))
W = RNG.normal(size=(5, 2))
MU = RNG.normal(size=5)
SD = RNG.uniform(0.5, 2, size=5)


def _system(box=5.0):
    return toy_system(X0, box, [12.0] * 5)


def _s(X, k):
    return float(np.sum(W[:, k] * (evaluate(TOY, X[None])[0] - MU) / SD))


def _var(cv, k):
    c = _s(X0, k)
    return BiasVariable(cv, c - 3.0, c + 3.0, 0.2, gridWidth=151)


def _meta(kind, n_comp=1, cls=BoundaryCorrectedMetadynamics, **kw):
    s = _system()
    if kind == "nested":
        cvs = [combined_cv(TOY, W[:, k], MU, SD, s, name=f"learned{k + 1}") for k in range(n_comp)]
    else:
        key = object()
        cvs = [composite_cv(TOY, W[:, k], MU, SD, s, key=key) for k in range(n_comp)]
    meta = cls(s, [_var(cv, k) for k, cv in enumerate(cvs)], T * u.kelvin, 10, 1.0, 10, **kw)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(X0)
    return meta, ctx


def _deposit(meta, ctx, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(6):
        p = [rng.uniform(v.minValue, v.maxValue) for v in meta.variables]
        meta._addGaussian(p, rng.uniform(0.5, 2.0) * u.kilojoule_per_mole, ctx)
    # one hill on the walker so the bias has a slope there
    meta._addGaussian(list(meta._cv_values(ctx)), 2.0 * u.kilojoule_per_mole, ctx)


def _bias(meta, ctx):
    st = ctx.getState(getEnergy=True, getForces=True, groups={meta._force.getForceGroup()})
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))


@pytest.mark.parametrize("n_comp", [1, 2])
def test_flat_matches_nested_values_energy_and_forces(n_comp):
    nest, cn = _meta("nested", n_comp)
    flat, cf = _meta("flat", n_comp)
    vn, vf = list(nest._cv_values(cn)), list(flat._cv_values(cf))
    assert vf == pytest.approx(vn, rel=1e-9, abs=1e-12)
    assert vf == pytest.approx([_s(X0, k) for k in range(n_comp)], rel=1e-6)
    _deposit(nest, cn)
    _deposit(flat, cf)
    assert np.allclose(nest._totalBias, flat._totalBias, rtol=1e-12, atol=1e-12)
    en, fn = _bias(nest, cn)
    ef, ff = _bias(flat, cf)
    assert abs(en) > 0.1 and np.abs(fn).max() > 0.1
    assert ef == pytest.approx(en, rel=1e-8)
    assert np.allclose(ff, fn, rtol=1e-6, atol=1e-8)


def test_two_components_add_each_feature_once():
    flat, ctx = _meta("flat", 2)
    assert flat._force.getNumCollectiveVariables() == len(TOY)
    assert list(flat.getCollectiveVariables(SimpleNamespace(context=ctx))) == \
        pytest.approx([_s(X0, 0), _s(X0, 1)], rel=1e-6)


def test_unshared_composites_and_a_plain_variable_in_one_force():
    s = _system()
    plain = mm.CustomExternalForce("x")
    plain.addParticle(0, [])
    variables = [BiasVariable(plain, 0.0, 5.0, 0.1, gridWidth=101),
                 _var(composite_cv(TOY, W[:, 1], MU, SD, s), 1)]
    meta = BoundaryCorrectedMetadynamics(s, variables, T * u.kelvin, 10, 1.0, 10)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(X0)
    assert meta._force.getNumCollectiveVariables() == 1 + len(TOY)
    assert list(meta._cv_values(ctx)) == pytest.approx([X0[0, 0], _s(X0, 1)], rel=1e-6)


def test_composite_rejects_periodic_and_mismatched_lengths():
    s = _system()
    c = composite_cv(TOY, W[:, 0], MU, SD, s)
    with pytest.raises(ValueError, match="periodic"):
        BoundaryCorrectedMetadynamics(s, [BiasVariable(c, -1.0, 1.0, 0.1, periodic=True, gridWidth=21)],
                                      T * u.kelvin, 10, 1.0, 10)
    with pytest.raises(ValueError, match="one weight"):
        LinearComposite(c.forces, W[:2, 0], MU, SD)


def test_tt_steps_deposit_the_composite_value():
    flat, ctx = _meta("flat", 1, cls=TransitionTemperedMetadynamics,
                      basins=[[_s(X0, 0) - 1.0], [_s(X0, 0) + 1.0]])
    nest, cn = _meta("nested", 1, cls=TransitionTemperedMetadynamics,
                     basins=[[_s(X0, 0) - 1.0], [_s(X0, 0) + 1.0]])
    for m, c in ((flat, ctx), (nest, cn)):
        sim = SimpleNamespace(context=c, currentStep=0)

        def step(n, sim=sim):
            sim.context.getIntegrator().step(n)
            sim.currentStep += n
        sim.step = step
        c.setVelocitiesToTemperature(T * u.kelvin, 1)
        m.step(sim, 50)
    hf, hn = np.array(flat.hills), np.array(nest.hills)
    assert hf.shape == hn.shape == (5, 3)
    assert np.allclose(hf, hn, rtol=1e-6, atol=1e-9)


def _npz(tmp_path):
    lo = [_s(X0, k) - 3.0 for k in range(2)]
    path = str(tmp_path / "cv.npz")
    np.savez(path, kept=np.arange(5), weights=W, mu=MU, sd=SD, grid_min=lo,
             grid_max=[a + 6.0 for a in lo], hill_width=[0.2, 0.2], **features_to_dict(TOY))
    return path


def test_learned_cv_spec_default_is_flat_and_nested_on_request(tmp_path):
    s = _system()
    flat = [sp.resolve(None, 5, None).cv for sp in learned_cv_spec(_npz(tmp_path), s, 2)]
    nest = [sp.resolve(None, 5, None).cv for sp in learned_cv_spec(_npz(tmp_path), s, 2, nested=True)]
    assert all(isinstance(c, LinearComposite) for c in flat) and flat[0].key is flat[1].key
    assert not any(isinstance(c, LinearComposite) for c in nest)


def test_md_run_records_the_composite_in_colvar_and_hills(tmp_path):
    top = Topology()
    res = top.addResidue("UNK", top.addChain())
    for i in range(5):
        top.addAtom(f"C{i}", Element.getBySymbol("C"), res)
    top.setPeriodicBoxVectors(np.eye(3) * 5.0 * u.nanometer)
    pdb = str(tmp_path / "toy.pdb")
    with open(pdb, "w") as fh:
        PDBFile.writeFile(top, X0 * u.nanometer, fh)
    s = _system()
    specs = learned_cv_spec(_npz(tmp_path), s, 2)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[1], platform="Reference",
                        out_dir=str(tmp_path / "out"))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=specs, mMD_time=0.02, bias_factor=10,
           hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), rct_ustride=1)
    colvar = np.load(tmp_path / "out" / "COLVAR_w.npy")
    hills = np.load(tmp_path / "out" / "HILLS_w.npz")
    assert colvar.shape == (11, 2) and hills["cv"].shape == (10, 2)
    assert np.array_equal(colvar[1:], hills["cv"])           # the hill sits at the recorded s
    assert colvar[0] == pytest.approx([_s(np.round(X0 * 10, 3) / 10, k) for k in range(2)], rel=1e-4)


XML3 = "examples/04_Full-Pipeline/3ptb/milestones/milestone_1_frame_0_relax_system.xml"
STATE3 = "examples/04_Full-Pipeline/3ptb/milestones/milestone_1_frame_0_relax_state.xml"
PDB3 = "examples/04_Full-Pipeline/3ptb/system.pdb"


@pytest.mark.learned_slow
@pytest.mark.skipif(not all(os.path.exists(p) for p in (XML3, STATE3, PDB3)), reason="3ptb data not on disk")
def test_flat_matches_nested_on_a_3ptb_frame():
    import MDAnalysis as mda
    from autopath.metadynamics.learned.define import define_features
    st = mm.XmlSerializer.deserialize(open(STATE3).read())
    X = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    uni = mda.Universe(PDB3)
    lig = list(uni.select_atoms("resname UNK").indices)
    pocket = list(uni.select_atoms("protein and around 6 resname UNK").residues.atoms.indices)
    feats = define_features(PDB3, "examples/data/3ptb.sdf", lig, pocket, bound_X=X[None], exit_X=X[None])
    feats = [f for f in feats if f.family in ("F1", "F3", "F4", "F5")][:12]
    rng = np.random.default_rng(0)
    w = rng.normal(size=len(feats))
    v = evaluate(feats, X[None])[0]
    mu, sd = v + rng.normal(scale=0.1, size=len(feats)), rng.uniform(0.5, 2.0, size=len(feats))
    out = {}
    for kind in ("nested", "flat"):
        s = mm.XmlSerializer.deserialize(open(XML3).read())
        cv = (combined_cv(feats, w, mu, sd, s) if kind == "nested"
              else composite_cv(feats, w, mu, sd, s))
        c = float(np.sum(w * (v - mu) / sd))
        meta = BoundaryCorrectedMetadynamics(s, [BiasVariable(cv, c - 2.0, c + 2.0, 0.1, gridWidth=201)],
                                             T * u.kelvin, 10, 1.0, 10)
        ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
        ctx.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
        ctx.setPositions(st.getPositions())
        _deposit(meta, ctx)
        out[kind] = (list(meta._cv_values(ctx)), *_bias(meta, ctx))
        del ctx
    (vn, en, fn), (vf, ef, ff) = out["nested"], out["flat"]
    assert vf == pytest.approx(vn, rel=1e-6, abs=1e-6)
    assert ef == pytest.approx(en, rel=1e-5, abs=1e-5)
    assert np.allclose(ff, fn, rtol=1e-4, atol=1e-3)
