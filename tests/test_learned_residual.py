"""Residualised s_w: numpy == OpenMM, and learned_cv_spec on rupture-class npz files."""
import sys
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, PDBFile, Topology
from openmm.app.metadynamics import BiasVariable

from autopath.metadynamics import (BoundaryCorrectedMetadynamics, LinearComposite, MetadynamicsMD,
                                   ResidualComposite)
from autopath.metadynamics.BoundaryCorrection import boundary_factor, composite_bias_force
from autopath.metadynamics.learned.features import features_to_dict
from autopath.metadynamics.learned.forces import feature_cv, funnel_z_force, is_rupture_cv, learned_cv_spec

from .learned_toys import (AXIS, FEATS, GUEST, HOST, MASSES, MU, SD, TABLES, W, X0, ZMAX, ZMIN, s_plain,
                           s_ref, toy_system, write_npz_r2, z_of)

REF = mm.Platform.getPlatformByName("Reference")


def residual(s):
    forces = [feature_cv(f, s, f"f{k}") for k, f in enumerate(FEATS)]
    return ResidualComposite(forces, W, MU, SD, z_force=funnel_z_force(HOST, GUEST, AXIS),
                             z_min=ZMIN, z_max=ZMAX, tables=TABLES)


def _energy_context(X):
    s = toy_system()
    force, _ = composite_bias_force([SimpleNamespace(force=residual(s), periodic=False)])
    force.setEnergyFunction(force.getEnergyFunction().replace("table(s0)", "s0", 1))   # energy = s_w
    force.setForceGroup(5)
    s.addForce(force)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(X)
    return ctx


def _e(ctx):
    st = ctx.getState(getEnergy=True, getForces=True, groups={5})
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))


def _moved(dz):
    X = X0.copy()
    X[GUEST] += dz * AXIS / np.linalg.norm(AXIS)
    return X


def test_toy_masses_are_unequal_and_z_is_mass_weighted():
    assert len(set(MASSES)) == len(MASSES)
    X = X0.copy()
    X[0] += 0.5 * AXIS / np.linalg.norm(AXIS)               # moving one guest atom: COM shifts by m0/(m0+m1)
    assert z_of(X) - z_of(X0) == pytest.approx(0.5 * MASSES[0] / (MASSES[0] + MASSES[1]))


@pytest.mark.parametrize("dz", [0.0, 0.12, -0.2, 0.8, -0.9])     # last two: z outside the table, clamped
def test_residual_composite_energy_matches_numpy(dz):
    X = _moved(dz)
    assert (z_of(X) < ZMIN or z_of(X) > ZMAX) == (abs(dz) > 0.5)
    e, _ = _e(_energy_context(X))
    assert e == pytest.approx(s_ref(X), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("dz", [0.12, -0.9, 0.8])
def test_residual_composite_forces_match_numpy_gradient(dz):
    X = _moved(dz)
    _, F = _e(_energy_context(X))
    h = 1e-5
    for i, a in [(0, 0), (1, 2), (3, 1), (4, 0)]:
        Xp, Xm = X.copy(), X.copy()
        Xp[i, a] += h
        Xm[i, a] -= h
        assert F[i, a] == pytest.approx(-(s_ref(Xp) - s_ref(Xm)) / (2 * h), rel=1e-4, abs=1e-4)


def test_metad_records_the_numpy_value():
    s = toy_system()
    c = s_ref(X0)
    meta = BoundaryCorrectedMetadynamics(s, [BiasVariable(residual(s), c - 3.0, c + 3.0, 0.2, gridWidth=151)],
                                         300 * u.kelvin, 10, 1.0, 10)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(X0)
    assert meta._force.getNumCollectiveVariables() == len(FEATS) + 1
    assert list(meta._cv_values(ctx)) == pytest.approx([c], rel=1e-6)


def test_residual_composite_validates_inputs():
    s = toy_system()
    forces = [feature_cv(f, s, f"f{k}") for k, f in enumerate(FEATS)]
    zf = funnel_z_force(HOST, GUEST, AXIS)
    with pytest.raises(ValueError, match="z_force"):
        ResidualComposite(forces, W, MU, SD, z_min=ZMIN, z_max=ZMAX, tables=TABLES)
    with pytest.raises(ValueError, match="one table row"):
        ResidualComposite(forces, W, MU, SD, z_force=zf, z_min=ZMIN, z_max=ZMAX, tables=TABLES[:2])
    with pytest.raises(ValueError, match="z_max"):
        ResidualComposite(forces, W, MU, SD, z_force=zf, z_min=ZMAX, z_max=ZMIN, tables=TABLES)


def _write_npz_r1(path):
    np.savez(path, kept=np.arange(3), weights=np.c_[W, -W], mu=MU, sd=SD, grid_min=[-3.0, -3.0],
             grid_max=[3.0, 3.0], hill_width=[0.2, 0.2], **features_to_dict(FEATS))
    return path


@pytest.mark.parametrize("residualised", [True, False])
def test_learned_cv_spec_builds_s_w_from_a_rupture_npz(tmp_path, residualised):
    specs = learned_cv_spec(write_npz_r2(str(tmp_path / "r2.npz"), residualised), toy_system(), 1)
    c = s_ref(X0) if residualised else s_plain(X0)
    assert len(specs) == 1 and specs[0].name == "learned_sw"
    assert specs[0].bounds == pytest.approx((c - 2.0, c + 2.0))           # boundary correction on
    assert (specs[0].grid_min, specs[0].grid_max) == pytest.approx((c - 2.6, c + 2.6))
    assert specs[0].hill_width == 0.2 and specs[0].grid_points == 131      # spacing hw / 5
    cv = specs[0].resolve(None, 5, None).cv
    assert isinstance(cv, LinearComposite) and isinstance(cv, ResidualComposite) == residualised


def test_rupture_npz_rejects_two_components_and_nested(tmp_path):
    path = write_npz_r2(str(tmp_path / "r2.npz"))
    with pytest.raises(ValueError, match="not supported"):
        learned_cv_spec(path, toy_system(), 2)
    with pytest.raises(ValueError, match="not supported"):
        learned_cv_spec(path, toy_system(), 1, nested=True)


def test_md_run_applies_the_boundary_factor_to_the_learned_sw_bias(tmp_path, monkeypatch):
    top = Topology()
    res = top.addResidue("UNK", top.addChain())
    for i in range(5):
        top.addAtom(f"C{i}", Element.getBySymbol("C"), res)
    top.setPeriodicBoxVectors(np.eye(3) * 10.0 * u.nanometer)
    pdb = str(tmp_path / "toy.pdb")
    with open(pdb, "w") as fh:
        PDBFile.writeFile(top, X0 * u.nanometer, fh)
    made = []

    class Spy(BoundaryCorrectedMetadynamics):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(sys.modules[MetadynamicsMD.__module__], "BoundaryCorrectedMetadynamics", Spy)
    s = toy_system()
    specs = learned_cv_spec(write_npz_r2(str(tmp_path / "r2.npz")), s, 1)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[1], platform="Reference",
                        out_dir=str(tmp_path / "out"))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=specs, mMD_time=0.02, bias_factor=10,
           hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), rct_ustride=1)
    meta, = made
    var = meta.variables[0]
    c = s_ref(X0)
    assert meta.bounds == [pytest.approx((c - 2.0, c + 2.0))]
    assert meta.boundaryFactor.max() > 1.5                         # factor is not trivially 1
    assert np.array_equal(meta.boundaryFactor, boundary_factor([var], meta.bounds))
    grid = np.linspace(var.minValue, var.maxValue, var.gridWidth)
    expect = sum(h * np.exp(-0.5 * (grid - p) ** 2 / 0.2 ** 2) for _, p, h in meta.hills) * meta.boundaryFactor
    assert len(meta.hills) == 10 and np.allclose(meta._selfBias, expect)


def test_is_rupture_cv(tmp_path):
    assert is_rupture_cv(write_npz_r2(str(tmp_path / "r2.npz")))
    assert not is_rupture_cv(_write_npz_r1(str(tmp_path / "r1.npz")))
