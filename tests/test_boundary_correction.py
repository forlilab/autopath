import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app.metadynamics import BiasVariable, Metadynamics

from autopath.metadynamics import BoundaryCorrectedMetadynamics, MetadynamicsAnalysis
from autopath.metadynamics.BoundaryCorrection import (
    boundary_factor, inside_fraction, smooth_floor)
from autopath.metadynamics.CV import CVSpec, funnel_z_cv

LO, HI, SIGMA = 0.1, 1.0, 0.05


def _toy(corrected, bias_dir=None):
    system = mm.System()
    system.addParticle(12.0)
    cv = mm.CustomExternalForce(f"min(max(x, 0), {HI})")
    cv.addParticle(0, [])
    bv = BiasVariable(cv, 0.0, HI, SIGMA, gridWidth=101)
    kw = dict(saveFrequency=10, biasDir=str(bias_dir)) if bias_dir else {}
    args = (system, [bv], 300 * u.kelvin, 15, 1.0 * u.kilojoule_per_mole, 10)
    meta = (BoundaryCorrectedMetadynamics(*args, bounds=[(LO, HI)], **kw) if corrected
            else Metadynamics(*args, **kw))
    ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0.5, 0, 0]])
    return meta, ctx


def _fill_uniformly(meta, ctx):
    for c in np.arange(LO + 0.0025, HI, 0.005):  # midpoints: uniform sampling of [LO, HI]
        meta._addGaussian([c], 1.0 * u.kilojoule_per_mole, ctx)


def test_inside_fraction_is_half_at_an_edge():
    n = inside_fraction([LO, 0.55, HI, -1.0], LO, HI, SIGMA)
    assert n[0] == pytest.approx(0.5, abs=1e-6)
    assert n[1] == pytest.approx(1.0, abs=1e-6)
    assert n[2] == pytest.approx(0.5, abs=1e-6)
    assert n[3] < 1e-12


def test_smooth_floor_is_c1_and_finite():
    a = 0.5
    x = np.linspace(0, 1, 2001)
    f = smooth_floor(x, a)
    assert np.all(f[x >= a] == x[x >= a])
    assert f[0] == pytest.approx(a / 2)
    assert np.all(np.diff(f) >= 0)
    slope = np.gradient(f, x)
    i = np.searchsorted(x, a)
    assert slope[i - 2] == pytest.approx(slope[i + 2], abs=5e-3)


def test_factor_corrects_edges_and_stays_finite_outside():
    bv = BiasVariable(mm.CustomExternalForce("x"), 0.0, HI, SIGMA, gridWidth=101)
    c = boundary_factor([bv], [(LO, HI)])
    grid = np.linspace(0, HI, 101)
    assert c[np.argmin(abs(grid - 0.55))] == pytest.approx(1.0, abs=1e-6)
    assert c[np.argmin(abs(grid - LO))] == pytest.approx(2.0, abs=1e-6)
    assert c[-1] == pytest.approx(2.0, abs=1e-6)
    assert c[0] <= 4.0 + 1e-9


def test_factor_on_a_2d_corner_and_unbounded_axes():
    a = BiasVariable(mm.CustomExternalForce("x"), 0.0, 1.0, SIGMA, gridWidth=51)
    b = BiasVariable(mm.CustomExternalForce("y"), 0.0, 1.0, SIGMA, gridWidth=41)
    c = boundary_factor([a, b], [(0.0, 1.0), (0.0, 1.0)])
    assert c.shape == (41, 51)
    assert c[0, 0] == pytest.approx(4.0, abs=1e-6)
    assert c[20, 25] == pytest.approx(1.0, abs=1e-6)
    assert np.all(boundary_factor([a, b], [None, None]) == 1.0)


def test_uniform_hills_give_a_flat_bias_up_to_the_walls():
    plain, ctx_p = _toy(False)
    corr, ctx_c = _toy(True)
    _fill_uniformly(plain, ctx_p)
    _fill_uniformly(corr, ctx_c)
    grid = np.linspace(0, HI, 101)
    d = (grid >= LO - 1e-9) & (grid <= HI + 1e-9)
    mid = plain._totalBias[np.argmin(abs(grid - 0.55))]
    assert plain._totalBias[d].min() / mid == pytest.approx(0.5, abs=0.02)
    flat = corr._totalBias[d] / corr._totalBias[np.argmin(abs(grid - 0.55))]
    assert np.all(abs(flat - 1.0) < 0.01)


def test_force_applies_the_corrected_bias():
    meta, ctx = _toy(True)
    meta._addGaussian([HI], 1.0 * u.kilojoule_per_mole, ctx)
    ctx.setPositions([[HI, 0, 0]])
    group = meta._force.getForceGroup()
    e = ctx.getState(getEnergy=True, groups={group}).getPotentialEnergy()
    assert e.value_in_unit(u.kilojoule_per_mole) == pytest.approx(2.0, rel=1e-4)


def test_bias_files_carry_the_corrected_bias(tmp_path):
    meta, ctx = _toy(True, bias_dir=tmp_path)
    _fill_uniformly(meta, ctx)
    meta._syncWithDisk()
    biases = list(MetadynamicsAnalysis.load_bias_files(str(tmp_path)).values())
    _, fes = MetadynamicsAnalysis.compute_fes(biases, 300, 15)
    assert np.allclose(fes, meta.getFreeEnergy().value_in_unit(u.kilojoule_per_mole))


def test_bounds_survive_resolve():
    spec = funnel_z_cv([0], [1], [0, 0, 1], -0.7, 2.65, bounds=(-0.39, 2.65))
    assert spec.resolve(None, 2, None).bounds == (-0.39, 2.65)
    assert CVSpec(cv=None, grid_min=0, grid_max=1, hill_width=0.05,
                  grid_points=10).bounds is None
