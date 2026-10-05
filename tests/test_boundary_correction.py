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


def _toy2d(corrected):
    system = mm.System()
    system.addParticle(12.0)
    fx, fy = mm.CustomExternalForce("x"), mm.CustomExternalForce("y")
    fx.addParticle(0, [])
    fy.addParticle(0, [])
    # unequal grids, widths and ranges so a transposed or mis-multiplied factor fails
    bx = BiasVariable(fx, 0.0, 1.0, 0.05, gridWidth=51)
    by = BiasVariable(fy, 0.0, 2.0, 0.10, gridWidth=41)
    args = (system, [bx, by], 300 * u.kelvin, 15, 1.0 * u.kilojoule_per_mole, 10)
    meta = (BoundaryCorrectedMetadynamics(*args, bounds=[(0.1, 1.0), (0.0, 2.0)]) if corrected
            else Metadynamics(*args))
    ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0.5, 1.0, 0]])
    for x in np.arange(0.1 + 0.01, 1.0, 0.02):        # midpoints: uniform sampling of the box
        for y in np.arange(0.02, 2.0, 0.04):
            meta._addGaussian([x, y], 1.0 * u.kilojoule_per_mole, ctx)
    return meta


def test_uniform_2d_sampling_gives_a_flat_bias_into_the_corners():
    plain, corr = _toy2d(False), _toy2d(True)
    gx, gy = np.linspace(0, 1, 51), np.linspace(0, 2, 41)
    inside = (gy[:, None] >= 0) & (gx[None, :] >= 0.1 - 1e-9)
    assert corr._totalBias.shape == (41, 51)
    centre = (np.argmin(abs(gy - 1.0)), np.argmin(abs(gx - 0.55)))
    p = plain._totalBias / plain._totalBias[centre]
    c = corr._totalBias / corr._totalBias[centre]
    corner = (0, np.argmin(abs(gx - 0.1)))
    assert p[corner] == pytest.approx(0.25, abs=0.02)
    assert np.all(abs(c[inside] - 1.0) < 0.02)


def test_inside_fraction_matches_the_erratum_normalization():
    """McGovern & de Pablo erratum (JCP 140, 229901): Eq. 12 with prod sigma_i/(U_i-L_i)
    restored is the domain average of a unit hill; N(s) is that over the full-Gaussian value."""
    from scipy.integrate import dblquad
    from scipy.special import erf
    lo, hi, sig = np.array([-0.391, 0.0]), np.array([2.650, 1.0]), np.array([0.05, 0.08])

    def erratum_eq12(s):
        b = erf((s - lo) / (np.sqrt(2) * sig)) + erf((hi - s) / (np.sqrt(2) * sig))
        return (np.pi / 2) ** (len(s) / 2) * np.prod(sig / (hi - lo)) * np.prod(b)

    for s in ([-0.441, 0.5], [-0.391, 0.0], [1.0, 0.04], [2.65, 1.03], [1.2, 0.5]):
        s = np.array(s)
        avg, _ = dblquad(lambda y, x: np.exp(-((x - s[0]) / sig[0]) ** 2 / 2
                                             - ((y - s[1]) / sig[1]) ** 2 / 2),
                         lo[0], hi[0], lo[1], hi[1], epsabs=1e-12)
        avg /= np.prod(hi - lo)
        n = np.prod([inside_fraction([s[i]], lo[i], hi[i], sig[i])[0] for i in range(2)])
        assert erratum_eq12(s) == pytest.approx(avg, rel=1e-6)
        assert n * np.prod(np.sqrt(2 * np.pi) * sig / (hi - lo)) == pytest.approx(avg, rel=1e-6)


def test_funnel_z_corrects_only_the_cap():
    from autopath.metadynamics import funnel_z_bounds_nm
    lo, hi = funnel_z_bounds_nm({"z_max": 26.5 * u.angstrom})
    assert lo == -np.inf and hi == pytest.approx(2.65)


def test_no_amplification_past_a_soft_lower_wall():
    bv = BiasVariable(mm.CustomExternalForce("x"), 0.0, HI, SIGMA, gridWidth=101)
    c = boundary_factor([bv], [(-np.inf, HI)])
    grid = np.linspace(0, HI, 101)
    assert np.allclose(c[grid < HI - 4 * SIGMA], 1.0, atol=1e-3)
    assert c[-1] == pytest.approx(2.0, abs=1e-6)
    # a hill dropped inside the lower wall region matches stock OpenMM exactly
    plain, ctx_p = _toy(False)
    system = mm.System()
    system.addParticle(12.0)
    cv = mm.CustomExternalForce(f"min(max(x, 0), {HI})")
    cv.addParticle(0, [])
    corr = BoundaryCorrectedMetadynamics(
        system, [BiasVariable(cv, 0.0, HI, SIGMA, gridWidth=101)], 300 * u.kelvin, 15,
        1.0 * u.kilojoule_per_mole, 10, bounds=[(-np.inf, HI)])
    ctx_c = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx_c.setPositions([[0.5, 0, 0]])
    plain._addGaussian([0.05], 1.0 * u.kilojoule_per_mole, ctx_p)
    corr._addGaussian([0.05], 1.0 * u.kilojoule_per_mole, ctx_c)
    assert np.allclose(corr._totalBias, plain._totalBias)
