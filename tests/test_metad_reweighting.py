import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, Simulation, Topology
from openmm.app.metadynamics import BiasVariable, Metadynamics
from scipy.special import logsumexp

from autopath.metadynamics import BoundaryCorrectedMetadynamics, reweighted_fes, rct_weights

T = 300.0
KT = 0.0083144626 * T
A, KY = 6.0, 2000.0                     # cos(x) wells; y harmonic around a centre that moves with x


def _one_particle(expr):
    s = mm.System()
    s.addParticle(12.0)
    f = mm.CustomExternalForce(expr)
    f.addParticle(0, [])
    s.addForce(f)
    cv = mm.CustomExternalForce("min(max(x, 0), 1)")
    cv.addParticle(0, [])
    return s, BiasVariable(cv, 0.0, 1.0, 0.03, gridWidth=201)


def _context(meta_system):
    return mm.Context(meta_system, mm.VerletIntegrator(0.001),
                      mm.Platform.getPlatformByName("Reference"))


def test_rct_of_a_constant_bias_is_that_constant():
    s, bv = _one_particle("0")
    meta = BoundaryCorrectedMetadynamics(s, [bv], T * u.kelvin, 10, 1.0, 10, bounds=[None])
    assert meta.rct() == pytest.approx(0.0, abs=1e-9)
    meta._totalBias[:] = 7.5
    assert meta.rct() == pytest.approx(7.5, rel=1e-9)
    w = rct_weights(np.full(5, 7.5), np.full(5, 7.5), T)
    assert np.allclose(w, 0.2)


def test_plain_hills_match_stock_openmm_and_are_logged():
    s1, bv1 = _one_particle("0")
    s2, bv2 = _one_particle("0")
    ours = BoundaryCorrectedMetadynamics(s1, [bv1], T * u.kelvin, 10, 1.0, 10, bounds=[None])
    stock = Metadynamics(s2, [bv2], T * u.kelvin, 10, 1.0, 10)
    c1, c2 = _context(s1), _context(s2)
    for x, h in ((0.02, 1.0), (0.5, 0.7), (0.99, 0.4)):
        ours._addGaussian([x], h * u.kilojoule_per_mole, c1)
        stock._addGaussian([x], h * u.kilojoule_per_mole, c2)
    assert np.allclose(ours._totalBias, stock._totalBias)
    assert [row[1:] for row in ours.hills] == [(0.02, 1.0), (0.5, 0.7), (0.99, 0.4)]


def test_reweighting_recovers_the_biased_and_an_unbiased_coordinate():
    """WT-MetaD on x of U = A cos(2πx/0.45) + ½ky(y − 0.3 sin(2πx/0.9))²: the
    reweighted F(x) and F(y) must match the exact marginals."""
    pi = np.pi
    s, bv = _one_particle(
        f"{A}*cos(2*{pi}*x/0.45) + 0.5*{KY}*(y-0.3*sin(2*{pi}*x/0.9))^2"
        "+ 0.5*41840*(step(0.1-x)*(0.1-x)^2 + step(x-1)*(x-1)^2) + 1000*z^2")
    meta = BoundaryCorrectedMetadynamics(s, [bv], T * u.kelvin, 10, 1.0 * u.kilojoule_per_mole,
                                         250, bounds=[(0.1, 1.0)])
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 5 / u.picosecond, 0.004 * u.picosecond)
    integ.setRandomNumberSeed(1)
    top = Topology()
    top.addAtom("C", Element.getBySymbol("C"), top.addResidue("X", top.addChain()))
    sim = Simulation(top, s, integ, mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions([[0.3, 0, 0]])
    sim.context.setVelocitiesToTemperature(T * u.kelvin, 1)
    xs, ys, bias, rct = [], [], [], []
    for _ in range(1200):
        meta.step(sim, 2500)
        p = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        x, y = p.value_in_unit(u.nanometer)[0][:2]
        xs.append(x), ys.append(y), bias.append(meta.bias_energy(sim)), rct.append(meta.rct())
    xs, ys, bias, rct = map(np.array, (xs, ys, bias, rct))
    n0 = len(xs) // 5
    w = rct_weights(bias[n0:], rct[n0:], T)

    def rms(f, ref):
        ok = np.isfinite(f)
        d = (f - ref)[ok]
        return np.sqrt(np.mean((d - d.mean()) ** 2))

    xc, fx = reweighted_fes(xs[n0:], w, np.linspace(0.12, 0.98, 44), T)
    ux = A * np.cos(2 * pi * xc / 0.45)
    _, fx_biased = reweighted_fes(xs[n0:], np.full(len(w), 1 / len(w)), np.linspace(0.12, 0.98, 44), T)
    assert rms(fx, ux) < 1.5
    assert rms(fx_biased, ux) > 3.0           # the weights do the work

    yb = np.linspace(-0.45, 0.45, 37)
    _, fy = reweighted_fes(ys[n0:], w, yb, T)
    gx, gy = np.linspace(0.1, 1.0, 2000), np.linspace(-0.6, 0.6, 1201)
    gu = A * np.cos(2 * pi * gx[:, None] / 0.45) + 0.5 * KY * (gy[None, :] - 0.3 * np.sin(2 * pi * gx[:, None] / 0.9)) ** 2
    exact = np.array([-KT * logsumexp(-gu[:, (gy >= lo) & (gy < hi)] / KT) for lo, hi in zip(yb[:-1], yb[1:])])
    exact -= exact.min()
    keep = exact < 12.0
    assert rms(fy[keep], exact[keep]) < 1.5
    assert len(meta.hills) == 12000 and meta.hills[-1][-1] < meta.hills[0][-1]
