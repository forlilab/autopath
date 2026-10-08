import heapq
import itertools

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, Simulation, Topology
from openmm.app.metadynamics import BiasVariable

from autopath.metadynamics import TransitionTemperedMetadynamics, rct_weights, reweighted_fes, vstar

T = 300.0
KT = 0.0083144626 * T
A = 6.0


def _widest_path(bias, a, b, periodic):
    """Max over paths a->b of the min bias on the path (Dijkstra on -min)."""
    best = {a: bias[a]}
    heap = [(-bias[a], a)]
    while heap:
        neg, cell = heapq.heappop(heap)
        if cell == b:
            return -neg
        for ax in range(bias.ndim):
            for d in (-1, 1):
                n = list(cell)
                n[ax] += d
                if periodic[ax]:
                    n[ax] %= bias.shape[ax]
                elif not 0 <= n[ax] < bias.shape[ax]:
                    continue
                n = tuple(n)
                w = min(-neg, bias[n])
                if w > best.get(n, -np.inf):
                    best[n] = w
                    heapq.heappush(heap, (-w, n))


def test_vstar_1d_is_the_lowest_bias_between_the_basins():
    v = np.array([5.0, 3.0, 8.0, 1.0, 6.0])
    assert vstar(v, [(0,), (4,)]) == 1.0
    assert vstar(v, [(0,), (2,)]) == 3.0
    assert vstar(v, [(0,), (2,), (4,)]) == 1.0
    assert vstar(np.zeros(7), [(0,), (6,)]) == 0.0


def test_vstar_periodic_axis_links_through_the_wrap():
    v = np.array([9.0, 0.0, 0.0, 0.0, 9.0, 8.0, 9.0])
    assert vstar(v, [(0,), (4,)], [False]) == 0.0
    assert vstar(v, [(0,), (4,)], [True]) == 8.0


@pytest.mark.parametrize("periodic", [(False, False), (True, False), (True, True)])
def test_vstar_2d_matches_brute_force_widest_path(periodic):
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.random((13, 11)) * 10
        a, b = (1, 2), (11, 9)
        assert vstar(v, [a, b], periodic) == pytest.approx(_widest_path(v, a, b, periodic))


def test_vstar_three_basins_is_the_weakest_pairwise_link():
    rng = np.random.default_rng(5)
    v = rng.random((9, 9, 9)) * 10
    pts = [(0, 0, 0), (8, 4, 2), (3, 8, 8)]
    pair = [_widest_path(v, p, q, [False] * 3) for p, q in itertools.combinations(pts, 2)]
    assert vstar(v, pts) == pytest.approx(min(pair))         # the bottleneck metric is ultrametric


def _toy(expr):
    s = mm.System()
    s.addParticle(12.0)
    f = mm.CustomExternalForce(expr)
    f.addParticle(0, [])
    s.addForce(f)
    cv = mm.CustomExternalForce("min(max(x, 0), 1)")
    cv.addParticle(0, [])
    return s, BiasVariable(cv, 0.0, 1.0, 0.03, gridWidth=201)


def _sim(s, seed=1):
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 5 / u.picosecond, 0.004 * u.picosecond)
    integ.setRandomNumberSeed(seed)
    top = Topology()
    top.addAtom("C", Element.getBySymbol("C"), top.addResidue("X", top.addChain()))
    sim = Simulation(top, s, integ, mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions([[0.225, 0, 0]])
    sim.context.setVelocitiesToTemperature(T * u.kelvin, seed)
    return sim


def test_basins_must_lie_on_the_grid():
    s, bv = _toy("0")
    with pytest.raises(ValueError):
        TransitionTemperedMetadynamics(s, [bv], T * u.kelvin, 5, 1.0, 10, basins=[[0.2]])
    with pytest.raises(ValueError):
        TransitionTemperedMetadynamics(s, [bv], T * u.kelvin, 5, 1.0, 10, basins=[[0.2], [1.5]])


def test_rct_of_a_constant_bias_is_that_constant():
    s, bv = _toy("0")
    meta = TransitionTemperedMetadynamics(s, [bv], T * u.kelvin, 5, 1.0, 10, basins=[[0.2], [0.8]])
    meta._totalBias[:] = 7.5
    assert meta.rct() == pytest.approx(7.5, rel=1e-9)


WELLS = (f"{A}*cos(2*{np.pi}*x/0.45) + 0.5*41840*(step(0.1-x)*(0.1-x)^2 + step(x-1)*(x-1)^2)"
         " + 1000*(y^2+z^2)")


def test_hills_are_untempered_until_the_basins_connect():
    s, bv = _toy(WELLS)
    meta = TransitionTemperedMetadynamics(s, [bv], T * u.kelvin, 5, 1.0, 50,
                                          bounds=[(0.1, 1.0)], basins=[[0.225], [0.675]],
                                          threshold=0.5)
    sim = _sim(s)
    meta.step(sim, 50 * 600)
    v = np.array(meta.vstar_log)
    h = np.array([row[-1] for row in meta.hills])
    assert len(v) == len(h) == 600
    expect = np.exp(-np.maximum(v - 0.5, 0.0) / (KT * 4))
    assert np.allclose(h, expect)
    assert (v[:20] < 1e-6).all() and v[-1] > 5.0          # starts disconnected, ends linked
    assert h[-1] < 0.5


def test_bias_converges_to_minus_f_and_reweights():
    """Same toy as the WT reweighting test: -V(x) and the c(t)-reweighted F(x)
    must match A cos(2πx/0.45) on the open domain."""
    s, bv = _toy(WELLS)
    meta = TransitionTemperedMetadynamics(s, [bv], T * u.kelvin, 5, 1.0 * u.kilojoule_per_mole,
                                          250, bounds=[(0.1, 1.0)], basins=[[0.225], [0.675]])
    sim = _sim(s)
    xs, bias, rct = [], [], []
    for _ in range(1200):
        meta.step(sim, 2500)
        xs.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True)
                  .value_in_unit(u.nanometer)[0][0])
        bias.append(meta.bias_energy(sim)), rct.append(meta.rct())
    xs, bias, rct = map(np.array, (xs, bias, rct))

    def rms(f, ref):
        ok = np.isfinite(f)
        d = (f - ref)[ok]
        return np.sqrt(np.mean((d - d.mean()) ** 2))

    grid = np.linspace(0.0, 1.0, 201)
    inner = (grid > 0.15) & (grid < 0.95)
    fe = meta.getFreeEnergy().value_in_unit(u.kilojoule_per_mole)
    assert rms(fe[inner], A * np.cos(2 * np.pi * grid[inner] / 0.45)) < 1.5

    n0 = len(xs) // 5
    xc, fx = reweighted_fes(xs[n0:], rct_weights(bias[n0:], rct[n0:], T),
                            np.linspace(0.12, 0.98, 44), T)
    assert rms(fx, A * np.cos(2 * np.pi * xc / 0.45)) < 1.5
