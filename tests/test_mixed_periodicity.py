"""Mixed periodic/non-periodic bias variables.

OpenMM's Metadynamics refuses them because Continuous{1,2,3}DFunction carries one periodic
flag for the whole table. The periodic axis is padded and the table declared non-periodic;
hills wrap over the true period so the padding holds the exact periodic continuation.
"""
from functools import reduce

import numpy as np
import openmm as mm
import pytest
from openmm.app.metadynamics import BiasVariable, Metadynamics

from autopath.metadynamics.BoundaryCorrection import BoundaryCorrectedMetadynamics

NP, ND = 61, 41


def _system():
    s = mm.System()
    for _ in range(4):
        s.addParticle(12.0)
    return s


def _torsion():
    f = mm.CustomTorsionForce("theta")
    f.addTorsion(0, 1, 2, 3, [])
    return f


def _distance():
    f = mm.CustomBondForce("r")
    f.addBond(0, 3, [])
    return f


def _vars(periodic_torsion):
    return [BiasVariable(_torsion(), -np.pi, np.pi, 0.25, periodic=periodic_torsion,
                         gridWidth=NP),
            BiasVariable(_distance(), 0.0, 2.0, 0.2, periodic=False, gridWidth=ND)]


def _meta(periodic_torsion, **kw):
    return BoundaryCorrectedMetadynamics(_system(), _vars(periodic_torsion), 300, 5, 1.0, 10,
                                         **kw)


def _deposit(meta, phis, heights=None):
    """Add hills at ``phis`` the way _addGaussian does, so the grid can be inspected."""
    heights = heights if heights is not None else np.ones(len(phis))
    for phi, h in zip(phis, heights):
        axes = []
        for i, v in enumerate(meta.variables):
            if meta._mixed and v.periodic:
                pv = meta._padded_variables[i]
                period = v.maxValue - v.minValue
                d = np.linspace(pv.minValue, pv.maxValue, pv.gridWidth) - phi
                d -= period * np.round(d / period)
                dist = np.abs(d) / period
            else:
                x = ((phi if i == 0 else 1.0) - v.minValue) / (v.maxValue - v.minValue)
                dist = np.abs(np.linspace(0, 1, v.gridWidth) - x)
            axes.append(np.exp(-0.5 * dist * dist / v._scaledVariance))
        meta._totalBias += h * reduce(np.multiply.outer, reversed(axes))
    return meta


@pytest.mark.parametrize("periodic", [False, True])
def test_uniform_periodicity_is_not_padded(periodic):
    """A wholly periodic or wholly non-periodic set keeps OpenMM's native grid."""
    vs = [BiasVariable(_torsion(), -np.pi, np.pi, 0.25, periodic=periodic, gridWidth=NP),
          BiasVariable(_torsion(), -np.pi, np.pi, 0.25, periodic=periodic, gridWidth=ND)]
    meta = BoundaryCorrectedMetadynamics(_system(), vs, 300, 5, 1.0, 10)
    assert meta._mixed is False
    assert meta._totalBias.shape == (ND, NP)


def test_stock_openmm_still_rejects_mixed():
    with pytest.raises(ValueError, match="mixed periodic"):
        Metadynamics(_system(), _vars(True), 300, 5, 1.0, 10)


def test_mixed_builds_a_padded_grid():
    meta = _meta(True)
    assert meta._mixed is True
    assert meta._pad[1] == 0                      # the non-periodic axis is never padded
    assert meta._totalBias.shape == (ND, NP + 2 * meta._pad[0])
    assert meta._bias_view(meta._totalBias).shape == (ND, NP)


def test_seam_is_closed_in_value_and_slope():
    """Both ends must agree, and so must the centred slopes that span the seam."""
    meta = _meta(True)
    rng = np.random.default_rng(0)
    _deposit(meta, rng.uniform(-np.pi, np.pi, 12), rng.uniform(0.5, 1.5, 12))
    pad = meta._pad[0]
    row = meta._totalBias[ND // 2]
    lo, hi = pad, pad + NP - 1
    assert row[lo] == pytest.approx(row[hi], abs=1e-12)
    d_lo = (row[lo + 1] - row[lo - 1]) / 2.0
    d_hi = (row[hi + 1] - row[hi - 1]) / 2.0
    assert d_lo == pytest.approx(d_hi, abs=1e-12)


def test_naive_non_periodic_leaves_the_seam_open():
    """Control: declaring the axis non-periodic without padding is NOT equivalent."""
    meta = _meta(False)
    rng = np.random.default_rng(0)
    _deposit(meta, rng.uniform(-np.pi, np.pi, 12), rng.uniform(0.5, 1.5, 12))
    row = meta._totalBias[ND // 2]
    assert abs(row[0] - row[-1]) > 0.1


def test_padding_is_the_periodic_extension_of_the_view():
    meta = _meta(True)
    rng = np.random.default_rng(1)
    _deposit(meta, rng.uniform(-np.pi, np.pi, 8))
    rebuilt = meta._pad_from_view(meta._bias_view(meta._totalBias))
    assert np.allclose(rebuilt, meta._totalBias, atol=1e-12)


def test_disk_round_trip_keeps_the_unpadded_shape(tmp_path):
    """The file holds the view, so existing readers of bias_*.npy are unaffected."""
    meta = _meta(True, saveFrequency=10, biasDir=str(tmp_path))
    rng = np.random.default_rng(2)
    _deposit(meta, rng.uniform(-np.pi, np.pi, 5))
    meta._selfBias = meta._totalBias.copy()
    meta._syncWithDisk()
    files = list(tmp_path.glob("bias_*.npy"))
    assert len(files) == 1
    saved = np.load(files[0])
    assert saved.shape == (ND, NP)
    assert np.allclose(meta._pad_from_view(saved), meta._totalBias, atol=1e-12)


def test_free_energy_and_rct_use_the_unpadded_grid():
    meta = _meta(True)
    _deposit(meta, [0.4, -1.2])
    assert meta.getFreeEnergy().shape == (ND, NP)
    assert np.isfinite(meta.rct())


def test_periodic_axis_is_never_edge_corrected():
    """A periodic axis has no edges, so the de Pablo factor must be exactly 1 along it.

    The padded copies are marked non-periodic so OpenMM will build the table; if that leaks
    into boundary_factor the axis is treated as bounded and hills are amplified at the seam,
    which injects a force spike exactly where the walker wraps.
    """
    meta = _meta(True)
    pad = meta._pad[0]
    row = meta.boundaryFactor[ND // 2]
    assert np.allclose(row, 1.0), f"periodic axis corrected: max {row.max():.3f}"
    # the non-periodic axis must still be corrected at its own edges
    col = meta.boundaryFactor[:, pad + NP // 2]
    assert col.max() > 1.0


def test_seam_bias_has_no_step_after_many_hills():
    """The bias either side of the seam must agree: a step there is what blows a run up."""
    meta = _meta(True)
    rng = np.random.default_rng(3)
    _deposit(meta, rng.uniform(-np.pi, np.pi, 40), rng.uniform(0.5, 1.5, 40))
    pad = meta._pad[0]
    row = meta._totalBias[ND // 2]
    lo, hi = pad, pad + NP - 1
    assert abs(row[lo] - row[hi]) < 1e-9
    # and one grid step either side of the seam, which is what the spline actually sees
    assert abs(row[lo - 1] - row[hi - 1]) < 1e-9
    assert abs(row[lo + 1] - row[hi + 1]) < 1e-9
