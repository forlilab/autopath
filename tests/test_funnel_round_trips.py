import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.Funnel import funnel_basins, funnel_z_ang
from autopath.metadynamics.Analysis import MetadynamicsAnalysis


def _params(z_cc=14.0):
    rng = np.random.default_rng(0)
    axial = np.r_[rng.uniform(1.0, 3.0, 200), rng.uniform(3.0, 20.0, 800)]
    return {"z_cc": z_cc * unit.angstrom, "axial_distances": axial}


def test_basins_are_the_pocket_and_the_cylinder():
    p = _params()
    bound_max, unbound_min = funnel_basins(p)
    z_site = np.percentile(np.abs(p["axial_distances"]), 10.0)
    assert unbound_min == pytest.approx(14.0)                      # cylinder starts at z_cc
    assert bound_max == pytest.approx(z_site + 0.5 * (14.0 - z_site))
    assert z_site < bound_max < unbound_min


def test_funnel_z_is_the_mass_weighted_projection():
    pos = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.2]])
    z = funnel_z_ang(pos, [0, 1], [2, 3], np.array([12.0, 12.0, 1.0, 3.0]), [0, 0, 1])
    assert z == pytest.approx((1.0 * 1.0 + 3.0 * 1.2) / 4.0 * 10.0)


def test_excursions_into_the_cylinder_count_as_round_trips():
    """A ligand that leaves the pocket, enters the cylinder and comes back is a round
    trip even if it never reaches the last 10% of the CV grid."""
    bound_max, unbound_min = funnel_basins(_params())
    z = np.r_[np.full(30, 2.0), np.full(30, 16.0), np.full(30, 2.5),   # trip 1
              np.full(30, 18.0), np.full(30, 3.0),                     # trip 2
              np.full(30, 9.0), np.full(30, 2.0)]                      # barrier only: no trip
    ts = MetadynamicsAnalysis.analyze_colvar_transitions(
        z[:, None], bound_threshold=bound_max, unbound_threshold=unbound_min)
    assert ts["n_round_trips"] == 2


def test_walker_transitions_prefer_the_recorded_funnel_z(tmp_path):
    s = np.r_[np.full(30, 0.05), np.full(30, 0.6), np.full(30, 0.05)]     # never > 0.9
    z = np.r_[np.full(30, 2.0), np.full(30, 16.0), np.full(30, 2.0)]      # reaches the cylinder
    np.save(tmp_path / "COLVAR_w.npy", s[:, None])
    a = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    basins = funnel_basins(_params())
    no_z = a.walker_transitions(str(tmp_path / "COLVAR_w.npy"), basins, grid_min=0.0, grid_max=1.0)
    assert no_z["coordinate"] == "cv" and no_z["n_round_trips"] == 0
    np.save(tmp_path / "FUNNELZ_w.npy", z)
    with_z = a.walker_transitions(str(tmp_path / "COLVAR_w.npy"), basins, grid_min=0.0, grid_max=1.0)
    assert with_z["coordinate"] == "funnel_z" and with_z["n_round_trips"] == 1
