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
    """Pocket -> cylinder -> pocket counts as a round trip regardless of CV grid fraction."""
    bound_max, unbound_min = funnel_basins(_params())
    z = np.r_[np.full(30, 2.0), np.full(30, 16.0), np.full(30, 2.5),   # trip 1
              np.full(30, 18.0), np.full(30, 3.0),                     # trip 2
              np.full(30, 9.0), np.full(30, 2.0)]                      # barrier only: no trip
    ts = MetadynamicsAnalysis.analyze_colvar_transitions(
        z[:, None], bound_threshold=bound_max, unbound_threshold=unbound_min)
    assert ts["n_round_trips"] == 2


def test_funnel_z_cv_is_tested_in_angstrom_against_the_basins(tmp_path):
    z_nm = np.r_[np.full(30, 0.2), np.full(30, 1.6), np.full(30, 0.2)]   # pocket -> cylinder -> pocket
    np.save(tmp_path / "COLVAR_w.npy", z_nm[:, None])
    a = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    basins = funnel_basins(_params())
    as_z = a.walker_transitions(str(tmp_path / "COLVAR_w.npy"), basins, grid_min=-0.7, grid_max=2.65,
                                cv_is_funnel_z=True)
    assert as_z["coordinate"] == "funnel_z" and as_z["n_round_trips"] == 1
    as_cv = a.walker_transitions(str(tmp_path / "COLVAR_w.npy"), basins, grid_min=-0.7, grid_max=2.65)
    assert as_cv["coordinate"] == "cv" and as_cv["n_round_trips"] == 0    # 1.6 nm < 0.9 of the grid
