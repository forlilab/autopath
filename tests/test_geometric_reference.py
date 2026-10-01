import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.Analysis import correct_fe_for_funnel, MetadynamicsAnalysis as MA
from autopath.metadynamics import report as R

S = np.linspace(0.0, 1.0, 125)
WELL = -50.0 * np.exp(-((S - 0.05) / 0.04) ** 2)
# flat solvated plateau at +13 kJ/mol, with the hill-deficit dip/jump at the hard edge
FES = WELL + 13.0 * (S > 0.3) - 10.0 * np.exp(-((S - 0.98) / 0.015) ** 2)


def _fp(z_cc=14.0, z_max=26.0):
    rng = np.random.default_rng(0)
    return {"z_cc": z_cc * unit.angstrom, "z_max": z_max * unit.angstrom,
            "R_cylinder": 4.0 * unit.angstrom,
            "axial_distances": np.r_[rng.uniform(1, 3, 200), rng.uniform(3, 20, 800)]}


def test_reference_range_sets_the_unbound_zero():
    ref = (0.55, 0.85)
    r = correct_fe_for_funnel(FES, _fp(), temperature=300.0, cv_length_per_unit_ang=20.0,
                              reference_range=ref, verbose=False)
    assert r["plateau_range"] == pytest.approx(ref, abs=0.01)
    plateau = FES[(S >= ref[0]) & (S <= ref[1])].mean()
    assert r["bound_depth_kj_mol"] == pytest.approx(FES[:25].min() - plateau)
    # the last-10-bins default sits on the edge dip and gives a shallower well
    d = correct_fe_for_funnel(FES, _fp(), temperature=300.0, cv_length_per_unit_ang=20.0,
                              verbose=False)
    assert d["dG_bind_std_kj_mol"] > r["dG_bind_std_kj_mol"]


def test_flatness_is_judged_on_the_reference_region():
    flat = MA.assess_convergence([{"n_round_trips": 2}], FES, reference_range=(0.55, 0.85))
    tail = MA.assess_convergence([{"n_round_trips": 2}], FES)
    assert flat["plateau_span_kj"] < 1.0 and tail["plateau_span_kj"] > 5.0


def test_reference_range_for_funnel_z_is_the_cylinder_clear_of_the_wall():
    lo, hi = R.reference_range_from_funnel(_fp(), grid=(0.0, 2.6), cv_is_funnel_z=True,
                                           sigma_ang=1.0)
    assert (lo, hi) == pytest.approx((1.4, 2.3))                 # 14 A .. 26 - 3 A, in nm


def test_reference_range_for_a_path_cv_is_mapped_through_the_recorded_z(tmp_path):
    s = np.linspace(0.0, 1.0, 4000)
    np.save(tmp_path / "COLVAR_w.npy", s[:, None])
    np.save(tmp_path / "FUNNELZ_w.npy", 2.0 + 24.0 * s)          # z = 2 + 24 s (A)
    lo, hi = R.reference_range_from_funnel(_fp(), grid=(0.0, 1.0), cv_is_funnel_z=False,
                                           sigma_ang=1.0, out_dir=str(tmp_path))
    assert lo == pytest.approx((14.0 - 2.0) / 24.0, abs=0.01)
    assert hi == pytest.approx((23.0 - 2.0) / 24.0, abs=0.01)


def test_reference_window_stays_clear_of_the_cv_edge():
    """The artifact sits at the CV's own edge, so no mapping may put the window there."""
    lo, hi = R.reference_range_from_funnel(_fp(), grid=(0.0, 1.0), cv_is_funnel_z=False,
                                           sigma_ang=1.0, hill_width_cv=0.05)
    assert hi <= 1.0 - 3 * 0.05 + 1e-9
