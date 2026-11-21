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


def test_reference_range_for_a_path_cv_is_mapped_linearly_from_the_site():
    from autopath.metadynamics.Funnel import funnel_basins
    z0 = 2.0 * funnel_basins(_fp())[0] - 14.0                     # sMD bound-site z
    lo, hi = R.reference_range_from_funnel(_fp(), grid=(0.0, 1.0), cv_is_funnel_z=False,
                                           sigma_ang=1.0)
    assert lo == pytest.approx((14.0 - z0) / (26.0 - z0))
    assert hi == pytest.approx((23.0 - z0) / (26.0 - z0))


def test_reference_window_stays_clear_of_the_cv_edge():
    """The reference window ends at least 3 hill widths before the CV edge."""
    lo, hi = R.reference_range_from_funnel(_fp(), grid=(0.0, 1.0), cv_is_funnel_z=False,
                                           sigma_ang=1.0, hill_width_cv=0.05)
    assert hi <= 1.0 - 3 * 0.05 + 1e-9


def test_funnel_z_grid_below_zero_keeps_the_bound_site():
    """Extending the funnel z grid below 0 leaves ΔG unchanged once the origin is given."""
    from autopath.metadynamics.CV import funnel_z_cv
    dz = 2.6 / 124
    z_old = np.linspace(0.0, 2.6, 125)
    z_new = np.linspace(-1.4, 2.6, int(round(4.0 / dz)) + 1)

    def fes(z):                                   # well at 0.2 nm, plateau 0, no weight below 0
        return -50.0 * np.exp(-((z - 0.2) / 0.08) ** 2) + 60.0 * (z < -0.05)

    kw = dict(temperature=300.0, cv_length_per_unit_ang=10.0, reference_range=(1.45, 2.3),
              verbose=False)
    ref = correct_fe_for_funnel(fes(z_old), _fp(), grid_min=0.0, grid_max=2.6, **kw)
    new = correct_fe_for_funnel(fes(z_new), _fp(), grid_min=-1.4, grid_max=2.6, cv_origin=0.0, **kw)
    assert new["dG_bind_std_kj_mol"] == pytest.approx(ref["dG_bind_std_kj_mol"], abs=0.2)
    assert new["site_range"][1] == pytest.approx(0.52, abs=dz)
    with pytest.raises(ValueError, match="Bound state not sampled"):
        correct_fe_for_funnel(fes(z_new), _fp(), grid_min=-1.4, grid_max=2.6, **kw)
    spec = funnel_z_cv([0], [1], [0, 0, 1], grid_min=-1.4, grid_max=2.6)
    assert spec.bound_end == 0.0 and spec.resolve(None, 2, None).bound_end == 0.0


def test_reference_range_ignores_a_funnel_z_grid_below_zero():
    lo, hi = R.reference_range_from_funnel(_fp(), grid=(-1.4, 2.6), cv_is_funnel_z=True,
                                           sigma_ang=1.0)
    assert (lo, hi) == pytest.approx((1.4, 2.3))


def test_well_depth_uses_the_origin():
    """Readers of the bound site must look at z >= 0, not the first bins of the grid."""
    z = np.linspace(-1.4, 2.6, 190)
    fe = -50.0 * np.exp(-((z - 0.2) / 0.08) ** 2) + 60.0 * (z < -0.05)
    assert R._well_depth(fe, (-1.4, 2.6), 0.0) == pytest.approx(-50.0, abs=1.0)
    assert R._well_depth(fe, (-1.4, 2.6)) > 0                   # grid start: nothing bound


def test_finite_fes_result_is_unchanged_by_finite_masking():
    # values recorded from the pre-masking implementation
    r = correct_fe_for_funnel(FES, _fp(), temperature=300.0, cv_length_per_unit_ang=20.0, verbose=False)
    assert r["dG_bind_std_kj_mol"] == -48.19572218672401
    r = correct_fe_for_funnel(FES, _fp(), temperature=300.0, cv_length_per_unit_ang=20.0,
                              reference_range=(0.55, 0.85), verbose=False)
    assert r["dG_bind_std_kj_mol"] == -51.459105816204726


def test_inf_bins_do_not_break_the_correction():
    fes = FES.copy()
    fes[60:70] = np.inf
    fes[-4:] = np.inf
    r = correct_fe_for_funnel(fes, _fp(), temperature=300.0, cv_length_per_unit_ang=20.0, verbose=False)
    assert np.isfinite(r["dG_bind_std_kj_mol"])
