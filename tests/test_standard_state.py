import math

import numpy as np
import pytest
from openmm import unit as u

from autopath.metadynamics.Analysis import correct_fe_for_funnel
from autopath.pulling.standard_state import dG_standard_from_profile

KB = 8.314e-3
C0 = 1.0 / 1660.0


def _profile(D=20.0, k_ang=2.0, T=300.0):
    # W = -D + (k/2) z^2 on the site (z in A), 0 in bulk
    kT = KB * T
    z_nm = np.linspace(0.0, 3.0, 301)
    z_ang = z_nm * 10.0
    G = np.where(z_nm <= 1.0, -D + 0.5 * k_ang * z_ang ** 2, 0.0)
    G = np.minimum(G, 0.0)  # the well is truncated where it reaches the bulk
    return z_nm, G, kT


def test_gaussian_well_matches_analytic():
    D, k, T, R = 30.0, 8.0, 300.0, 5.0
    kT = KB * T
    z_nm = np.linspace(0.0, 3.0, 3001)
    z_ang = z_nm * 10.0
    # site ends where the well is >40 kT above its minimum, so the tail is negligible
    G = np.where(z_nm <= 0.8, -D + 0.5 * k * z_ang ** 2, 0.0)
    r = dG_standard_from_profile(z_nm, G, site_max_nm=0.8, bulk_range_nm=(2.0, 3.0),
                                 R_cyl_ang=R, temperature=T)
    integral = math.sqrt(math.pi * kT / (2 * k)) * math.exp(D / kT)
    expected = -kT * math.log(math.pi * R ** 2 * C0 * integral)
    assert abs(r["dG_bind_std"] - expected) < 1e-6
    assert r["integral_ang"] == pytest.approx(integral, rel=1e-6)


def test_correct_fe_for_funnel_unchanged():
    z = np.linspace(0, 1, 101)
    fe = (-40 * np.exp(-((z - 0.05) / 0.04) ** 2) + 30 * z ** 2 * (z < 0.5)
          + 7.5 * (z >= 0.5))
    fp = {"R_cylinder": 4.0 * u.angstrom, "z_cc": 12.0 * u.angstrom,
          "z_max": 30.0 * u.angstrom}
    r = correct_fe_for_funnel(fe, fp, temperature=300.0, cv_length_per_unit_ang=25.0,
                              reference_range=(0.7, 1.0))
    assert r["dG_bind_sim_kj_mol"] == pytest.approx(-47.425, rel=1e-12)
    assert r["correction_kj_mol"] == pytest.approx(-8.722851608368604, rel=1e-12)
    assert r["dG_bind_std_kj_mol"] == pytest.approx(-36.73187199079935, rel=1e-12)
    assert r["bound_integral_ang"] == pytest.approx(82159210.39300555, rel=1e-12)
    assert r["dG_basin_kj_mol"] == pytest.approx(-45.454723599167956, rel=1e-12)
    assert r["pKd"] == pytest.approx(6.395818024048098, rel=1e-12)
    assert float(r["fe_corrected"].sum()) == pytest.approx(549.8651208992907, rel=1e-12)


def test_bulk_slope_reported():
    z_nm, G, _ = _profile()
    G = np.where(z_nm >= 2.0, 3.0 * z_nm + 1.0, G)
    r = dG_standard_from_profile(z_nm, G, 1.0, (2.0, 3.0), R_cyl_ang=5.0)
    assert r["bulk_slope_kJ_per_nm"] == pytest.approx(3.0, rel=1e-9)
    assert r["bulk_ref_kJ"] == pytest.approx(3.0 * 2.5 + 1.0, rel=1e-9)


def test_nonuniform_grid_not_supported():
    z_nm, G, _ = _profile()
    z_nm = z_nm.copy()
    z_nm[10] += 0.002
    with pytest.raises(ValueError, match="not supported"):
        dG_standard_from_profile(z_nm, G, 1.0, (2.0, 3.0), R_cyl_ang=5.0)


def test_few_bulk_points_and_nan_not_supported():
    z_nm, G, _ = _profile()
    with pytest.raises(ValueError, match="not supported"):
        dG_standard_from_profile(z_nm, G, 1.0, (2.0, 2.01), R_cyl_ang=5.0)
    G = G.copy()
    G[5] = np.nan
    with pytest.raises(ValueError, match="not supported"):
        dG_standard_from_profile(z_nm, G, 1.0, (2.0, 3.0), R_cyl_ang=5.0)
