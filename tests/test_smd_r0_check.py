"""Regression tests for SteeredMD._check_r0_and_cap_moves.

Reproduces the intermittent bug where cvpack's CentroidFunction CV
(com_dist.getValue) returns a value several nm off the true minimum-image
group-centroid distance (observed: 7.6 nm vs a true 1.7 nm). The old,
inline version of this logic only logged a WARNING on disagreement and
then kept using the (wrong) cvpack scalar to compute the move cap, which
made r_max - initial_r0_nm go negative and floored sMD_moves to 1 (log:
"Capping sMD_moves 2700 -> 1"). The fixed version must raise instead of
silently proceeding, and must never let a genuinely-fitting pull collapse
to a tiny move count.
"""
import numpy as np
import pytest

from autopath.pulling.steered_md import SteeredMD


def _two_point_groups(sep_x, box=(12.7, 13.3, 10.0)):
    """Two 2-atom groups, both atoms coincident with their group's centroid,
    separated by sep_x nm along x. Returns (positions, groupA, groupB,
    masses_A, masses_B, box_nm)."""
    positions = np.array([
        [sep_x, 0.0, 0.0],
        [sep_x, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    groupA, groupB = [0, 1], [2, 3]
    masses = np.ones(2)
    box_nm = np.diag(np.array(box, dtype=float))
    return positions, groupA, groupB, masses, masses, box_nm


def test_raises_when_cvpack_cv_disagrees_with_geometry():
    """Mirrors the observed failure: cvpack claims 7.612 nm, true |v| is
    1.743 nm. Must raise, not silently proceed."""
    positions, groupA, groupB, mA, mB, box_nm = _two_point_groups(sep_x=1.743)
    with pytest.raises(RuntimeError, match="disagrees"):
        SteeredMD._check_r0_and_cap_moves(
            initial_r0_nm=7.612,
            positions_nm=positions, box_nm=box_nm,
            groupA_atoms=groupA, groupB_atoms=groupB,
            masses_A=mA, masses_B=mB,
            dx_nm=0.001, n_moves=2700, apply_pbc_cap=True,
            sMD_max_r_offset=3.0, pbc_safety_nm=0.5,
        )


def test_no_raise_when_cvpack_cv_agrees_with_geometry():
    positions, groupA, groupB, mA, mB, box_nm = _two_point_groups(sep_x=1.7)
    n_moves = SteeredMD._check_r0_and_cap_moves(
        initial_r0_nm=1.7,
        positions_nm=positions, box_nm=box_nm,
        groupA_atoms=groupA, groupB_atoms=groupB,
        masses_A=mA, masses_B=mB,
        dx_nm=0.001, n_moves=2700, apply_pbc_cap=True,
        sMD_max_r_offset=3.0, pbc_safety_nm=0.5,
    )
    # r_max = min(half-box-x - safety, r0+offset) = min(5.85, 4.7) = 4.7 nm
    # moves_safe = (4.7 - 1.7)/0.001 = 3000 >= requested 2700: not capped.
    assert n_moves == 2700


def test_cap_still_applies_when_geometry_genuinely_binds():
    positions, groupA, groupB, mA, mB, box_nm = _two_point_groups(sep_x=1.7)
    n_moves = SteeredMD._check_r0_and_cap_moves(
        initial_r0_nm=1.7,
        positions_nm=positions, box_nm=box_nm,
        groupA_atoms=groupA, groupB_atoms=groupB,
        masses_A=mA, masses_B=mB,
        dx_nm=0.001, n_moves=5000, apply_pbc_cap=True,
        sMD_max_r_offset=3.0, pbc_safety_nm=0.5,
    )
    assert n_moves == 3000


def test_cap_skipped_when_not_requested():
    """Reverse-direction pulls (apply_pbc_cap=False) still get the geometry
    cross-check, but n_moves passes through unmodified."""
    positions, groupA, groupB, mA, mB, box_nm = _two_point_groups(sep_x=1.7)
    n_moves = SteeredMD._check_r0_and_cap_moves(
        initial_r0_nm=1.7,
        positions_nm=positions, box_nm=box_nm,
        groupA_atoms=groupA, groupB_atoms=groupB,
        masses_A=mA, masses_B=mB,
        dx_nm=0.001, n_moves=2700, apply_pbc_cap=False,
        sMD_max_r_offset=3.0, pbc_safety_nm=0.5,
    )
    assert n_moves == 2700


def test_magnitude_computed_once(monkeypatch):
    """|v| must be computed a single time per call (no duplicated recompute)."""
    positions, groupA, groupB, mA, mB, box_nm = _two_point_groups(sep_x=1.7)
    calls = []
    real_norm = np.linalg.norm

    def counting_norm(*args, **kwargs):
        calls.append(1)
        return real_norm(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", counting_norm)
    SteeredMD._check_r0_and_cap_moves(
        initial_r0_nm=1.7,
        positions_nm=positions, box_nm=box_nm,
        groupA_atoms=groupA, groupB_atoms=groupB,
        masses_A=mA, masses_B=mB,
        dx_nm=0.001, n_moves=2700, apply_pbc_cap=True,
        sMD_max_r_offset=3.0, pbc_safety_nm=0.5,
    )
    assert len(calls) == 1
