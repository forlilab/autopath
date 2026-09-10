"""The pocket exit-direction feature must be invariant to global rigid motion.

geom_exit_1/2/3 project the ligand->pocket unit vector onto the pocket
principal axes. Eigenvectors are defined only up to a sign, so the sign must be
fixed by some convention. The original convention ("make the largest Cartesian
component positive") is evaluated in the LAB frame, so it is not rotation
invariant: rotating the whole system -- which changes no physics -- flipped the
reported signs. Because the protein tumbles freely during a pull, this produced
discontinuous sign flips mid-trajectory, a bimodal feature distribution with a
hard gap around zero, and single-step jumps of ~1.4 in a unit-vector component.

Everything here exercises the per-frame call only; no trajectory is involved,
matching how the feature is computed live during pulling.
"""
import numpy as np

from autopath.pulling.steered_md import SteeredMD


def _make_smd(seed=0, n_pocket=15):
    """Minimal SteeredMD carrying only what _exit_direction needs."""
    rng = np.random.default_rng(seed)
    pocket = rng.normal(size=(n_pocket, 3)) * np.array([1.0, 0.6, 0.3])
    ligand = pocket.mean(0) + np.array([0.9, 0.35, 0.25]) + rng.normal(scale=0.02, size=(6, 3))
    positions = np.vstack([pocket, ligand])

    smd = object.__new__(SteeredMD)
    smd.subset_protein_CA = np.arange(n_pocket)
    smd.groupA_atoms = np.arange(n_pocket, n_pocket + len(ligand))
    smd._set_exit_reference(positions)
    return smd, positions


def _rand_rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]          # proper rotation, no reflection
    return q


def test_exit_direction_is_invariant_under_global_rotation():
    smd, pos = _make_smd()
    ref = np.array(smd._exit_direction(pos))
    for seed in range(25):
        R = _rand_rot(seed)
        got = np.array(smd._exit_direction(pos @ R.T))
        assert np.allclose(got, ref, atol=1e-6), (
            f"rotation {seed} changed the exit vector: {ref} -> {got}"
        )


def test_exit_direction_is_invariant_under_translation():
    smd, pos = _make_smd()
    ref = np.array(smd._exit_direction(pos))
    got = np.array(smd._exit_direction(pos + np.array([13.0, -7.5, 2.25])))
    assert np.allclose(got, ref, atol=1e-6)


def test_exit_direction_tracks_real_ligand_motion():
    """Invariance must not be bought by making the feature constant."""
    smd, pos = _make_smd()
    ref = np.array(smd._exit_direction(pos))
    moved = pos.copy()
    moved[smd.groupA_atoms] += np.array([-1.8, 0.5, 0.2])   # ligand exits elsewhere
    got = np.array(smd._exit_direction(moved))
    assert not np.allclose(got, ref, atol=1e-2), "feature is insensitive to ligand motion"


def test_exit_direction_is_a_unit_vector():
    smd, pos = _make_smd()
    e = np.array(smd._exit_direction(pos))
    assert np.isclose(np.linalg.norm(e), 1.0, atol=1e-8)


def test_exit_direction_is_continuous_along_a_smooth_path():
    """A smoothly moving ligand in a tumbling frame must give a smooth feature.

    This is the failure that produced the isolated PCA cloud: single-step jumps
    of ~1.4 in a unit-vector component.
    """
    smd, pos = _make_smd()
    prev = None
    max_step = 0.0
    for t in range(60):
        frame = pos.copy()
        frame[smd.groupA_atoms] += np.array([0.02 * t, 0.01 * t, 0.0])
        frame = frame @ _rand_rot(1000 + t).T          # tumbling lab frame
        e = np.array(smd._exit_direction(frame))
        if prev is not None:
            max_step = max(max_step, float(np.abs(e - prev).max()))
        prev = e
    assert max_step < 0.2, f"discontinuous exit vector: max single-step |delta|={max_step:.3f}"


def test_exit_direction_returns_none_when_coms_coincide():
    smd, pos = _make_smd()
    degenerate = pos.copy()
    degenerate[smd.groupA_atoms] = pos[smd.subset_protein_CA].mean(0)
    assert smd._exit_direction(degenerate) is None
