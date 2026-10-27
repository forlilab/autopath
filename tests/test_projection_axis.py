import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
from autopath.pulling.projection import axis_from_pulls

LIG = [4, 5, 6]
POCKET = [0, 1, 2, 3]
PROTEIN_XYZ = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]], float)
OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)


def _universe(paths, n=40, step=1.0, seed=0):
    """Protein atoms + 3-atom ligand; each path moves the ligand along its direction."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in paths:
        d = np.asarray(d, float) / np.linalg.norm(d)
        for t in range(n):
            lig = np.array([1.5, 1.5, 3.0]) + OFFSETS + d * step * t
            lig = lig + rng.normal(scale=0.05, size=lig.shape)
            frames.append(np.vstack([PROTEIN_XYZ, lig]))
    u = mda.Universe.empty(7, n_residues=2, atom_resindex=[0] * 4 + [1] * 3, trajectory=True)
    u.add_TopologyAttr("resnames", ["ALA", "UNK"])
    u.add_TopologyAttr("names", ["CA"] * 4 + ["C1", "C2", "C3"])
    u.add_TopologyAttr("masses", [12.0] * 7)
    u.load_new(np.array(frames), format=MemoryReader)
    return u


def test_axis_recovered():
    true = np.array([0.3, 0.2, 1.0])
    true /= np.linalg.norm(true)
    u = _universe([true])
    out = axis_from_pulls(u, LIG, POCKET, LIG)
    assert np.dot(out["unbinding_axis"], true) > 0.99
    assert out["z_max_ang"] > 0 and out["z_cc_ang"] > 0


def test_frames_subset_selects_path():
    a, b = np.array([0, 0, 1.0]), np.array([1.0, 0, 0.5])
    u = _universe([a, b])
    ax_a = axis_from_pulls(u, LIG, POCKET, LIG, frames=range(0, 40))["unbinding_axis"]
    ax_b = axis_from_pulls(u, LIG, POCKET, LIG, frames=range(40, 80))["unbinding_axis"]
    assert np.dot(ax_a, a) > 0.99
    assert np.dot(ax_b, b / np.linalg.norm(b)) > 0.99


def test_z_max_clears_bulk():
    u = _universe([[0, 0, 1.0]])
    out = axis_from_pulls(u, LIG, POCKET, LIG, bulk_cutoff_ang=10.0)
    u.trajectory[0]
    pos = u.atoms.positions.copy()
    z0 = float(np.dot(pos[LIG].mean(0) - pos[POCKET].mean(0), out["unbinding_axis"]))
    pos[LIG] += out["unbinding_axis"] * (out["z_max_ang"] - z0)
    d = np.linalg.norm(pos[LIG][:, None] - pos[POCKET][None], axis=-1).min()
    assert d > 10.0 - 1e-6
    assert d < 10.0 + 1.0


def test_generate_funnel_parameters_unchanged():
    # recorded from the pre-refactor implementation on this universe
    u = _universe([[0.3, 0.2, 1.0]])
    p = generate_funnel_parameters_from_trajectory(u, "protein", "resname UNK", stride=1, verbose=False)
    np.testing.assert_allclose(p["unbinding_axis"], [0.28282172339811973, 0.18790632527933115, 0.9405865646999941],
                               atol=1e-12)
    np.testing.assert_allclose(p["z_max"]._value, 41.99347620841751, atol=1e-9)


def test_generate_funnel_parameters_unchanged_no_pca():
    u = _universe([[0.3, 0.2, 1.0]])
    p = generate_funnel_parameters_from_trajectory(u, "protein", "resname UNK", stride=1, verbose=False,
                                                   use_pca=False)
    np.testing.assert_allclose(p["unbinding_axis"], [0.2831356280146114, 0.1881860315379592, 0.9404361933075341],
                               atol=1e-12)
    np.testing.assert_allclose(p["z_max"]._value, 41.99323543599985, atol=1e-9)
