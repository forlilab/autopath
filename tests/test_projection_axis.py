import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader

from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
from autopath.pulling.projection import axis_from_pulls

LIG = [4, 5, 6]
POCKET = [0, 1, 2, 3]
PROTEIN_XYZ = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]], float)
OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)


EXTRA_PROTEIN = np.array([8.0, 1.5, 4.0])
START = np.array([1.5, 1.5, 1.5])


def _universe(paths, n=40, step=1.0, seed=0, noise=0.05, extra=False, solvent=False, box=None,
              start=START):
    """Protein atoms + 3-atom ligand (+ far protein atom, solvent); each path moves the ligand along its direction."""
    rng = np.random.default_rng(seed)
    frames = []
    d0 = np.asarray(paths[0], float) / np.linalg.norm(paths[0])
    tail = []
    if extra:
        tail.append(EXTRA_PROTEIN)
    if solvent:
        tail += [np.asarray(start) + d0 * s + [0.5, 0.5, 0] for s in (3.0, 5.0, 8.0, 12.0)]
    tail = np.array(tail).reshape(-1, 3)
    for d in paths:
        d = np.asarray(d, float) / np.linalg.norm(d)
        for t in range(n):
            lig = np.asarray(start) + OFFSETS + d * step * t + rng.normal(scale=noise, size=(3, 3))
            frames.append(np.vstack([PROTEIN_XYZ, lig, tail]))
    resnames = ["ALA", "UNK"] + (["ALA"] if extra else []) + (["HOH", "NA", "CL", "HOH"] if solvent else [])
    names = ["CA"] * 4 + ["C1", "C2", "C3"] + (["CA"] if extra else []) + (["O", "NA", "CL", "O"] if solvent else [])
    resindex = [0] * 4 + [1] * 3 + list(range(2, len(resnames)))
    u = mda.Universe.empty(len(names), n_residues=len(resnames), atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("masses", [12.0] * len(names))
    dims = None if box is None else np.array([box, box, box, 90, 90, 90], float)
    u.load_new(np.array(frames), format=MemoryReader, dimensions=dims)
    return u


def test_axis_recovered():
    true = np.array([0.3, 0.2, 1.0])
    true /= np.linalg.norm(true)
    u = _universe([true], n=6)
    out = axis_from_pulls(u, LIG, POCKET, LIG)
    assert np.dot(out["unbinding_axis"], true) > 0.99


def test_frames_subset_selects_path():
    a, b = np.array([0, 0, 1.0]), np.array([1.0, 0, 0.5])
    u = _universe([a, b], n=6)
    ax_a = axis_from_pulls(u, LIG, POCKET, LIG, frames=range(0, 6))["unbinding_axis"]
    ax_b = axis_from_pulls(u, LIG, POCKET, LIG, frames=range(6, 12))["unbinding_axis"]
    assert np.dot(ax_a, a) > 0.99
    assert np.dot(ax_b, b / np.linalg.norm(b)) > 0.99


def test_z_cc_matches_known_path():
    d = np.array([0.0, 0.0, 1.0])
    u = _universe([d], n=6, noise=0.0)
    out = axis_from_pulls(u, LIG, POCKET, LIG, z_cc_percentile=80.0)
    lig_com = START + OFFSETS.mean(0)
    pocket_com = PROTEIN_XYZ.mean(0)
    z = [(lig_com + d * t - pocket_com) @ d for t in range(6)]
    assert out["z_cc_ang"] == pytest.approx(np.percentile(z, 80.0), abs=1e-9)


def test_z_max_clears_protein_not_solvent():
    kw = dict(n=6, noise=0.0, extra=True)
    d = [0, 0, 1.0]
    plain = axis_from_pulls(_universe([d], **kw), LIG, POCKET, LIG, bulk_cutoff_ang=10.0)
    wet = _universe([d], solvent=True, **kw)
    out = axis_from_pulls(wet, LIG, POCKET, LIG, bulk_cutoff_ang=10.0)
    assert out["z_max_ang"] == pytest.approx(plain["z_max_ang"], abs=1e-9)
    protein = list(POCKET) + [7]
    wet.trajectory[0]
    pos = wet.atoms.positions.copy()
    z0 = float(np.dot(pos[LIG].mean(0) - pos[POCKET].mean(0), out["unbinding_axis"]))
    pos[LIG] += out["unbinding_axis"] * (out["z_max_ang"] - z0)
    dist = np.linalg.norm(pos[LIG][:, None] - pos[protein][None], axis=-1).min()
    assert 10.0 - 1e-6 < dist < 10.0 + 1.0
    # the far protein atom, not the pocket, sets the clearance
    assert np.linalg.norm(pos[LIG][:, None] - pos[POCKET][None], axis=-1).min() > 10.0 + 0.5


def test_empty_protein_not_supported():
    u = _universe([[0, 0, 1.0]], n=6)
    with pytest.raises(ValueError, match="not supported"):
        axis_from_pulls(u, LIG, POCKET, LIG, protein_idx=LIG)


def test_z_max_not_beyond_z_cc_not_supported():
    u = _universe([[0, 0, 1.0]], n=40)
    with pytest.raises(ValueError, match="not supported"):
        axis_from_pulls(u, LIG, POCKET, LIG)


def test_unimaged_jump_not_supported():
    u = _universe([[0, 0, 1.0]], n=6, box=8.0, step=5.0)
    with pytest.raises(ValueError, match="not supported"):
        axis_from_pulls(u, LIG, POCKET, LIG)
    ok = _universe([[0, 0, 1.0]], n=6, box=60.0, step=1.0)
    axis_from_pulls(ok, LIG, POCKET, LIG)


def test_generate_funnel_parameters_unchanged():
    # recorded from the pre-refactor implementation on this universe
    u = _universe([[0.3, 0.2, 1.0]], start=[1.5, 1.5, 3.0])
    p = generate_funnel_parameters_from_trajectory(u, "protein", "resname UNK", stride=1, verbose=False)
    np.testing.assert_allclose(p["unbinding_axis"], [0.28282172339811973, 0.18790632527933115, 0.9405865646999941],
                               atol=1e-12)
    np.testing.assert_allclose(p["z_max"]._value, 41.99347620841751, atol=1e-9)


def test_generate_funnel_parameters_unchanged_no_pca():
    u = _universe([[0.3, 0.2, 1.0]], start=[1.5, 1.5, 3.0])
    p = generate_funnel_parameters_from_trajectory(u, "protein", "resname UNK", stride=1, verbose=False,
                                                   use_pca=False)
    np.testing.assert_allclose(p["unbinding_axis"], [0.2831356280146114, 0.1881860315379592, 0.9404361933075341],
                               atol=1e-12)
    np.testing.assert_allclose(p["z_max"]._value, 41.99323543599985, atol=1e-9)


# Protein with a frame: pocket residues at the origin, two blobs of residues (N, CA, C) beside it, all one chain.
N_POCKET_RES, N_BLOB_RES = 4, 25


def _framed_protein(rng):
    xyz, centres = [], [np.zeros(3)] * N_POCKET_RES
    for c in (np.array([20.0, 0.0, 0.0]), np.array([0.0, 20.0, 0.0])):
        centres += [c + rng.normal(scale=3.0, size=3) for _ in range(N_BLOB_RES)]
    for i, c in enumerate(centres):
        ca = c + (rng.normal(scale=1.0, size=3) if i < N_POCKET_RES else 0.0)
        xyz += [ca + [-1.2, 0.5, 0.0], ca, ca + [1.2, 0.5, 0.0]]
    return np.array(xyz)


def _framed_system(n=6, drift=(0.0, 0.0, 0.0), seed=1):
    """(universe, openmm topology, ligand idx, pocket idx); the ligand leaves the pocket along +z relative
    to the pocket while the whole protein drifts by ``drift`` Å per frame."""
    import openmm.app as app
    rng = np.random.default_rng(seed)
    prot = _framed_protein(rng)
    n_prot = len(prot)
    pocket = [3 * i + 1 for i in range(N_POCKET_RES)]
    lig = [n_prot, n_prot + 1, n_prot + 2]
    lig0 = np.array([0.0, 0.0, 2.0]) + OFFSETS
    frames = []
    for t in range(n):
        shift = np.asarray(drift) * t
        frames.append(np.vstack([prot + shift, lig0 + shift + [0.0, 0.0, 1.0 * t]
                                 + rng.normal(scale=0.05, size=(3, 3))]))
    n_res = n_prot // 3
    resindex = [i // 3 for i in range(n_prot)] + [n_res] * 3
    u = mda.Universe.empty(n_prot + 3, n_residues=n_res + 1, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("resnames", ["ALA"] * n_res + ["UNK"])
    u.add_TopologyAttr("names", ["N", "CA", "C"] * n_res + ["C1", "C2", "C3"])
    u.add_TopologyAttr("masses", [14.0, 12.0, 12.0] * n_res + [12.0] * 3)
    u.load_new(np.array(frames), format=MemoryReader)

    top = app.Topology()
    chain = top.addChain()
    atoms = []
    for r in range(n_res):
        res = top.addResidue("ALA", chain)
        atoms += [top.addAtom(nm, app.element.get_by_symbol(nm[0]), res) for nm in ("N", "CA", "C")]
    res = top.addResidue("UNK", chain)
    atoms += [top.addAtom(f"C{i}", app.element.carbon, res) for i in (1, 2, 3)]
    for i in range(n_prot - 1):
        top.addBond(atoms[i], atoms[i + 1])
    top.addBond(atoms[lig[0]], atoms[lig[1]])
    top.addBond(atoms[lig[1]], atoms[lig[2]])
    return u, top, lig, pocket


def _rotation(seed=3):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_axis_uses_ligand_relative_to_pocket():
    # the protein drifts along x while the ligand leaves along z relative to the pocket
    u, top, lig, pocket = _framed_system(drift=(1.0, 0.0, 0.0))
    out = axis_from_pulls(u, lig, pocket, lig)
    assert np.dot(out["unbinding_axis"], [0.0, 0.0, 1.0]) > 0.999


def test_params_frame_from_same_universe():
    from autopath.metadynamics.Funnel import funnel_axis_at
    from autopath.pulling.projection import check_projection_params
    u, top, lig, pocket = _framed_system()
    out = axis_from_pulls(u, lig, pocket, lig, topology=top, R_cylinder_ang=1.5)
    p = out["params"]
    check_projection_params(p, lig, pocket)
    u.trajectory[0]
    n = funnel_axis_at(p, u.atoms.positions / 10.0, u.atoms.masses)
    np.testing.assert_allclose(n, out["unbinding_axis"], atol=1e-9)
    assert p["z_cc"].value_in_unit(p["z_cc"].unit) == pytest.approx(out["z_cc_ang"])
    assert p["R_cylinder"]._value == pytest.approx(1.5)


def test_axis_coeffs_invariant_to_rigid_rotation():
    u, top, lig, pocket = _framed_system()
    ref = axis_from_pulls(u, lig, pocket, lig, topology=top)
    R = _rotation()
    rot = np.array([ts.positions.copy() for ts in u.trajectory]) @ R.T + [5.0, -3.0, 7.0]
    u_rot = u.copy()
    u_rot.load_new(rot, format=MemoryReader)
    out = axis_from_pulls(u_rot, lig, pocket, lig, topology=top)
    # MDAnalysis keeps float32 coordinates
    np.testing.assert_allclose(out["unbinding_axis"], R @ ref["unbinding_axis"], atol=1e-6)
    np.testing.assert_allclose(out["params"]["axis_coeffs"], ref["params"]["axis_coeffs"], atol=1e-6)
    assert out["params"]["frame_b_index"] == ref["params"]["frame_b_index"]


def test_topology_size_mismatch_not_supported():
    u, top, lig, pocket = _framed_system()
    with pytest.raises(ValueError, match="not supported"):
        axis_from_pulls(u, lig, pocket, lig, topology=_drop_atom(top))


def _drop_atom(top):
    import openmm.app as app
    mod = app.Modeller(top, [[0.0, 0.0, 0.0]] * top.getNumAtoms())
    mod.delete([list(top.atoms())[-1]])
    return mod.topology
