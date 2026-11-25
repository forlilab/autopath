"""Regression tests for milestone extraction."""
import os
import numpy as np
import pandas as pd
import pytest
import MDAnalysis as mda

from autopath.utils import _reimaging_ligand
from autopath.clustering import order_milestones_bound_to_unbound


def _lig_universe():
    """One protein CA plus a 2-atom ligand residue (heavy atom + its hydrogen)."""
    u = mda.Universe.empty(3, n_residues=2, atom_resindex=[0, 1, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "C1", "H1"])
    u.add_TopologyAttr("resname", ["ALA", "LIG"])
    u.add_TopologyAttr("resid", [1, 2])
    u.dimensions = [40, 40, 40, 90, 90, 90]
    # ligand sits a box length away from the protein, so re-imaging must fire
    u.atoms.positions = np.array([[0., 0., 0.], [34., 0., 0.], [35., 0., 0.]])
    return u


def test_reimaging_moves_ligand_hydrogens_with_their_heavy_atoms():
    """Re-imaging via a heavy-atom selection also moves the ligand's hydrogens."""
    u = _lig_universe()
    assert _reimaging_ligand(u, "resname LIG and not name H*") is True

    c1, h1 = u.atoms[1].position, u.atoms[2].position
    assert np.linalg.norm(h1 - c1) == pytest.approx(1.0, abs=1e-4)
    # the ligand actually moved
    assert c1[0] == pytest.approx(-6.0, abs=1e-4)
    assert h1[0] == pytest.approx(-5.0, abs=1e-4)


def test_reimaging_leaves_protein_untouched():
    u = _lig_universe()
    _reimaging_ligand(u, "resname LIG and not name H*")
    np.testing.assert_allclose(u.atoms[0].position, [0., 0., 0.], atol=1e-4)


def test_reimaging_noop_when_ligand_already_at_minimum_image():
    u = _lig_universe()
    u.atoms.positions = np.array([[0., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
    assert _reimaging_ligand(u, "resname LIG and not name H*") is False
    np.testing.assert_allclose(u.atoms[1].position, [3., 0., 0.], atol=1e-4)


# --- milestone direction anchoring -----------------------------------------

def _write_state_pair(tmp_path, seps):
    """Write one .pdb per requested pocket-ligand separation (in nm).

    Atom 0 is the pocket, atom 1 the ligand; PDB coordinates are Angstrom.
    """
    paths = []
    for i, sep in enumerate(seps):
        p = tmp_path / f"milestone_{i + 1}_frame_{i}.pdb"
        with open(p, "w") as f:
            f.write(f"ATOM      1  C1  POC A   1    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            f.write(f"ATOM      2  C2  LIG A   2    {sep * 10:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            f.write("END\n")
        paths.append(str(p))
    return paths


def test_bound_to_unbound_order_is_left_alone(tmp_path):
    paths = _write_state_pair(tmp_path, [0.4, 0.9, 1.5])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == paths


def test_unbound_to_bound_order_is_reversed(tmp_path):
    """PC1's sign is arbitrary, so the centroid order is as likely to be reversed."""
    paths = _write_state_pair(tmp_path, [1.5, 0.9, 0.4])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == list(reversed(paths))


def test_single_milestone_is_returned_unchanged(tmp_path):
    paths = _write_state_pair(tmp_path, [0.7])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == paths


# --- no loadState anywhere -------------------------------------------------

def test_no_module_calls_load_state():
    """loadState is forbidden: it restores global parameters by name and keeps stale values."""
    import autopath
    root = os.path.dirname(os.path.abspath(autopath.__file__))
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                for lineno, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    if ".loadState(" in code:
                        offenders.append(f"{os.path.relpath(path, root)}:{lineno}")
    assert offenders == [], f"loadState is forbidden; found at {offenders}"


# --- surviving-trajectory table ---------------------------------------------

def _write_processed_csv(path, rows, with_path_ok=True):
    import pandas as pd
    cols = ["step", "trajname", "path"] + (["path_ok"] if with_path_ok else [])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)


def test_read_trajname_to_path_keeps_only_surviving_rows(tmp_path):
    from autopath.utils import read_trajname_to_path

    _write_processed_csv(
        tmp_path / "sMD_processed_data.csv",
        [[0, "sMD_replica-1_v0.01_forward", "path-0_v0.01", True],
         [1, "sMD_replica-1_v0.01_forward", "path-0_v0.01", True],
         [0, "sMD_replica-2_v0.01_forward", "path-1_v0.01", False]],
    )
    mapping = read_trajname_to_path(str(tmp_path))
    assert mapping == {"sMD_replica-1_v0.01_forward": "path-0_v0.01"}


def test_read_trajname_to_path_treats_missing_path_ok_as_all_kept(tmp_path):
    from autopath.utils import read_trajname_to_path

    _write_processed_csv(
        tmp_path / "sMD_processed_data.csv",
        [[0, "a", "path-0"], [0, "b", "path-1"]],
        with_path_ok=False,
    )
    assert read_trajname_to_path(str(tmp_path)) == {"a": "path-0", "b": "path-1"}


def test_read_trajname_to_path_is_empty_when_the_table_is_missing(tmp_path):
    from autopath.utils import read_trajname_to_path

    assert read_trajname_to_path(str(tmp_path)) == {}


# --- medoid filtering keys on the trajectory, not the stale label ----------

def test_medoid_filter_keeps_a_medoid_whose_label_is_stale():
    """Medoids are kept by surviving trajectory, even when their path label is stale."""
    from autopath.utils import filter_medoids_by_surviving_trajectories

    medoid_to_path = {"repA": "path-0_v0.02", "repB": "path-1_v0.02"}
    surviving = {"repA": "path-0_v0.02", "repB": "path-0_v0.02"}  # repB relabelled

    kept, dropped, disagree = filter_medoids_by_surviving_trajectories(
        medoid_to_path, surviving)

    assert dropped == {}, "a medoid whose trajectory passed QC must not be dropped"
    assert kept == {"repA": "path-0_v0.02", "repB": "path-0_v0.02"}
    assert disagree == {"repB": ("path-1_v0.02", "path-0_v0.02")}


def test_medoid_filter_drops_a_medoid_whose_trajectory_failed_qc():
    from autopath.utils import filter_medoids_by_surviving_trajectories

    kept, dropped, disagree = filter_medoids_by_surviving_trajectories(
        {"repA": "path-0", "repGone": "path-1"}, {"repA": "path-0"})
    assert kept == {"repA": "path-0"}
    assert dropped == {"repGone": "path-1"}
    assert disagree == {}


def test_medoid_filter_is_a_noop_without_analysis_output():
    """An empty surviving map means 'unknown', never 'nothing survived'."""
    from autopath.utils import filter_medoids_by_surviving_trajectories

    m = {"repA": "path-0", "repB": "path-1"}
    kept, dropped, disagree = filter_medoids_by_surviving_trajectories(m, {})
    assert kept == m and dropped == {} and disagree == {}


# --- aligned trajectories must not gain atomic overlaps --------------------

def _triclinic_solvent_traj(tmp_path, n_wat=80, rot=0.7):
    """Two frames in a dodecahedron cell; in frame 1 only the solute is rotated, so
    superposing onto frame 0 rotates the solvent relative to the box."""
    import numpy as np, mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    # standard residues, since bonds must survive a PDB round-trip; they form one
    # molecule for image_molecules to anchor on
    ca = []
    bb_atoms = []
    for i in range(4):
        r = top.add_residue('ALA', ch)
        n = top.add_atom('N', md.element.nitrogen, r)
        c_a = top.add_atom('CA', md.element.carbon, r)
        c = top.add_atom('C', md.element.carbon, r)
        o = top.add_atom('O', md.element.oxygen, r)
        ca.append(c_a); bb_atoms.append((n, c_a, c, o))
    wat = []
    for i in range(n_wat):
        r = top.add_residue('HOH', ch)
        o = top.add_atom('O', md.element.oxygen, r)
        h1 = top.add_atom('H1', md.element.hydrogen, r)
        h2 = top.add_atom('H2', md.element.hydrogen, r)
        top.add_bond(o, h1); top.add_bond(o, h2)
        wat.append((o.index, h1.index, h2.index))
    L = 3.0
    box = np.array([[L, 0, 0], [0, L, 0], [L/2, L/2, L*0.7071]])
    n_at = top.n_atoms
    xyz = np.zeros((2, n_at, 3))
    # non-collinear solute, otherwise superpose has no unique rotation
    for k, (n, c_a, c, o) in enumerate(bb_atoms):
        base = np.array([1.20 + 0.30*np.cos(2.1*k),
                         1.50 + 0.30*np.sin(2.1*k),
                         0.80 + 0.22*k])
        xyz[:, n.index]   = base
        xyz[:, c_a.index] = base + [0.147, 0.0, 0.0]
        xyz[:, c.index]   = base + [0.147 + 0.153, 0.0, 0.0]
        xyz[:, o.index]   = base + [0.147 + 0.153, 0.123, 0.0]
    # solvent on a regular fractional lattice so the input has no clashes
    g = np.array([(i, j, k) for i in range(5) for j in range(4) for k in range(4)],
                 dtype=float)
    f = (g + 0.5) / np.array([5.0, 4.0, 4.0])
    for i, (o, h1, h2) in enumerate(wat):
        c = f[i] @ box
        xyz[:, o] = c
        xyz[:, h1] = c + [0.0957, 0, 0]
        xyz[:, h2] = c + [-0.024, 0.0927, 0]
    R = np.array([[np.cos(rot), -np.sin(rot), 0],
                  [np.sin(rot),  np.cos(rot), 0], [0, 0, 1]])
    idx = [a.index for grp in bb_atoms for a in grp]
    pivot = xyz[1, idx].mean(axis=0)
    xyz[1, idx] = (xyz[1, idx] - pivot) @ R.T + pivot     # solute rotates only
    t = md.Trajectory(xyz, top)
    t.unitcell_vectors = np.repeat(box[None, :, :], 2, axis=0)
    pdb = str(tmp_path / 'sys.pdb'); dcd = str(tmp_path / 'traj.dcd')
    t[0].save_pdb(pdb); t.save_dcd(dcd)
    return pdb, dcd, top


def _rigid_rmsd(A, B):
    """RMSD of B onto A after optimal rigid superposition (nm)."""
    import numpy as np
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Bc.T @ Ac)
    d = np.sign(np.linalg.det(V @ W))
    R = V @ np.diag([1, 1, d]) @ W
    return float(np.sqrt((((Bc @ R) - Ac) ** 2).sum(axis=1).mean()))


def test_alignment_applies_only_a_rigid_transform(tmp_path):
    """Alignment applies only a rigid transform; imaging after superposition would not,
    because the unit cell vectors are not rotated with the coordinates."""
    import copy
    import mdtraj as md
    from autopath.utils import wrap_align_save_traj
    pdb, dcd, top = _triclinic_solvent_traj(tmp_path)
    # baseline: imaging before superposition may shift molecules by lattice vectors
    src = md.load(dcd, top=pdb).center_coordinates().image_molecules(make_whole=True)

    # fixture check: imaging after the rotation must visibly damage the frame
    sel = src.topology.select('name CA')
    rot = copy.deepcopy(src).superpose(src[0], atom_indices=sel)
    buggy = copy.deepcopy(rot).image_molecules(make_whole=True)
    damage = max(_rigid_rmsd(rot.xyz[i], buggy.xyz[i]) for i in range(src.n_frames))
    assert damage > 0.05, (
        f"fixture does not exercise imaging-after-rotation (damage {damage*10:.3f} A); "
        "a passing result here would be meaningless")

    out = wrap_align_save_traj(dcd, pdb, remove_original=False,
                               align_select='name CA', out_dir=str(tmp_path))
    got = md.load(out[0], top=pdb)
    assert got.n_frames == src.n_frames
    worst = max(_rigid_rmsd(src.xyz[i], got.xyz[i]) for i in range(src.n_frames))
    assert worst < 1e-3, (
        f"alignment moved atoms by {worst*10:.2f} A beyond any rigid transform; "
        "imaging must happen before superposition, never after")


def test_path_readers_use_pulls_csv(tmp_path):
    from autopath.utils import read_surviving_paths, read_trajname_to_path
    pd.DataFrame({"trajname": ["a", "b", "c"], "path": ["path-0", "path-1", "path-1"],
                  "path_ok": [True, True, False]}).to_csv(tmp_path / "pulls.csv", index=False)
    assert read_trajname_to_path(str(tmp_path)) == {"a": "path-0", "b": "path-1"}
    assert read_surviving_paths(str(tmp_path)) == {"path-0", "path-1"}
