"""Aligned trajectories must differ from their input by a rigid transform only.

Imaging after superposition wraps against a box that no longer matches the
rotated coordinates, translating whole molecules onto each other.
"""

# --- aligned trajectories must not gain atomic overlaps --------------------

def _triclinic_solvent_traj(tmp_path, n_wat=80, rot=0.7):
    """Two frames in a dodecahedron-shaped cell.

    Frame 1 has the solute rotated in place while the solvent stays put, which is
    what tumbling during MD looks like. Superposing frame 1 onto frame 0 therefore
    applies a net rotation to the solvent relative to the box.
    """
    import numpy as np, mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    # Real backbone residues: the topology is round-tripped through a PDB, so
    # bonds have to come from standard residue templates, not add_bond. Three
    # ALA give one peptide-bonded molecule (>3 atoms, the image_molecules anchor)
    # and three CA atoms for the superposition.
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
    # Solute near the middle of the cell, at plausible bond distances, and
    # deliberately NOT collinear: three points on a line give superpose no unique
    # rotation and mdtraj silently returns identity, which would make this fixture
    # exercise nothing.
    for k, (n, c_a, c, o) in enumerate(bb_atoms):
        base = np.array([1.20 + 0.30*np.cos(2.1*k),
                         1.50 + 0.30*np.sin(2.1*k),
                         0.80 + 0.22*k])
        xyz[:, n.index]   = base
        xyz[:, c_a.index] = base + [0.147, 0.0, 0.0]
        xyz[:, c.index]   = base + [0.147 + 0.153, 0.0, 0.0]
        xyz[:, o.index]   = base + [0.147 + 0.153, 0.123, 0.0]
    # solvent on a REGULAR fractional lattice so the input cannot start clashing
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
    """Superposition rotates coordinates but not unitcell_vectors, so imaging
    afterwards translates whole molecules by vectors that are wrong in the rotated
    frame -- measured on real data as 942 overlapping pairs appearing in a frame
    that had none, and 32-107 per extracted milestone driving NonbondedForce to
    +1e11 kJ/mol.

    Asserting on overlap COUNTS needs realistic solvent density to be reliable; the
    density-independent statement is that alignment may only apply a rigid
    transform, which preserves every interatomic distance.
    """
    import copy
    import mdtraj as md
    from autopath.utils import wrap_align_save_traj
    pdb, dcd, top = _triclinic_solvent_traj(tmp_path)
    # Baseline = the legitimate pre-superposition steps. Imaging BEFORE superposing
    # is allowed to move molecules by lattice vectors; only what happens AFTER the
    # rotation has to be rigid.
    src = md.load(dcd, top=pdb).center_coordinates().image_molecules(make_whole=True)

    # Fixture validity. This assertion exists because an earlier version of this
    # test passed while the bug was present: the solute was collinear, so mdtraj
    # could not derive a rotation, silently returned identity, and there was
    # nothing for the stray imaging pass to corrupt.
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
