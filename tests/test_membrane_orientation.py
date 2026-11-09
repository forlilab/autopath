"""Membrane orientation: PPM decides, the transform is verified rigid, nothing bends.

The synthetic systems here are cylinders of one-atom pseudo-residues with a hydrophobic
belt, because tests/data/membrane_peptide.pdb is a 60-residue fragment with no real
bilayer-spanning belt to find. The peptide is used only where a real topology is needed.

The file-level tests monkeypatch the frame rather than running PPM, because they are about
the file plumbing: which paths are written, what moves with what. The one test that needs a
real PPM is marked slow and skips when the binary is absent.
"""
import json
import os

import numpy as np
import pytest
from openmm import Vec3
from openmm.app import Element, PDBFile, Topology  # noqa: F401
import openmm.unit as openmmunit
from scipy.spatial.transform import Rotation

from autopath import membrane_orient as mo
from autopath.membrane_orient import (
    MembraneFrame,
    OrientationError,
    orient_to_membrane_frame,
)
from autopath.metadynamics.Funnel import rotation_angle_deg

DATA = os.path.join(os.path.dirname(__file__), "data")
PEPTIDE = os.path.join(DATA, "membrane_peptide.pdb")
_Z = np.array([0.0, 0.0, 1.0])


def slab_system(n_res=300, half=1.7, radius=2.0, z_span=4.0, seed=0):
    """A cylinder of pseudo-residues, hydrophobic within ``half`` of z=0."""
    rng = np.random.default_rng(seed)
    z = np.linspace(-z_span, z_span, n_res)
    angle = rng.uniform(0, 2 * np.pi, n_res)
    pos = np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)

    top = Topology()
    chain = top.addChain("A")
    carbon = Element.getBySymbol("C")
    for zi in z:
        name = "LEU" if abs(zi) < half else "SER"
        res = top.addResidue(name, chain)
        top.addAtom("CA", carbon, res)
    return top, pos


def transformed(pos, degrees=0.0, axis=(0.6, 0.8, 0.0), shift=(0.0, 0.0, 0.0)):
    """Rotate and shift. The default axis is perpendicular to z, so the tilt of the
    membrane normal equals ``degrees``; about any other axis it would be less."""
    rot = Rotation.from_rotvec(np.deg2rad(degrees) * np.asarray(axis, float)
                               / np.linalg.norm(axis)).as_matrix()
    return np.asarray(pos) @ rot.T + np.asarray(shift, float)


def write_pdb(path, top, pos, box=False):
    if box:
        top.setPeriodicBoxVectors(np.eye(3) * 9.0 * openmmunit.nanometer)
    with open(path, "w") as handle:
        PDBFile.writeFile(top, [Vec3(*p) for p in pos] * openmmunit.nanometer, handle)
    return path


def coords_nm(path):
    pdb = PDBFile(path)
    return np.array(pdb.positions.value_in_unit(openmmunit.nanometer))


# ------------------------------------------------------------------------ MembraneFrame

def a_frame(rotation=None, translation=None):
    return MembraneFrame(
        rotation=np.eye(3) if rotation is None else rotation,
        translation=np.zeros(3) if translation is None else translation,
        fit_rmsd=0.0,
    )


def test_apply_is_rigid():
    """Pairwise distances are the invariant that matters: nothing may bend."""
    rot = Rotation.from_rotvec([0.3, -0.5, 0.8]).as_matrix()
    frame = a_frame(rot, np.array([1.0, -2.0, 3.0]))
    _, pos = slab_system(n_res=50)
    before = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    after = frame.apply(pos)
    assert np.allclose(
        before, np.linalg.norm(after[:, None, :] - after[None, :, :], axis=-1), atol=1e-9
    )


def test_normal_is_derived_from_the_rotation():
    rot = Rotation.from_rotvec([0.2, 0.4, -0.1]).as_matrix()
    frame = a_frame(rot)
    assert np.allclose(frame.normal, rot.T @ _Z)
    assert np.allclose(frame.rotation @ frame.normal, _Z)


def test_is_identity():
    assert a_frame().is_identity()
    assert not a_frame(translation=np.array([0.0, 0.0, 1.0])).is_identity()
    assert not a_frame(Rotation.from_rotvec([0.0, 0.0, 0.5]).as_matrix()).is_identity()


def test_record_states_its_units_and_convention():
    """A transform without its convention is unusable by anyone but its author."""
    rec = a_frame(Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()).as_record()
    assert rec["units"] == "nanometer"
    assert rec["convention"] == "new = old @ rotation.T + translation"
    assert rec["normal_frame"] == "input"
    assert np.shape(rec["rotation"]) == (3, 3)
    assert rec["method"] == "ppm"
    json.dumps(rec)


# ----------------------------------------------------- recovering a backend's transform

def oriented_pair(tmp_path, degrees=35.0, shift=(1.0, -2.0, 0.5), mangle=None):
    """Write an input PDB and a transformed copy, as PPM would return it."""
    top, pos = slab_system()
    moved = transformed(pos, degrees=degrees, shift=shift)
    if mangle is not None:
        moved = mangle(moved)
    return (write_pdb(str(tmp_path / "in.pdb"), top, pos),
            write_pdb(str(tmp_path / "out.pdb"), top, moved),
            pos, moved)


def test_transform_is_recovered_from_an_oriented_copy(tmp_path):
    """This is how every backend answer is read: superpose the output onto the input."""
    inp, out, pos, moved = oriented_pair(tmp_path)
    rotation, translation, rmsd = mo._recover_transform(inp, out)
    # PDB carries 3 decimal places in angstroms, so 1e-3 nm is the floor.
    assert rmsd < 1e-3
    assert np.allclose(pos @ rotation.T + translation, moved, atol=1e-3)


def test_non_rigid_output_is_rejected(tmp_path):
    """A backend that bent the structure must not be trusted with a rigid transform."""
    def bend(xyz):
        xyz = xyz.copy()
        xyz[: len(xyz) // 2] += 0.8
        return xyz

    inp, out, _, _ = oriented_pair(tmp_path, mangle=bend)
    with pytest.raises(OrientationError, match="rigidly"):
        mo._recover_transform(inp, out)


def test_a_mirrored_structure_is_rejected(tmp_path):
    """A reflection superposes just as well and would make D-amino acids."""
    inp, out, _, _ = oriented_pair(tmp_path, degrees=0.0, shift=(0.0, 0.0, 0.0),
                                   mangle=lambda xyz: xyz * np.array([1.0, 1.0, -1.0]))
    with pytest.raises(OrientationError, match="reflection|rigidly"):
        mo._recover_transform(inp, out)


def test_too_few_matched_atoms_is_rejected(tmp_path):
    top, pos = slab_system(n_res=300)
    inp = write_pdb(str(tmp_path / "in.pdb"), top, pos)
    small_top, small_pos = slab_system(n_res=60)
    out = write_pdb(str(tmp_path / "out.pdb"), small_top, small_pos)
    # Renumber so almost nothing matches by (chain, resSeq, icode).
    text = open(out).read().replace(" A ", " B ", 1)
    open(out, "w").write(text)
    with pytest.raises(OrientationError):
        mo._recover_transform(inp, out)


def test_ca_map_skips_alternate_locations(tmp_path):
    """Pairing a residue against a different conformer would corrupt the fit silently."""
    path = str(tmp_path / "altloc.pdb")
    with open(path, "w") as handle:
        handle.write(
            "ATOM      1  CA ALEU A   1      １.000   2.000   3.000  1.00  0.00           C\n"
            .replace("１", "1")
        )
        handle.write(
            "ATOM      2  CA BLEU A   1       9.000   9.000   9.000  0.50  0.00           C\n"
        )
    found = mo._ca_map(path)
    assert len(found) == 1
    assert np.allclose(list(found.values())[0], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------- finding PPM

def test_ppm_path_comes_from_the_environment(tmp_path, monkeypatch):
    exe = tmp_path / "immers"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("AUTOPATH_PPM", str(exe))
    assert mo._find_ppm() == str(exe)


def test_an_empty_res_lib_is_refused(tmp_path, monkeypatch):
    """Fortran creates a missing res.lib empty, then PPM runs blind and still exits 0."""
    (tmp_path / "res.lib").write_text("")
    monkeypatch.setattr(mo, "_packmol_memgen_dir", lambda: None)
    monkeypatch.delenv("AMBERHOME", raising=False)
    assert mo._find_res_lib(str(tmp_path / "immers")) is None

    (tmp_path / "res.lib").write_text("ALA 1.0\n")
    assert mo._find_res_lib(str(tmp_path / "immers")) == str(tmp_path / "res.lib")


def test_missing_ppm_raises_with_instructions(tmp_path, monkeypatch):
    """No silent fallback exists any more, so the error has to say what to install."""
    monkeypatch.setattr(mo, "_find_ppm", lambda: None)
    top, pos = slab_system()
    with pytest.raises(OrientationError, match="AUTOPATH_PPM"):
        orient_to_membrane_frame(top, pos, work_dir=str(tmp_path))


def test_n_term_is_validated(tmp_path):
    top, pos = slab_system()
    with pytest.raises(ValueError, match="n_term"):
        orient_to_membrane_frame(top, pos, n_term="sideways", work_dir=str(tmp_path))


# ------------------------------------------------------- orient_membrane, on files

@pytest.fixture
def fake_ppm(monkeypatch):
    """Replace PPM with a known rotation, so the file plumbing can be tested alone."""
    rot = Rotation.from_rotvec(np.deg2rad(35.0) * np.array([0.6, 0.8, 0.0])).as_matrix()
    frame = MembraneFrame(rotation=rot, translation=np.array([0.3, -0.4, 1.2]),
                          fit_rmsd=1e-5)
    monkeypatch.setattr(mo, "orient_to_membrane_frame", lambda *a, **k: frame)
    return frame


def write_sdf(tmp_path, coords_ang, name="lig.sdf", n_records=1):
    """A minimal V2000 SDF at exactly the given angstrom coordinates."""
    coords_ang = np.atleast_2d(np.asarray(coords_ang, float))
    n = len(coords_ang)
    block = ["lig", "  autopath test", ""]
    block.append(f"{n:3d}{max(n - 1, 0):3d}  0  0  0  0  0  0  0  0999 V2000")
    for x, y, z in coords_ang:
        block.append(f"{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0")
    for i in range(1, n):
        block.append(f"{i:3d}{i + 1:3d}  1  0")
    block += ["M  END", "$$$$"]
    path = str(tmp_path / name)
    with open(path, "w") as handle:
        handle.write("\n".join(["\n".join(block)] * n_records) + "\n")
    return path


def sdf_coords(path):
    from rdkit import Chem
    out = []
    with open(path, "rb") as handle:
        for mol in Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False):
            assert mol is not None
            out.append(np.asarray(mol.GetConformer().GetPositions()))
    return out


def a_receptor(tmp_path, box=False):
    top, pos = slab_system()
    moved = transformed(pos, degrees=35.0, shift=(1.0, -2.0, 5.0))
    return write_pdb(str(tmp_path / "receptor.pdb"), top, moved, box=box), moved


def test_orient_membrane_writes_the_oriented_receptor(tmp_path, fake_ppm):
    receptor, before = a_receptor(tmp_path)
    out, _ = mo.orient_membrane(receptor, None, str(tmp_path / "sys"))
    assert out.endswith("receptor_oriented.pdb")
    assert np.allclose(coords_nm(out), fake_ppm.apply(before), atol=1e-3)


def test_orient_membrane_keeps_the_ligand_in_register(tmp_path, fake_ppm):
    """The whole point: one transform for every file, so a docked pose does not drift."""
    receptor, before_rec = a_receptor(tmp_path)
    pocket = before_rec[:8] * 10.0 + np.array([0.6, 0.0, 0.0])
    ligand = write_sdf(tmp_path, pocket)

    out_rec, out_ligs = mo.orient_membrane(
        receptor, [("UNK", ligand, None, False)], str(tmp_path / "sys")
    )

    before = np.linalg.norm(
        before_rec[:, None, :] * 10.0 - pocket[None, :, :], axis=-1
    )
    after_rec = coords_nm(out_rec) * 10.0
    after_lig = sdf_coords(out_ligs[0][1])[0]
    after = np.linalg.norm(after_rec[:, None, :] - after_lig[None, :, :], axis=-1)
    assert np.abs(after - before).max() < 1e-2


def test_orient_membrane_returns_the_ligand_tuple_shape(tmp_path, fake_ppm):
    receptor, _ = a_receptor(tmp_path)
    ligand = write_sdf(tmp_path, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    _, out = mo.orient_membrane(receptor, [("LIG", ligand, "CC", True)],
                                str(tmp_path / "sys"))
    assert len(out) == 1
    assert out[0][0] == "LIG" and out[0][2] == "CC" and out[0][3] is True
    assert out[0][1].endswith("_oriented.sdf")


def test_orient_membrane_accepts_a_bare_ligand_path(tmp_path, fake_ppm):
    receptor, _ = a_receptor(tmp_path)
    ligand = write_sdf(tmp_path, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    _, out = mo.orient_membrane(receptor, ligand, str(tmp_path / "sys"))
    assert isinstance(out, str) and out.endswith("_oriented.sdf")


def test_orient_membrane_keeps_every_sdf_record(tmp_path, fake_ppm):
    """SDMolSupplier drops records past 64 KiB; the reader here must not."""
    receptor, _ = a_receptor(tmp_path)
    ligand = write_sdf(tmp_path, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], n_records=40)
    _, out = mo.orient_membrane(receptor, ligand, str(tmp_path / "sys"))
    assert len(sdf_coords(out)) == 40


def test_orient_membrane_moves_a_pdb_ligand(tmp_path, fake_ppm):
    receptor, _ = a_receptor(tmp_path)
    lig_top, lig_pos = slab_system(n_res=4, z_span=0.3)
    lig_path = write_pdb(str(tmp_path / "colig.pdb"), lig_top, lig_pos)
    _, out = mo.orient_membrane(receptor, lig_path, str(tmp_path / "sys"))
    assert out.endswith("_oriented.pdb")
    assert np.allclose(coords_nm(out), fake_ppm.apply(lig_pos), atol=1e-3)


def test_orient_membrane_writes_the_record(tmp_path, fake_ppm):
    receptor, _ = a_receptor(tmp_path)
    out_dir = str(tmp_path / "sys")
    mo.orient_membrane(receptor, None, out_dir)
    record = json.load(open(os.path.join(out_dir, "membrane_orientation.json")))
    assert record["method"] == "ppm"
    assert record["convention"] == "new = old @ rotation.T + translation"
    assert record["receptor"] == os.path.abspath(receptor)


def test_orient_membrane_drops_a_stale_unit_cell(tmp_path, fake_ppm):
    """A rotation invalidates CRYST1, and addMembrane would size the box from it."""
    receptor, _ = a_receptor(tmp_path, box=True)
    assert any(l.startswith("CRYST1") for l in open(receptor))
    out, _ = mo.orient_membrane(receptor, None, str(tmp_path / "sys"))
    assert not any(l.startswith("CRYST1") for l in open(out))


def test_orient_membrane_writes_remarks(tmp_path, fake_ppm):
    receptor, _ = a_receptor(tmp_path)
    out, _ = mo.orient_membrane(receptor, None, str(tmp_path / "sys"))
    head = "".join(open(out).readlines()[:8])
    assert "AUTOPATH MEMBRANE ORIENTATION" in head
    assert "method=ppm" in head


def test_orient_membrane_leaves_an_aligned_system_alone(tmp_path, monkeypatch):
    """Nothing is written when there is nothing to do, so paths stay as they were."""
    identity = MembraneFrame(rotation=np.eye(3), translation=np.zeros(3),
                             fit_rmsd=0.0)
    monkeypatch.setattr(mo, "orient_to_membrane_frame", lambda *a, **k: identity)
    receptor, _ = a_receptor(tmp_path)
    ligand = write_sdf(tmp_path, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    out_dir = str(tmp_path / "sys")

    out_rec, out_lig = mo.orient_membrane(receptor, [("UNK", ligand, None, False)],
                                          out_dir)
    assert out_rec == receptor
    assert out_lig[0][1] == ligand
    assert os.listdir(out_dir) == []


def test_orient_membrane_rejects_an_unsupported_ligand_format(tmp_path, fake_ppm):
    receptor, _ = a_receptor(tmp_path)
    bad = str(tmp_path / "lig.mol2")
    open(bad, "w").write("@<TRIPOS>MOLECULE\n")
    with pytest.raises(OrientationError, match="mol2"):
        mo.orient_membrane(receptor, bad, str(tmp_path / "sys"))


# ------------------------------------------------------------------------ with real PPM

@pytest.mark.slow
@pytest.mark.skipif(mo._find_ppm() is None, reason="no immers binary")
def test_real_ppm_on_the_peptide(tmp_path):
    pdb = PDBFile(PEPTIDE)
    pos = np.array(pdb.positions.value_in_unit(openmmunit.nanometer))
    try:
        frame = orient_to_membrane_frame(pdb.topology, pos, work_dir=str(tmp_path))
    except OrientationError:
        pytest.skip("PPM declined this fragment, which is a legitimate answer")
    assert frame.fit_rmsd < mo.FIT_RMSD_TOL_NM
    assert np.linalg.det(frame.rotation) == pytest.approx(1.0)
    assert not any(name.startswith("ppm_") for name in os.listdir(tmp_path))
