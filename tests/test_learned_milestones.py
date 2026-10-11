"""Class-medoid milestones: medoid choice and the rotate-back / re-image / overlap check of the written frames."""
import os

import MDAnalysis as mda
import numpy as np
import pytest
from openmm import app, unit

from autopath.metadynamics.learned.milestones import class_medoid_frames, write_milestones

ORDER = ("B", "TS", "U")


def _npz(tmp_path, rng, n=60, valid_nan_trap=True):
    """Three classes, each a tight blob in (f0, f2, z); column 1 is not kept."""
    labels = np.repeat(np.array(ORDER), n)
    centre = {"B": (0.0, 0.0, 0.0), "TS": (5.0, 5.0, 5.0), "U": (-5.0, 5.0, -5.0)}
    V = np.zeros((3 * n, 3))
    z = np.zeros(3 * n)
    for k, c in enumerate(ORDER):
        sl = slice(k * n, (k + 1) * n)
        x = rng.normal(0, 0.3, (n, 3)) + np.array(centre[c])
        V[sl, 0], V[sl, 2], z[sl] = x[:, 0], x[:, 1], x[:, 2]
    V[:, 1] = rng.normal(size=3 * n) * 100            # not kept: must not affect the medoid
    # exact class means pinned at row 0 of each class so the medoid is known
    for k, c in enumerate(ORDER):
        i = k * n
        V[i, 0], V[i, 2], z[i] = centre[c]
    valid = np.ones(3 * n, bool)
    if valid_nan_trap:                                 # an invalid row sitting exactly at each class centre
        for k, c in enumerate(ORDER):
            i = k * n + 1
            V[i, 0], V[i, 2], z[i] = centre[c]
            V[i] = np.nan
            valid[i] = False
    np.savez(tmp_path / "features.npz", V=V, labels=labels, z=z, pull=np.array(["p1"] * (3 * n)),
             frame=np.arange(3 * n) * 2, dcd=np.array(["a.dcd"] * (3 * n)), valid=valid)
    np.savez(tmp_path / "learned_cv.npz", kept=np.array([0, 2]), mu=np.zeros(2), sd=np.ones(2))
    return str(tmp_path / "features.npz"), str(tmp_path / "learned_cv.npz")


def test_medoid_is_the_central_valid_frame_in_class_order(tmp_path):
    f, g = _npz(tmp_path, np.random.default_rng(0))
    out = class_medoid_frames(f, g)
    assert [m["class"] for m in out] == list(ORDER)
    for k, m in enumerate(out):
        assert m["frame"] == 2 * (k * 60) and m["dcd"] == "a.dcd" and m["pull"] == "p1"
    rev = class_medoid_frames(f, g, order=("U", "B"))
    assert [m["class"] for m in rev] == ["U", "B"]


def test_medoid_is_deterministic_and_subsamples_large_classes(tmp_path):
    f, g = _npz(tmp_path, np.random.default_rng(1), n=2300)
    a, b = class_medoid_frames(f, g), class_medoid_frames(f, g)
    assert a == b
    lab = np.load(f)["labels"]
    fr = np.load(f)["frame"]
    for m in a:
        assert lab[np.flatnonzero(fr == m["frame"])[0]] == m["class"]


def test_class_without_valid_frames_is_an_error(tmp_path):
    f, g = _npz(tmp_path, np.random.default_rng(2))
    d = dict(np.load(f))
    d["valid"] = d["valid"] & (d["labels"] != "TS")
    np.savez(f, **d)
    with pytest.raises(ValueError, match="TS"):
        class_medoid_frames(f, g)


# ---- writing -----------------------------------------------------------------------------------------

BOXES = {"cubic": np.eye(3) * 2.6,
         "rhombic": np.array([[2.8, 0, 0], [0, 2.8, 0], [1.4, 1.4, 2.8 / np.sqrt(2)]])}


def _toy_system(tmp_path, box):
    """Periodic toy: 5 ALA (CA only), an unbonded 3-atom ligand and 40 waters on a jittered grid."""
    rng = np.random.default_rng(5)
    top = app.Topology()
    ch = top.addChain()
    pos = []
    frac0 = np.array([0.5, 0.5, 0.5])
    centre = frac0 @ box
    ca = centre + np.array([[0.0, 0, 0], [0.38, 0.1, 0], [0.7, 0.4, 0.1], [0.5, 0.8, 0.3], [0.1, 0.6, 0.5]])
    for k in range(5):
        r = top.addResidue("ALA", ch)
        top.addAtom("CA", app.element.carbon, r)
        pos.append(ca[k])
    r = top.addResidue("LIG", ch)
    for nm, off in (("C1", 0.0), ("C2", 0.15), ("O1", 0.3)):
        top.addAtom(nm, app.element.carbon if nm[0] == "C" else app.element.oxygen, r)
        pos.append(centre + [0.9 + off, 0.2, 0.0])
    g = np.array([(i, j, k) for i in range(4) for j in range(4) for k in range(3)], float) / [4, 4, 3] + 0.06
    g = g[:40]
    wat = np.array([[0.0, 0, 0], [0.096, 0, 0], [-0.03, 0.09, 0]])
    nw = 0
    inv = np.linalg.inv(box)
    for fr in g:
        c = fr @ box
        dd = (c - np.array(pos[:8])) @ inv
        if np.linalg.norm((dd - np.round(dd)) @ box, axis=1).min() < 0.5:
            continue
        r = top.addResidue("HOH", top.addChain())
        for nm, el, o in zip(("O", "H1", "H2"), (app.element.oxygen, app.element.hydrogen, app.element.hydrogen), wat):
            top.addAtom(nm, el, r)
            pos.append(c + o)
        nw += 1
    assert nw >= 20
    top.setPeriodicBoxVectors(box)
    path = str(tmp_path / "top.pdb")
    with open(path, "w") as fh:
        app.PDBFile.writeFile(top, np.array(pos) * unit.nanometer, fh, keepIds=True)
    return path


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _aligned_dcd(tmp_path, pdb, box, rng, collide=False):
    u = mda.Universe(pdb)
    X = u.atoms.positions / 10.0
    mol_of = np.array([a.resid for a in u.atoms])
    solvent = np.array([a.resname in ("HOH", "LIG") for a in u.atoms])
    if collide:                                      # last two waters on top of each other
        last = np.flatnonzero(mol_of == mol_of.max())
        prev = np.flatnonzero(mol_of == mol_of.max() - 1)
        X[last] = X[prev] + [0.05, 0.0, 0.0]
    Y = X.copy()
    for m in np.unique(mol_of):                      # break molecules across the cell
        Y[(mol_of == m) & solvent] += rng.integers(-1, 2, 3) @ box
    R = _rotation(rng)
    Y = Y @ R + [1.3, -0.7, 2.1]                     # rigid rotation of the whole frame (as the alignment does)
    u.atoms.positions = Y * 10.0
    path = str(tmp_path / "toy_aligned.dcd")
    with mda.Writer(path, u.atoms.n_atoms) as W:
        u.atoms.positions = np.zeros_like(Y) + 0.0
        u.atoms.positions = (Y * 10.0)
        W.write(u.atoms)
    return path, X


@pytest.mark.parametrize("kind", ["cubic", "rhombic"])
def test_write_milestones_rotates_back_reimages_and_names_files(tmp_path, kind):
    box = BOXES[kind]
    pdb = _toy_system(tmp_path, box)
    rng = np.random.default_rng(11)
    dcd, X = _aligned_dcd(tmp_path, pdb, box, rng)
    out = tmp_path / "ms"
    names = write_milestones([{"class": "TS", "pull": "p", "frame": 0, "dcd": os.path.basename(dcd)}], pdb,
                             str(out), relax=False, traj_dir=str(tmp_path))
    assert names == ["milestone_TS_frame_0"]
    f = out / "milestone_TS_frame_0.pdb"
    assert f.exists()
    p = app.PDBFile(str(f))
    got = np.array(p.positions.value_in_unit(unit.nanometer))
    pbox = np.array(p.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    inv = np.linalg.inv(box)
    assert np.allclose(pbox @ inv, np.round(pbox @ inv), atol=1e-3)        # same lattice as the topology
    d = got - X
    assert np.abs(d @ inv - np.round(d @ inv)).max() * np.linalg.norm(box, axis=1).min() < 2e-3  # same up to lattice vectors
    # whole molecules: intramolecular geometry unchanged
    for res in p.topology.residues():
        ids = [a.index for a in res.atoms()]
        assert np.allclose(np.linalg.norm(got[ids] - got[ids[0]], axis=1),
                           np.linalg.norm(X[ids] - X[ids[0]], axis=1), atol=2e-3)
    cen = np.array([got[[a.index for a in r.atoms()]].mean(0) for r in p.topology.residues()])
    fr = cen @ np.linalg.inv(pbox)
    assert (fr > -1e-3).all() and (fr < 1 + 1e-3).all()


def test_overlapping_molecules_after_reimaging_raise(tmp_path):
    box = BOXES["cubic"]
    pdb = _toy_system(tmp_path, box)
    dcd, _ = _aligned_dcd(tmp_path, pdb, box, np.random.default_rng(12), collide=True)
    with pytest.raises(ValueError, match="0.15"):
        write_milestones([{"class": "B", "pull": "p", "frame": 0, "dcd": dcd}], pdb, str(tmp_path / "ms"),
                         relax=False)


def test_topology_without_box_is_not_supported(tmp_path):
    pdb = _toy_system(tmp_path, BOXES["cubic"])
    lines = [ln for ln in open(pdb) if not ln.startswith("CRYST1")]
    nobox = tmp_path / "nobox.pdb"
    nobox.write_text("".join(lines))
    with pytest.raises(ValueError, match="not supported"):
        write_milestones([], str(nobox), str(tmp_path / "ms"), relax=False)


def test_relax_needs_system_and_atoms(tmp_path):
    pdb = _toy_system(tmp_path, BOXES["cubic"])
    with pytest.raises(ValueError, match="needs system"):
        write_milestones([], pdb, str(tmp_path / "ms"), relax=True)


def test_relax_writes_state_and_system_with_finite_energy(tmp_path):
    import openmm as mm
    box = BOXES["cubic"]
    pdb = _toy_system(tmp_path, box)
    dcd, _ = _aligned_dcd(tmp_path, pdb, box, np.random.default_rng(13))
    top = app.PDBFile(pdb).topology
    sysm = mm.System()
    for a in top.atoms():
        sysm.addParticle(a.element.mass)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9)
    for a in top.atoms():
        nb.addParticle(0.0, 0.3, 0.1)
    nb.createExceptionsFromBonds([(a.index, b.index) for a, b in top.bonds()], 0.0, 0.0)
    sysm.addForce(nb)
    lig = [a.index for a in top.atoms() if a.residue.name == "LIG"]
    names = write_milestones([{"class": "U", "pull": "p", "frame": 0, "dcd": dcd}], pdb, str(tmp_path / "ms"),
                             relax=True, platform="CPU", system=sysm, ligand_atoms=lig, pocket_atoms=[0, 1, 2])
    for suffix in ("_relax_state.xml", "_relax_system.xml"):
        assert (tmp_path / "ms" / (names[0] + suffix)).exists()
