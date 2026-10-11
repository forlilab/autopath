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


def test_medoid_does_not_depend_on_class_order(tmp_path):
    f, g = _npz(tmp_path, np.random.default_rng(3), n=2300)
    a = {m["class"]: m for m in class_medoid_frames(f, g)}
    b = {m["class"]: m for m in class_medoid_frames(f, g, order=("U", "TS", "B"))}
    assert a == b


def test_class_without_valid_frames_is_an_error(tmp_path):
    f, g = _npz(tmp_path, np.random.default_rng(2))
    d = dict(np.load(f))
    d["valid"] = d["valid"] & (d["labels"] != "TS")
    np.savez(f, **d)
    with pytest.raises(ValueError, match="TS"):
        class_medoid_frames(f, g)


# ---- writing -----------------------------------------------------------------------------------------

L_OWN = 2.4                                         # the frame's own (shrunken) brick
L_TOP = 2.6                                         # topology box, deliberately different
WAT = np.array([[0.0, 0, 0], [0.096, 0, 0], [-0.03, 0.09, 0]])


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()


def _lab_frame(rng, box=L_OWN):
    """Protein (one 30-atom bonded residue), a 3-atom ligand and Poisson-disc waters filling the brick."""
    prot = np.cumsum(rng.normal(0, 0.05, (30, 3)), axis=0) + np.array([box / 2] * 3) + [0.4, 0.4, 0.4]
    prot += (np.array([box / 2] * 3) - prot.mean(0))
    lig = prot.mean(0) + np.array([0.7, 0.2, 0.0]) + np.array([[0, 0, 0], [0.15, 0, 0], [0.3, 0, 0]])
    solute = np.vstack([prot, lig])
    wat = []
    cand = rng.uniform(0, box, (30000, 3))
    # candidates sitting on the brick faces make the brick fit well determined
    cand[np.arange(6000), rng.integers(0, 3, 6000)] = rng.choice([5e-4, box - 5e-4], 6000)
    for c in cand:
        d = solute - c
        if np.linalg.norm(d - box * np.round(d / box), axis=1).min() < 0.5:
            continue
        if wat:
            d = np.array(wat) - c
            if np.linalg.norm(d - box * np.round(d / box), axis=1).min() < 0.27:
                continue
        wat.append(c)
    wat = np.array(wat)
    return prot, lig, wat


def _toy_top(tmp_path, n_wat, box=L_TOP):
    top = app.Topology()
    ch = top.addChain()
    r = top.addResidue("PRT", ch)
    atoms = [top.addAtom(f"C{i}", app.element.carbon, r) for i in range(30)]
    for a, b in zip(atoms[:-1], atoms[1:]):
        top.addBond(a, b)
    r = top.addResidue("LIG", ch)
    for nm in ("C1", "C2", "O1"):
        top.addAtom(nm, app.element.carbon if nm[0] == "C" else app.element.oxygen, r)
    for _ in range(n_wat):
        r = top.addResidue("HOH", top.addChain())
        for nm, el in (("O", app.element.oxygen), ("H1", app.element.hydrogen), ("H2", app.element.hydrogen)):
            top.addAtom(nm, el, r)
    top.setPeriodicBoxVectors(np.eye(3) * box)
    path = str(tmp_path / "top.pdb")
    with open(path, "w") as fh:
        app.PDBFile.writeFile(top, np.zeros((top.getNumAtoms(), 3)) * unit.nanometer, fh, keepIds=True)
    return path


def _xyz(prot, lig, wat):
    return np.vstack([prot, lig, (wat[:, None] + (WAT - WAT.mean(0))[None]).reshape(-1, 3)])


def _aligned_dcd(tmp_path, pdb, X_lab, R_align):
    """Lab frame (imaged into its brick) -> aligned frame: rigid rotation + translation."""
    u = mda.Universe(pdb)
    Y = X_lab @ R_align + [1.3, -0.7, 2.1]
    path = str(tmp_path / "toy_aligned.dcd")
    u.dimensions = [L_OWN * 10] * 3 + [90.0] * 3
    with mda.Writer(path, u.atoms.n_atoms) as W:
        u.atoms.positions = Y * 10.0
        W.write(u.atoms)
    return path


def _setup(tmp_path, seed=11, deg=12.0):
    rng = np.random.default_rng(seed)
    prot, lig, wat = _lab_frame(rng)
    X = _xyz(prot, lig, wat)
    pdb = _toy_top(tmp_path, len(wat))
    R_align = _rot([1, 2, -1], deg)
    return rng, pdb, X, _aligned_dcd(tmp_path, pdb, X, R_align), R_align


def test_brick_fit_recovers_the_lab_rotation_and_uses_the_frames_own_box(tmp_path):
    from autopath.metadynamics.learned.milestones import brick_fit
    from scipy.spatial.transform import Rotation
    rng, pdb, X, dcd, R_align = _setup(tmp_path)
    names, infos = write_milestones([{"class": "TS", "pull": "p", "frame": 0, "dcd": os.path.basename(dcd)}],
                                    pdb, str(tmp_path / "ms"), relax=False, traj_dir=str(tmp_path),
                                    return_info=True)
    assert names == ["milestone_TS_frame_0"] and (tmp_path / "ms" / "milestone_TS_frame_0.pdb").exists()
    info = infos[0]
    assert info["angle_deg"] == pytest.approx(12.0, abs=0.1)            # the applied rotation undoes R_align
    assert info["residual_nm"] < 1e-2 and info["dmin_nm"] > 0.15
    assert np.allclose(info["box"], np.eye(3) * L_OWN, atol=1e-4)
    p = app.PDBFile(str(tmp_path / "ms" / "milestone_TS_frame_0.pdb"))
    assert np.allclose(np.array(p.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)),
                       np.eye(3) * L_OWN, atol=1e-3)
    got = np.array(p.positions.value_in_unit(unit.nanometer))
    # recovered coordinates equal the lab frame up to a rigid translation and lattice vectors
    d = got - X
    d -= d[:30].mean(0)
    assert np.abs(d - L_OWN * np.round(d / L_OWN)).max() < 3e-3
    # direct check of the fitted rotation
    Xa = X @ R_align
    water_mols = np.arange(33, len(X)).reshape(-1, 3)
    R, resid = brick_fit(Xa, np.full(3, L_OWN / 2), list(range(30)), water_mols)
    assert np.degrees(Rotation.from_matrix(R_align @ R).magnitude()) < 0.1 and resid < 1e-2
    for res in p.topology.residues():
        ids = [a.index for a in res.atoms()]
        assert np.allclose(np.linalg.norm(got[ids] - got[ids[0]], axis=1),
                           np.linalg.norm(X[ids] - X[ids[0]], axis=1), atol=2e-3)
    cen = np.array([got[[a.index for a in r.atoms()]].mean(0) for r in p.topology.residues()])
    assert (cen > -1e-3).all() and (cen < L_OWN + 1e-3).all()


def test_frame_whose_waters_cannot_be_fit_is_not_supported(tmp_path):
    rng = np.random.default_rng(14)
    prot, lig, wat = _lab_frame(rng)
    wat = rng.uniform(-2.0, 2.0 + L_OWN, wat.shape)                    # scrambled: far outside any brick
    pdb = _toy_top(tmp_path, len(wat))
    dcd = _aligned_dcd(tmp_path, pdb, _xyz(prot, lig, wat), _rot([0, 0, 1], 5.0))
    with pytest.raises(ValueError, match="not supported"):
        write_milestones([{"class": "B", "pull": "p", "frame": 0, "dcd": dcd}], pdb, str(tmp_path / "ms"),
                         relax=False)


def test_overlapping_molecules_after_reimaging_raise(tmp_path):
    rng = np.random.default_rng(15)
    prot, lig, wat = _lab_frame(rng)
    wat[-1] = wat[-2] + [0.05, 0.0, 0.0]                              # last water on top of the one before
    pdb = _toy_top(tmp_path, len(wat))
    dcd = _aligned_dcd(tmp_path, pdb, _xyz(prot, lig, wat), _rot([1, 0, 0], 6.0))
    with pytest.raises(ValueError, match="0.15"):
        write_milestones([{"class": "B", "pull": "p", "frame": 0, "dcd": dcd}], pdb, str(tmp_path / "ms"),
                         relax=False)


def test_relax_needs_system_and_atoms(tmp_path):
    pdb = _toy_top(tmp_path, 3)
    with pytest.raises(ValueError, match="needs system"):
        write_milestones([], pdb, str(tmp_path / "ms"), relax=True)


def test_relax_writes_state_and_system_with_finite_energy(tmp_path):
    import openmm as mm
    rng, pdb, X, dcd, _ = _setup(tmp_path, seed=16, deg=8.0)
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
