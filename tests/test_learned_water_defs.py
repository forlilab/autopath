import logging

import numpy as np
import openmm as mm
import pytest
from openmm import app

from autopath.metadynamics.learned import water
from autopath.metadynamics.learned.features import evaluate
from autopath.metadynamics.learned.forces import fused_sw_forces, feature_cv

BOX = 5.0


def _write_pdb(path, residues, box=BOX):
    """residues: list of (resname, [(atom, element, xyz_nm)]); atoms written in order."""
    lines = [f"CRYST1{box * 10:9.3f}{box * 10:9.3f}{box * 10:9.3f}  90.00  90.00  90.00 P 1           1"]
    n = 0
    for r, (resname, atoms) in enumerate(residues, 1):
        for name, el, xyz in atoms:
            n += 1
            x, y, z = np.asarray(xyz) * 10
            lines.append(f"ATOM  {n:5d} {name:<4s} {resname:>3s} A{r % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                         f"  1.00  0.00          {el:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _toy(tmp_path, all_polar=False, lig_backbone=False, few_apolar=False):
    c = np.array([0.15, 0.3, 0.0])
    ser = ("SER", [("N", "N", (1.0, 1.0, 0.0)), ("CA", "C", (1.1, 1.1, 0.0)), ("C", "C", (1.2, 1.0, 0.0)),
                   ("O", "O", (1.3, 1.0, 0.0)), ("CB", "C", c + [0.05, 0.05, 0.1]),
                   ("OG", "O", c + [0.1, 0.1, 0.1])])
    met = ("MET", [("N", "N", (1.0, 1.4, 0.0)), ("CA", "C", (1.1, 1.4, 0.0)), ("C", "C", (1.2, 1.4, 0.0)),
                   ("O", "O", (1.3, 1.4, 0.0)), ("CB", "C", c), ("CG", "C", c + [0.15, 0.0, 0.0]),
                   ("SD", "S", c + [0.0, 0.15, 0.0]), ("CE", "C", c + [0.15, 0.15, 0.0])])
    val = ("VAL", [("N", "N", c + [0.1, -0.1, 0.0]), ("CA", "C", (1.1, 1.8, 0.0)), ("C", "C", (1.2, 1.8, 0.0)),
                   ("O", "O", (1.3, 1.8, 0.0)), ("CB", "C", c + [0.3, 0.0, 0.0]),
                   ("CG1", "C", c + [0.3, 0.15, 0.0]), ("CG2", "C", (2.5, 2.5, 2.5))])
    if all_polar:
        met = ("MET", [a for a in met[1] if a[0] not in ("CB", "CG", "SD", "CE")])
        val = ("VAL", [a for a in val[1] if a[0] not in ("CB", "CG1", "CG2")])
    if few_apolar:
        met = ("MET", [a for a in met[1] if a[0] not in ("CG", "SD", "CE")])
        val = ("VAL", [a for a in val[1] if a[0] != "CG1"])
    ln = ("N", "CA", "C") if lig_backbone else ("C1", "C2", "O1")
    lig = ("LIG", [(ln[0], "C", (0.0, 0.0, 0.0)), (ln[1], "C", (0.15, 0.0, 0.0)), (ln[2], "O", (0.3, 0.0, 0.0))])
    rng = np.random.default_rng(0)
    near = [np.array([0.15, 0.0, 0.0]) + rng.normal(0, 0.45, 3) for _ in range(17)]
    far = [np.array(p) for p in ((2.8, 2.8, 2.8), (3.2, 3.3, 3.0), (3.5, 2.9, 3.4))]
    wat = [("HOH", [("O", "O", p), ("H1", "H", p + [0.096, 0, 0]), ("H2", "H", p + [-0.03, 0.09, 0])])
           for p in near + far]
    pdb = _write_pdb(tmp_path / "toy.pdb", [ser, met, val, lig] + wat)
    if lig_backbone:
        first = 1 + len(ser[1]) + len(met[1]) + len(val[1])
        conect = "".join(f"CONECT{a:5d}{b:5d}\n" for a, b in ((first, first + 1), (first + 1, first + 2)))
        text = open(pdb).read()
        open(pdb, "w").write(text.replace("END\n", conect + "END\n"))
    top = app.PDBFile(pdb).topology
    idx = {(a.residue.index, a.name): a.index for a in top.atoms()}
    lig_idx = [idx[(3, n)] for n in ln]
    return pdb, lig_idx, idx


def _system(pdb):
    pdbf = app.PDBFile(pdb)
    s = mm.System()
    for _ in pdbf.topology.atoms():
        s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(*pdbf.topology.getPeriodicBoxVectors())
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0)
    for _ in pdbf.topology.atoms():
        nb.addParticle(0.0, 0.3, 0.0)
    nb.createExceptionsFromBonds([(a.index, b.index) for a, b in pdbf.topology.bonds()], 0.0, 0.0)
    s.addForce(nb)
    return s


def test_apolar_rule(tmp_path):
    pdb, lig, idx = _toy(tmp_path)
    got = set(water.apolar_pocket_atoms(pdb, lig))
    want = {idx[(1, n)] for n in ("CB", "CG", "SD", "CE")} | {idx[(2, "CB")], idx[(2, "CG1")]}
    assert got == want
    assert idx[(0, "CB")] not in got          # CB bonded to OG
    assert idx[(1, "C")] not in got           # backbone C bonded to O
    assert idx[(2, "CG2")] not in got         # outside the shell
    assert idx[(1, "SD")] in got              # sulphur


def test_apolar_non_standard_residue_without_bonds_raises(tmp_path):
    pdb, lig, _ = _toy(tmp_path)
    text = open(pdb).read().replace(" MET A", " XYZ A")
    p2 = tmp_path / "ns.pdb"
    p2.write_text(text)
    with pytest.raises(ValueError, match="XYZ"):
        water.apolar_pocket_atoms(str(p2), lig)


def test_patches_two_blobs_small_merge_and_order():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.05, (6, 3))
    b = rng.normal(0, 0.05, (5, 3)) + [3.0, 0, 0]
    X = np.vstack([a, b])
    got = water.patches(X, list(range(11)))
    assert got == [list(range(6)), list(range(6, 11))]
    # a 2-atom blob near blob a is merged into it; atom ids are global and order is by min index
    c = np.array([[0.0, 1.0, 0.0], [0.05, 1.0, 0.0]])
    X2 = np.vstack([c, a, b])
    got2 = water.patches(X2, list(range(13)))
    assert got2 == [list(range(8)), list(range(8, 13))]
    # deterministic under permutation of the atom list
    assert water.patches(X2, list(range(12, -1, -1))) == got2


def _density_pdb(tmp_path, n_core, water_pos, box):
    core = [("GLY", [("CA", "C", (box / 2 + 0.1 * (i - n_core / 2 + 0.5), box / 2 + 0.1 * (j - n_core / 2 + 0.5),
                                  box / 2 + 0.1 * (k - n_core / 2 + 0.5)))])
            for i in range(n_core) for j in range(n_core) for k in range(n_core)]
    wat = [("HOH", [("O", "O", p)]) for p in water_pos]
    return _write_pdb(tmp_path / "dens.pdb", core + wat, box=box)


def test_protein_weight_matches_known_densities(tmp_path):
    a, box, n = 0.32, 6.4, 20
    g = np.arange(int(round(box / a))) * a + a / 2
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    keep = np.linalg.norm(pos - box / 2, axis=1) > 1.3
    pdb = _density_pdb(tmp_path, n, pos[keep], box)
    h = np.arange(n) * 0.1                              # core lattice, counted independently
    P = np.array(np.meshgrid(h, h, h)).reshape(3, -1).T
    rg = np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1)))
    r = 0.5 * rg
    rho_p = np.sum(np.linalg.norm(P - P.mean(0), axis=1) <= r) / (4 / 3 * np.pi * r ** 3)
    expect = (1 / a ** 3) / rho_p
    assert water.protein_weight(pdb) == pytest.approx(expect, rel=0.02)


def test_protein_weight_raises_without_bulk_water(tmp_path):
    rng = np.random.default_rng(2)
    pdb = _density_pdb(tmp_path, 6, 3.0 + rng.normal(0, 0.3, (30, 3)), 6.0)
    with pytest.raises(ValueError, match="bulk"):
        water.protein_weight(pdb)


def test_define_term_structure(tmp_path):
    pdb, lig, idx = _toy(tmp_path)
    feats, info = water.define_water_features(pdb, lig, _system(pdb))
    top = app.PDBFile(pdb).topology
    wat_o = [a.index for a in top.atoms() if a.residue.name == "HOH" and a.name == "O"]
    prot = [a.index for a in top.atoms() if a.residue.index < 3 and a.element.symbol != "H"]
    w1 = [f for f in feats if f.family == "W1"]
    w3 = [f for f in feats if f.family == "W3"]
    assert [f.name for f in w1] == ["W1_C1", "W1_C2", "W1_O1"]
    assert w1[0].terms == [[[lig[0]], wat_o, 1.0]]
    assert len(w3) == len(info["patches"]) == 1 and w3[0].name == "W3_p0"
    assert info["n_apolar"] == 6 and info["w_p"] > 0
    res_of = {a.index: a.residue.index for a in top.atoms()}
    assert len(w3[0].terms) == 4                        # water + protein term for MET and VAL
    for a, b, w in w3[0].terms:
        r = {res_of[i] for i in a}
        assert len(r) == 1
        if set(b) & set(wat_o):
            assert b == wat_o and w == 1.0
        else:
            assert w == pytest.approx(info["w_p"])
            assert not r & {res_of[i] for i in b}      # own residue excluded
            assert set(b) == set(prot) - {i for i in prot if res_of[i] in r}
    assert sorted(info["patches"][0]) == sorted(["2:CB", "2:CG", "2:SD", "2:CE", "3:CB", "3:CG1"])
    assert w3[0].exclude, "cross-residue exception must be linked"


def test_define_needs_system_for_w3(tmp_path):
    pdb, lig, _ = _toy(tmp_path)
    with pytest.raises(ValueError, match="not supported"):
        water.define_water_features(pdb, lig, None)


def test_all_polar_pocket_has_no_w3(tmp_path, caplog):
    pdb, lig, _ = _toy(tmp_path, all_polar=True)
    with caplog.at_level(logging.WARNING):
        feats, info = water.define_water_features(pdb, lig, None)
    assert [f.family for f in feats] == ["W1"] * 3
    assert info["patches"] == [] and info["n_apolar"] == 0
    assert "apolar" in caplog.text


def test_features_build_fused_and_equal_openmm(tmp_path):
    pdb, lig, _ = _toy(tmp_path)
    system = _system(pdb)
    feats, _ = water.define_water_features(pdb, lig, system)
    w1, w3 = feats[0], feats[-1]
    pdbf = app.PDBFile(pdb)
    box = np.diag([BOX] * 3)
    X = np.asarray(pdbf.getPositions(asNumpy=True).value_in_unit(mm.unit.nanometer))
    X = X + np.random.default_rng(3).normal(0, 0.02, X.shape)
    ref = evaluate([w1, w3], X[None], box=box[None])[0]
    assert np.all(ref > 0)

    def energy(force):
        s = _system(pdb)
        force.setForceGroup(1)
        s.addForce(force)
        c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
        c.setPositions(X)
        return c.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value

    for f, r in ((w1, ref[0]), (w3, ref[1])):
        assert energy(feature_cv(f, system, "f")) == pytest.approx(r, rel=1e-10)
    forces, _ = fused_sw_forces([w1, w3], np.array([1.0, -2.0]), np.zeros(2), np.ones(2), system)
    assert energy(forces[0]) == pytest.approx(ref[0] - 2.0 * ref[1], rel=1e-10)


def test_ligand_with_backbone_names_is_not_protein(tmp_path):
    (tmp_path / "ref").mkdir()
    ref_pdb, ref_lig, _ = _toy(tmp_path / "ref")
    pdb, lig, _ = _toy(tmp_path, lig_backbone=True)
    feats, info = water.define_water_features(pdb, lig, _system(pdb))
    assert info["w_p"] == pytest.approx(water.protein_weight(ref_pdb, ref_lig), rel=1e-12)
    assert set(water.apolar_pocket_atoms(pdb, lig)).isdisjoint(lig)
    for f in feats:
        if f.family == "W3":
            for a, b, _ in f.terms:
                assert not set(a) & set(lig) and not set(b) & set(lig)


def test_bulk_water_distance_uses_hydrogens(tmp_path):
    box = 8.0
    core = [("GLY", [("CA", "C", (1.0 + 0.1 * i, 1.0, 1.0))]) for i in range(6)]
    core.append(("GLY", [("CA", "C", (1.0, 1.1, 1.0)), ("HA", "H", (6.5, 6.5, 6.5))]))
    wat = [("HOH", [("O", "O", (6.5, 6.5, 6.0))]), ("HOH", [("O", "O", (4.0, 4.0, 4.0))])]
    pdb = _write_pdb(tmp_path / "h.pdb", core + wat, box=box)
    # O at (4,4,4) is > 1 nm from every heavy atom; the first water is 0.5 nm from an H
    assert np.linalg.norm(np.array([4, 4, 4]) - np.array([6.5, 6.5, 6.5])) > 1.0
    w = water.protein_weight(pdb)
    assert w > 0
    pdb2 = _write_pdb(tmp_path / "h2.pdb", core + wat[:1], box=box)
    with pytest.raises(ValueError, match="bulk"):
        water.protein_weight(pdb2)


def test_only_small_patches_means_no_w3(tmp_path, caplog):
    pdb, lig, _ = _toy(tmp_path, few_apolar=True)
    assert len(water.apolar_pocket_atoms(pdb, lig)) == 2
    with caplog.at_level(logging.WARNING):
        feats, info = water.define_water_features(pdb, lig, None)
    assert {f.family for f in feats} == {"W1"} and info["patches"] == [] and info["n_apolar"] == 2
    assert "W3" in caplog.text


def test_system_size_mismatch_raises(tmp_path):
    pdb, lig, _ = _toy(tmp_path)
    s = _system(pdb)
    s.addParticle(1.0)
    with pytest.raises(ValueError, match="not supported"):
        water.define_water_features(pdb, lig, s)


def test_exclude_equals_independent_exception_enumeration(tmp_path):
    pdb, lig, _ = _toy(tmp_path)
    s = _system(pdb)
    feats, _ = water.define_water_features(pdb, lig, s)
    nb = s.getForce(0)
    exc = [tuple(nb.getExceptionParameters(e)[:2]) for e in range(nb.getNumExceptions())]
    n_nonempty = 0
    for f in feats:
        want = set()
        for a, b, _ in f.terms:
            for i, j in exc:
                if (i in a and j in b) or (j in a and i in b):
                    want.add((min(i, j), max(i, j)))
        assert {tuple(p) for p in (f.exclude or [])} == want
        n_nonempty += bool(want)
    assert n_nonempty >= 1
