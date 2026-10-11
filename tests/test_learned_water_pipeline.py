"""feature_set="water" in learn_cv: safe-radius frames, W1/W3 extraction and the composition report."""
import json
import os

import numpy as np
import openmm as mm
import pandas as pd
import pytest
from openmm import app, unit

from autopath.metadynamics.learned.features import CN_DMAX, Feature, cn_switch, evaluate

O_OFF = -np.array([0.022, 0.03, 0.0])                       # O relative to the water centre of geometry
H_OFF = (np.array([0.096, 0.0, 0.0]), np.array([-0.03, 0.09, 0.0]))


def _water_xyz(cog):
    o = cog + O_OFF
    return np.stack([o, o + H_OFF[0], o + H_OFF[1]], axis=1).reshape(-1, 3)


def _frame(box, atoms, rng, n_w=3000, spill=0.0):
    """Atoms (nm, relative to the centre) then waters whose centres fill the brick of box around the centre.

    Atoms 0, 1 are the imaging centre (symmetric about 0); spill > 0 pushes one water centre that far out of
    the brick along +x.
    """
    h = np.diag(box) / 2
    cog = rng.uniform(-h, h, (n_w, 3))
    cog[0] = [h[0] + spill, 0.0, 0.0] if spill else cog[0]
    X = np.vstack([[[0.3, 0, 0], [-0.3, 0, 0]], np.asarray(atoms, float), _water_xyz(cog)])
    mols = np.arange(2 + len(atoms), len(X)).reshape(-1, 3)
    return X, mols


def _rd(a):
    return np.array([[a, 0, 0], [0, a, 0], [a / 2, a / 2, a / np.sqrt(2)]])


def test_safe_frames_uses_the_faces_of_the_wrapped_brick():
    from autopath.metadynamics.learned.imaging import safe_frames
    rng = np.random.default_rng(0)
    L = 2 * (CN_DMAX + 1.0)                                       # inscribed sphere radius CN_DMAX + 1
    cube = np.eye(3) * L
    atoms = [[1.5, 0.0, 0.0],                                     # inside the inscribed sphere, 0.5 from a face
             [0.9, 0.9, 0.0],                                     # outside the safe sphere, 1.1 from every face
             [0.0, 0.0, 0.99]]                                    # 1.01 from the z faces
    for k, want in enumerate([False, True, True]):
        X, mols = _frame(cube, [atoms[k]], rng)
        valid, margin = safe_frames(X[None], cube[None], [0, 1], [2], mols)
        assert valid[0] == want
        face = L / 2 - np.abs(atoms[k]).max()
        assert margin[0] == pytest.approx(face - CN_DMAX, abs=1e-6)
    # rhombic dodecahedron: one cell is the brick a x a x a/sqrt(2), so x reaches beyond the inscribed sphere
    a = 2 * np.sqrt(2) * (CN_DMAX + 0.5)                          # c_z / 2 = CN_DMAX + 0.5
    box = _rd(a)
    X, mols = _frame(box, [[a / 2 - CN_DMAX - 0.05, 0.0, 0.0], [0.0, 0.0, 0.4]], rng)
    valid, margin = safe_frames(X[None], box[None], [0, 1], [2, 3], mols)
    assert valid[0] and margin[0] == pytest.approx(0.05, abs=1e-6)


def test_safe_frames_margin_shrinks_by_the_water_excess_and_unwrapped_cells_are_not_supported():
    from autopath.metadynamics.learned.imaging import IMAGED_TOL_NM, safe_frames
    rng = np.random.default_rng(1)
    cube = np.eye(3) * 2 * (CN_DMAX + 1.0)
    X, mols = _frame(cube, [[0.0, 0.0, 0.95]], rng, spill=0.1)    # a cell rotated slightly in the aligned frame
    valid, margin = safe_frames(X[None], cube[None], [0, 1], [2], mols)
    assert margin[0] == pytest.approx((CN_DMAX + 1.0 - 0.95) - CN_DMAX - 0.1, abs=1e-6) and not valid[0]
    X, mols = _frame(cube, [[0.0, 0.0, 0.0]], rng, spill=IMAGED_TOL_NM + 0.1)
    with pytest.raises(ValueError, match="not imaged into one cell.*not supported"):
        safe_frames(X[None], cube[None], [0, 1], [2], mols)
    with pytest.raises(ValueError, match="not supported"):
        safe_frames(X[None], None, [0, 1], [2], mols)
    with pytest.raises(ValueError, match="not supported"):
        safe_frames(X[None], cube[None], [0, 1], [2], np.zeros((0, 3), int))


def test_parallelepiped_wrapped_rhombic_dodecahedron_is_not_supported():
    from autopath.metadynamics.learned.imaging import safe_frames
    rng = np.random.default_rng(3)
    box = _rd(6.7)
    frac = rng.uniform(0, 1, (4000, 3))                           # waters wrapped by fractional floor
    cog = frac @ box
    centre = box.sum(0) / 2
    X = np.vstack([[centre + [0.3, 0, 0], centre - [0.3, 0, 0]], [centre + [0.0, 0.0, 0.5]], _water_xyz(cog)])
    mols = np.arange(3, len(X)).reshape(-1, 3)
    with pytest.raises(ValueError, match="not imaged into one cell.*not supported"):
        safe_frames(X[None], box[None], [0, 1], [2], mols)


def test_open_box_gives_plain_distances():
    from autopath.metadynamics.learned.imaging import open_box
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 4.0, (1, 400, 3))
    f = Feature("w", "cn", "W1", terms=[[[0], list(range(1, 400)), 1.0]])
    d = np.linalg.norm(X[0, 1:] - X[0, 0], axis=1)
    assert evaluate([f], X, box=open_box(X))[0, 0] == pytest.approx(cn_switch(d).sum(), rel=1e-12)


# ---- learn_cv on a toy complex -------------------------------------------------------------------------

L = 5.0                          # cn centre atoms must stay CN_DMAX from the faces of the cell around the protein
N_FR, N_PULL = 80, 6
REST = 0.9                       # waters are kept out of a sphere around the pocket (shrinks during the pull)
POCKET = np.array([0.15, 0.3, 0.0])


def _protein_and_ligand():
    c = POCKET
    ser = ("SER", [("N", "N", (-0.4, -0.2, 0.0)), ("CA", "C", (-0.3, -0.1, 0.0)), ("C", "C", (-0.2, -0.2, 0.0)),
                   ("O", "O", (-0.1, -0.2, 0.0)), ("CB", "C", c + [0.05, 0.05, 0.1]),
                   ("OG", "O", c + [0.1, 0.1, 0.1])])
    met = ("MET", [("N", "N", (-0.4, 0.4, 0.0)), ("CA", "C", (-0.3, 0.4, 0.0)), ("C", "C", (-0.2, 0.4, 0.0)),
                   ("O", "O", (-0.1, 0.5, 0.0)), ("CB", "C", c), ("CG", "C", c + [0.15, 0.0, 0.0]),
                   ("SD", "S", c + [0.0, 0.15, 0.0]), ("CE", "C", c + [0.15, 0.15, 0.0])])
    val = ("VAL", [("N", "N", c + [0.1, -0.1, 0.0]), ("CA", "C", (-0.3, 0.8, 0.0)), ("C", "C", (-0.2, 0.8, 0.0)),
                   ("O", "O", (-0.1, 0.8, 0.0)), ("CB", "C", c + [0.3, 0.0, 0.0]),
                   ("CG1", "C", c + [0.3, 0.15, 0.0]), ("CG2", "C", (-0.5, -0.5, -0.5))])
    lig = ("LIG", [("C1", "C", (0.0, 0.0, 0.0)), ("C2", "C", (0.15, 0.0, 0.0)), ("O1", "O", (0.3, 0.0, 0.0))])
    return [ser, met, val, lig]


def _write_pdb(path, residues, box=L):
    lines = [f"CRYST1{box * 10:9.3f}{box * 10:9.3f}{box * 10:9.3f}  90.00  90.00  90.00 P 1           1"]
    n = 0
    for r, (resname, atoms) in enumerate(residues, 1):
        for name, el, xyz in atoms:
            n += 1
            x, y, z = np.asarray(xyz) * 10
            lines.append(f"ATOM  {n % 100000:5d} {name:<4s} {resname:>3s} A{r % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                         f"  1.00  0.00          {el:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _system(pdb):
    pdbf = app.PDBFile(pdb)
    s = mm.System()
    for a in pdbf.topology.atoms():
        s.addParticle(a.element.mass)
    s.setDefaultPeriodicBoxVectors(*pdbf.topology.getPeriodicBoxVectors())
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0)
    for _ in pdbf.topology.atoms():
        nb.addParticle(0.0, 0.3, 0.0)
    nb.createExceptionsFromBonds([(a.index, b.index) for a, b in pdbf.topology.bonds()], 0.0, 0.0)
    s.addForce(nb)
    return s


def _waters(rng, n_w, centre, rest):
    """Water centres uniform in the cell around centre, outside the shrinking pocket sphere."""
    out = []
    while len(out) < n_w:
        cog = rng.uniform(-L / 2, L / 2, (n_w, 3)) + centre
        out.extend(cog[np.linalg.norm(cog - POCKET, axis=1) > rest])
    return np.array(out[:n_w])


def _force(t):
    return np.select([t < 30.0, t < 50.0, t < 64.0], [0.0, 5.0 * (t - 30.0), 100.0 - 6.5 * (t - 50.0)], 9.0)


def _ligand_shift(t):
    return 1.6 * np.clip((t - 30.0) / 70.0, 0.0, 1.0)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    import MDAnalysis as mda
    d = tmp_path_factory.mktemp("water_toy")
    traj = d / "trajectories"
    traj.mkdir()
    rng = np.random.default_rng(7)
    solute = _protein_and_ligand()
    n_sol = sum(len(r[1]) for r in solute)
    P0 = np.array([np.asarray(x, float) for _, at in solute for _, _, x in at])
    lig = list(range(n_sol - 3, n_sol))
    prot = list(range(n_sol - 3))
    centre = P0[prot].mean(0)
    n_w = 5000
    res = [(n, [(a, e, np.asarray(x, float)) for a, e, x in at]) for n, at in solute]
    res += [("HOH", [(nm, el, x) for (nm, el), x in zip((("O", "O"), ("H1", "H"), ("H2", "H")), _water_xyz(w[None]))])
            for w in _waters(rng, n_w, centre, REST)]
    pdb = _write_pdb(d / "system.pdb", res)
    system = _system(pdb)
    xml = d / "system.xml"
    xml.write_text(mm.XmlSerializer.serialize(system))
    u = mda.Universe(pdb)
    coords = {}
    for p in range(N_PULL):
        name = f"sMD_replica-{p + 1}_v{(0.01, 0.02)[p % 2]}_forward"
        t_dat = np.linspace(0.0, 100.0, 201)
        r_dat = 0.8 + _ligand_shift(t_dat)
        pd.DataFrame({"time": t_dat, "r_target": r_dat + 0.1, "r_after": r_dat, "force": _force(t_dat),
                      "dW_protocol": np.full(201, 0.5)}).to_csv(traj / f"{name}.dat", index=False)
        frames = []
        with mda.Writer(str(traj / f"{name}_aligned.dcd"), u.atoms.n_atoms) as W:
            for k in range(N_FR):
                t = 100.0 * k / (N_FR - 1)
                X = P0.copy()
                X[lig] += [_ligand_shift(t), 0.0, 0.0] + rng.normal(0.0, 0.02, 3)
                rest = REST - 0.4 * min(1.0, max(0.0, (t - 30.0) / 34.0))
                X = np.vstack([X, _water_xyz(_waters(rng, n_w, centre, rest))])
                u.atoms.positions = X * 10.0
                u.dimensions = [L * 10.0] * 3 + [90.0] * 3
                W.write(u.atoms)
                frames.append(u.atoms.positions / 10.0)
        coords[f"{name}_aligned.dcd"] = np.array(frames, dtype=float)
    fp = {"host_index": [1, 7], "guest_index": lig, "unbinding_axis": [1.0, 0.0, 0.0]}
    return dict(dir=d, traj=str(traj), pdb=pdb, xml=str(xml), system=system, lig=lig, prot=prot,
                coords=coords, funnel=fp, centre=centre)


def test_toy_imaging_centre_is_the_protein(toy):
    from autopath.metadynamics.learned.imaging import anchor_atoms
    assert anchor_atoms(toy["pdb"]) == toy["prot"]


def _learn(toy, out, **kw):
    from autopath.metadynamics.learned.pipeline import learn_cv
    (toy["dir"] / "l.sdf").write_text("sdf")
    kw = dict(dict(system=toy["xml"]), **kw)
    return learn_cv(toy["pdb"], str(toy["dir"] / "l.sdf"), toy["lig"], [1, 7], toy["traj"], str(out),
                    hidden=False, classes="rupture", funnel=toy["funnel"], mosaic_resolution=0.5, n_perm=10, **kw)


@pytest.fixture(scope="module")
def learned_water(toy, tmp_path_factory):
    out = tmp_path_factory.mktemp("water_out")
    return _learn(toy, out), str(out)


def test_learn_cv_water_writes_only_water_features(learned_water, toy):
    path, out = learned_water
    d = np.load(os.path.join(out, "features.npz"), allow_pickle=False)
    assert str(d["feature_set"]) == "water"
    fams = {str(n).split("_")[0] for n in d["names"]}
    assert fams == {"W1", "W3"}
    assert float(d["w_p"]) > 0 and json.loads(str(d["patches_json"]))
    rep = json.load(open(os.path.join(out, "rupture_report.json")))
    assert set(rep["family_composition"]) <= {"W1", "W3"}
    assert sum(rep["family_composition"].values()) == len(rep["kept"])
    c = rep["corr_W1_W3"]
    names = list(d["names"])
    m = (d["labels"] != "") & d["valid"]
    for i, a in enumerate(c["W1"]):
        for j, b in enumerate(c["W3"]):
            x, y = d["V"][m, names.index(a)], d["V"][m, names.index(b)]
            assert c["r"][i][j] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-9)
    assert str(np.load(path)["feature_set"]) == "water"


def test_features_npz_locates_every_row(learned_water, toy):
    _, out = learned_water
    d = np.load(os.path.join(out, "features.npz"), allow_pickle=False)
    for dcd in toy["coords"]:
        rows = d["dcd"] == dcd
        assert d["frame"][rows].tolist() == list(range(0, N_FR, 2))
    assert len(d["frame"]) == len(d["V"]) == N_PULL * len(range(0, N_FR, 2))


def test_frames_beyond_the_safe_radius_are_excluded_and_counted(learned_water, toy):
    from autopath.metadynamics.learned.features import features_from_dict
    from autopath.metadynamics.learned.imaging import open_box
    _, out = learned_water
    d = np.load(os.path.join(out, "features.npz"), allow_pickle=False)
    feats = features_from_dict(d)
    valid = d["valid"]
    lab = d["labels"]
    assert valid.any() and not valid.all()
    seen = 0
    for dcd, X in toy["coords"].items():
        rows = np.flatnonzero(d["dcd"] == dcd)
        Xr = X[d["frame"][rows]]
        far = np.abs(Xr[:, toy["lig"]] - toy["centre"]).max(axis=(1, 2)) > L / 2 - CN_DMAX
        assert valid[rows].tolist() == (~far).tolist()
        assert np.isnan(d["V"][rows[far]]).all()
        k = rows[~far]
        np.testing.assert_allclose(d["V"][k], evaluate(feats, Xr[~far], box=open_box(Xr[~far])), rtol=1e-6, atol=1e-6)
        seen += far.sum()
    assert seen == (~valid).sum()
    rep = json.load(open(os.path.join(out, "rupture_report.json")))
    sr = rep["safe_radius"]
    assert sr["n_excluded"] == int((~valid).sum()) and sr["n_frames"] == len(valid)
    assert sr["excluded_per_class"] == {c: int(np.sum(~valid & (lab == c))) for c in ("B", "TS", "U")}
    assert sr["excluded_per_class"]["U"] > 0 and sr["excluded_per_class"]["B"] == 0
    assert sum(sr["excluded_per_pull"].values()) == sr["n_excluded"]
    assert rep["class_counts"]["U"] == int(np.sum(valid & (lab == "U")))


def test_rupture_report_has_water_and_fused_cn_stats(learned_water):
    _, out = learned_water
    rep = json.load(open(os.path.join(out, "rupture_report.json")))
    assert rep["water"]["n_apolar"] >= 3 and rep["extraction"]["s_per_frame"] > 0
    fc = rep["fused_cn"]
    assert fc["n_classes"] >= 2 and fc["n_groups"] >= 1 and fc["build_s"] >= 0


def test_water_needs_the_system(toy, tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        _learn(toy, tmp_path / "o", system=None, masses=[12.0] * toy["system"].getNumParticles())


def test_water_takes_masses_from_a_system_object(toy, tmp_path):
    path = _learn(toy, tmp_path / "o", system=toy["system"])
    assert str(np.load(path)["feature_set"]) == "water"


def test_water_extraction_without_waters_is_not_supported(toy, tmp_path, monkeypatch):
    from autopath.metadynamics.learned import pipeline
    pdb = _write_pdb(tmp_path / "dry.pdb", _protein_and_ligand())
    monkeypatch.setattr(pipeline, "define_water_features", lambda *a, **k: ([], {}))
    monkeypatch.setattr(pipeline, "_frames_table", lambda *a, **k: [pd.DataFrame({"r": [0.0]})])
    pulls = pipeline.find_pulls(toy["traj"], "forward")
    with pytest.raises(ValueError, match="not supported"):
        pipeline._extract(pdb, None, toy["lig"], [1, 7], pulls, 1, classes="rupture", feature_set="water",
                          funnel=toy["funnel"], masses=None, system=toy["system"])


def test_contacts_feature_set_keeps_the_round2_families(toy, tmp_path, monkeypatch):
    from autopath.metadynamics.learned import pipeline
    a, b = toy["lig"][:2]
    feats = [Feature("F1_p", "pair", "F1", pairs=[(16, a)], r0=0.35),
             Feature("F2_c", "contacts", "F2", group_a=list(toy["lig"]), group_b=[4, 10, 11, 12], r0=0.3),
             Feature("F3_x", "pair", "F3", pairs=[(16, b)], r0=0.35),
             Feature("F5_x", "pair", "F5", pairs=[(17, b)], r0=0.35),
             Feature("F7_d", "centroid_distance", "F7", groups=[toy["lig"], [10, 11, 12]])]
    monkeypatch.setattr(pipeline, "define_features", lambda *a, **k: feats)
    real_select = pipeline.select

    def keep_all(V, names, *a, **k):                                 # the toy features are redundant
        return dict(real_select(V, names, *a, **k), kept=list(range(len(names))))
    monkeypatch.setattr(pipeline, "select", keep_all)
    path = _learn(toy, tmp_path / "o", feature_set="contacts", system=None,
                  masses=[toy["system"].getParticleMass(i).value_in_unit(unit.dalton)
                          for i in range(toy["system"].getNumParticles())])
    d = np.load(os.path.join(tmp_path / "o", "features.npz"), allow_pickle=False)
    assert str(d["feature_set"]) == "contacts"
    assert [str(n) for n in d["names"]] == ["F1_p", "F2_c", "F7_d"]
    rep = json.load(open(os.path.join(tmp_path / "o", "rupture_report.json")))
    assert set(rep["family_composition"]) <= {"F1", "F2", "F7"} and rep["corr_W1_W3"]["r"] == []
    assert "fused_cn" not in rep and "safe_radius" not in rep


def test_cache_key_covers_feature_set_and_system(toy):
    from autopath.metadynamics.learned.pipeline import _cache_key
    from autopath.metadynamics.learned.select import find_pulls
    pulls = find_pulls(toy["traj"])
    args = (pulls, 2, "forward", toy["pdb"], toy["pdb"], toy["lig"], [1])
    k = _cache_key(*args, code_hash="a", classes="rupture", feature_set="water", system_sha="s1")
    assert k != _cache_key(*args, code_hash="a", classes="rupture", feature_set="contacts", system_sha="s1")
    assert k != _cache_key(*args, code_hash="a", classes="rupture", feature_set="water", system_sha="s2")
