"""write_feature_pse: objects, groups, scenes, colours, waters and pocket alignment of the saved session."""
import json

import numpy as np
import pytest

pymol2 = pytest.importorskip("pymol2")

from autopath.metadynamics.learned.features import Feature, features_to_dict  # noqa: E402
from autopath.metadynamics.learned.visualize import write_feature_pse  # noqa: E402

N_PROT = 24
LIG = [24, 25]
WATERS = [26, 29, 32, 35, 38]                       # O atoms; first two are near the ligand


def _coords():
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    prot = []
    for a in ang:
        c = 0.5 * np.array([np.cos(a), np.sin(a), 0.0])
        prot += [c + [0.1, 0, 0.1], c, c + [0, 0.1, 0.1], c + [0, 0, -0.1]]
    lig = [[0, 0, 0], [0.15, 0, 0]]
    wat = []
    for o in ([0.0, 0.0, 0.4], [0.0, 0.0, -0.4], [2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]):
        o = np.array(o)
        wat += [o, o + [0.1, 0, 0], o + [0, 0.1, 0]]
    return np.array(prot + lig + wat)


def _write_pdb(path, X, shift=None, resn="UNK", cryst=None):
    X = X * 10.0 if shift is None else X * 10.0 @ shift[0].T + shift[1]
    lines, serial = [], 1
    if cryst:
        lines.append(f"CRYST1{cryst:9.3f}{cryst:9.3f}{cryst:9.3f}  90.00  90.00  90.00 P 1           1")
    for r in range(6):
        for k, nm in enumerate(("N", "CA", "C", "O")):
            x = X[4 * r + k]
            lines.append(f"ATOM  {serial:5d} {nm:<4s} ALA A{r + 1:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           {nm[0]:>2s}")
            serial += 1
    for k, nm in enumerate(("C1", "C2")):
        x = X[24 + k]
        lines.append(f"HETATM{serial:5d} {nm:<4s} {resn:<3s} L{7:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           C")
        serial += 1
    for w in range(5):
        for k, nm in enumerate(("O", "H1", "H2")):
            x = X[26 + 3 * w + k]
            lines.append(f"HETATM{serial:5d} {nm:<4s} HOH W{8 + w:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           {nm[0]:>2s}")
            serial += 1
    open(path, "w").write("\n".join(lines) + "\nEND\n")


@pytest.fixture
def toy(tmp_path):
    X = _coords()
    _write_pdb(tmp_path / "bound.pdb", X)
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    _write_pdb(tmp_path / "m2.pdb", X, shift=(Rz, np.array([5.0, -3.0, 2.0])))
    wat = WATERS
    feats = [
        Feature("W1_a", "cn", "W1", terms=[[[24], wat, 1.0]]),
        Feature("W1_b", "cn", "W1", terms=[[[25], wat, 1.0]]),
        Feature("W3_p", "cn", "W3", terms=[[[1, 5], wat, 1.0]]),
        Feature("F1_p", "pair", "F1", pairs=[(1, 24)], r0=0.35),
        Feature("W1_c", "cn", "W1", terms=[[[24, 25], wat, 1.0]]),
        Feature("F1_q", "pair", "F1", pairs=[(5, 25)], r0=0.35),
    ]
    npz = str(tmp_path / "learned_cv.npz")
    np.savez(npz, kept=np.arange(4), names=np.array([f.name for f in feats]),
             sw_weights=np.array([0.8, -0.4, 0.3, 0.5]), w_p=1.0,
             patches_json=np.array(json.dumps([["1:CA"]])), **features_to_dict(feats))
    return npz, {"bound": str(tmp_path / "bound.pdb"), "m2": str(tmp_path / "m2.pdb")}, tmp_path


def _open(pse):
    pm = pymol2.PyMOL()
    pm.start()
    pm.cmd.load(pse)
    return pm


def _color(cmd, obj):
    cols = []
    cmd.iterate(obj, "cols.append(color)", space={"cols": cols})
    return np.array(cmd.get_color_tuple(cols[0]))


def test_session_objects_groups_scenes_and_colours(toy):
    npz, structs, tmp = toy
    out = write_feature_pse(npz, structs, str(tmp / "features.pse"))
    assert out == str(tmp / "features.pse")
    pm = _open(out)
    cmd = pm.cmd
    try:
        objs = cmd.get_names("objects")
        for o in ("state_bound", "state_m2", "waters_bound", "waters_m2"):
            assert o in objs
        groups = cmd.get_names("group_objects")
        for g in ("W1", "W3", "F1", "W1_dropped", "F1_dropped", "shells"):
            assert g in groups
        assert "W3_dropped" not in groups
        assert {"scene_bound", "scene_m2"} <= set(cmd.get_scene_list())
        assert "W1_c_dropped" in cmd.get_object_list("W1_dropped")
        assert "F1_q_dropped" in cmd.get_names("all")
        red, blue = _color(cmd, "W1_a"), _color(cmd, "W1_b")
        assert red[0] > 0.7 and red[2] < 0.3 and blue[2] > 0.7 and blue[0] < 0.3
        assert cmd.count_atoms("W1_a") == 1 and cmd.count_atoms("W3_p") == 2
        # larger |weight| gives a larger sphere
        s = [cmd.get_setting_float("sphere_scale", o) for o in ("W1_a", "W1_b")]
        assert s[0] > s[1]
        # only waters within the cutoff of a centre: the two near waters, one oxygen each
        assert cmd.count_atoms("waters_bound") == 2
        on = cmd.get_names("objects", enabled_only=1)
        assert "shell_cgo" not in on and "W1_c_dropped" not in on and "W1_a" in on
    finally:
        pm.stop()


def test_milestone_aligned_on_pocket_calpha(toy):
    npz, structs, tmp = toy
    pm = _open(write_feature_pse(npz, structs, str(tmp / "f.pse")))
    try:
        a = np.array(pm.cmd.get_coords("state_bound and name CA"))
        b = np.array(pm.cmd.get_coords("state_m2 and name CA"))
        assert np.abs(a - b).max() < 1e-2
    finally:
        pm.stop()


def test_waters_off_and_cutoff(toy):
    npz, structs, tmp = toy
    pm = _open(write_feature_pse(npz, structs, str(tmp / "off.pse"), waters=False))
    try:
        assert not any(o.startswith("waters_") for o in pm.cmd.get_names("objects"))
    finally:
        pm.stop()
    pm = _open(write_feature_pse(npz, structs, str(tmp / "wide.pse"), water_cut_nm=5.0))
    try:
        assert pm.cmd.count_atoms("waters_bound") == 5
    finally:
        pm.stop()


def test_funnel_cgo_is_added(toy):
    from openmm import unit
    npz, structs, tmp = toy
    funnel = {"z_cc": 2.0 * unit.nanometer, "R_cylinder": 0.1 * unit.nanometer, "alpha": 30 * unit.degree,
              "z_max": 3.0 * unit.nanometer, "unbinding_axis": [0.0, 0.0, 1.0], "z_low": -0.6 * unit.nanometer,
              "host_index": list(range(N_PROT))}
    pm = _open(write_feature_pse(npz, structs, str(tmp / "fun.pse"), funnel=funnel))
    try:
        assert "funnel" in pm.cmd.get_names("objects")
    finally:
        pm.stop()


def test_atom_count_mismatch_raises(toy):
    npz, structs, tmp = toy
    lines = open(structs["m2"]).read().splitlines()
    open(tmp / "short.pdb", "w").write("\n".join(lines[:-4]) + "\nEND\n")
    with pytest.raises(ValueError, match="atom count"):
        write_feature_pse(npz, {"bound": structs["bound"], "bad": str(tmp / "short.pdb")}, str(tmp / "x.pse"))


def _across_boundary(tmp):
    X = _coords()
    X[WATERS[2]] = [4.9, 0.0, 0.0]                  # 0.1 nm from the ligand through the periodic image
    return X


def test_water_selection_uses_the_cryst1_minimum_image(toy):
    npz, structs, tmp = toy
    X = _across_boundary(tmp)
    _write_pdb(tmp / "box.pdb", X, cryst=50.0)
    _write_pdb(tmp / "nobox.pdb", X)
    for name, want in (("box", 3), ("nobox", 2)):
        pm = _open(write_feature_pse(npz, {"bound": str(tmp / f"{name}.pdb")}, str(tmp / f"{name}.pse")))
        try:
            assert pm.cmd.count_atoms("waters_bound") == want
        finally:
            pm.stop()


def test_explicit_ligand_idx_needs_no_unk_residue(toy):
    npz, structs, tmp = toy
    X = _coords()
    _write_pdb(tmp / "lig.pdb", X, resn="LIG")
    _write_pdb(tmp / "lig2.pdb", X, resn="LIG")
    npz2 = str(tmp / "nowater.npz")
    d = dict(np.load(npz))
    d["features_json"] = np.array([x for x, nm in zip(d["features_json"], d["names"]) if nm.startswith("F1")])
    d.update(kept=np.array([0]), sw_weights=np.array([0.5]), names=np.array(["F1_p"]))
    np.savez(npz2, **d)
    s = {"bound": str(tmp / "lig.pdb"), "m2": str(tmp / "lig2.pdb")}
    with pytest.raises(ValueError, match="not supported"):
        write_feature_pse(npz2, s, str(tmp / "a.pse"))
    pm = _open(write_feature_pse(npz2, s, str(tmp / "b.pse"), ligand_idx=LIG))
    try:
        assert "state_bound" in pm.cmd.get_names("objects")
    finally:
        pm.stop()
