import numpy as np
from openmm import unit

from autopath.metadynamics import MetadynamicsMD
from autopath.metadynamics.Funnel import save_funnel_params
from autopath.metadynamics.summary import load_run

from ._metad_toy import toy, x_spec

A = unit.angstrom


def _funnel(path):
    p = {"host_index": [0], "guest_index": [1],
         "k_xy": 10.0 * unit.kilocalorie_per_mole / A**2,
         "z_cc": 14.0 * A, "z_max": 26.0 * A, "alpha": 45.0 * unit.degrees,
         "R_cylinder": 4.0 * A, "unbinding_axis": np.array([0.0, 0.0, 1.0]),
         "com_trajectory": np.zeros((2, 3)), "host_com_trajectory": np.zeros((2, 3)),
         "guest_com_trajectory": np.zeros((2, 3)), "radial_distances": np.zeros(2),
         "axial_distances": np.array([4.0, 4.0]), "trajectory_length": 2, "force_group": 10}
    save_funnel_params(p, str(path))
    return str(path) + ".npz"


def _run(tmp_path, **kw):
    top, s, pdb = toy(tmp_path)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[0],
                        platform="Reference", out_dir=str(tmp_path))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=[x_spec()], mMD_time=0.04,
           bias_factor=20, hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), **kw)


def test_run_meta_is_recorded_and_found_by_load_run(tmp_path):
    fp = _funnel(tmp_path / "fp")
    _run(tmp_path, run_meta={"funnel_file": fp, "learned_cv_file": "", "x": None})
    h = np.load(tmp_path / "HILLS_w.npz")
    assert str(h["funnel_file"]) == fp
    assert "learned_cv_file" not in h.files and "x" not in h.files
    assert load_run(tmp_path).funnel is not None


def test_run_without_meta_has_no_keys(tmp_path):
    _run(tmp_path)
    h = np.load(tmp_path / "HILLS_w.npz")
    assert "funnel_file" not in h.files
    assert load_run(tmp_path).funnel is None


def test_reserved_run_meta_key_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="HILLS record keys: not supported"):
        _run(tmp_path, run_meta={"run_id": "x"})
    assert not (tmp_path / "HILLS_w.npz").exists()


def test_hills_keys_cover_the_record(tmp_path):
    from autopath.metadynamics.Hills import HILLS_KEYS
    _run(tmp_path, run_meta={"funnel_file": "f.npz"})
    with np.load(tmp_path / "HILLS_w.npz") as h:
        assert set(h.files) - {"funnel_file"} <= HILLS_KEYS


def test_pipeline_run_meta(tmp_path):
    from autopath.autopath_core import metad_run_meta
    assert metad_run_meta(str(tmp_path), str(tmp_path / "l")) == {}
    (tmp_path / "funnel_params.npz").write_bytes(b"x")
    (tmp_path / "l").mkdir()
    (tmp_path / "l" / "learned_cv.npz").write_bytes(b"x")
    m = metad_run_meta(str(tmp_path), str(tmp_path / "l"))
    assert m == {"funnel_file": str(tmp_path / "funnel_params.npz"),
                 "learned_cv_file": str(tmp_path / "l" / "learned_cv.npz")}
    assert metad_run_meta(None, None) == {}


def test_control_script_run_meta_contacts_learned(tmp_path, monkeypatch):
    import argparse
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "examples" / "04_Full-Pipeline" / "run_single_walker_control.py"
    spec = importlib.util.spec_from_file_location("_control_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    a = argparse.Namespace(cv="contacts_learned", funnel_from="fp.npz", learned_cv_file=None, cv_def="def.json")
    m = mod.run_meta_for(a, {"contacts": {}, "learned_cv_file": "sw/learned_cv.npz"})
    assert m == {"funnel_file": str(tmp_path / "fp.npz"), "learned_cv_file": str(tmp_path / "sw/learned_cv.npz"),
                 "cv_def_file": str(tmp_path / "def.json")}
    a = argparse.Namespace(cv="learned_z", funnel_from="fp.npz", learned_cv_file="l.npz", cv_def=None)
    assert mod.run_meta_for(a)["learned_cv_file"] == str(tmp_path / "l.npz")
