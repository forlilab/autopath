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
