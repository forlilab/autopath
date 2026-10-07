import os

import MDAnalysis as mda
import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, PDBFile, Topology

from autopath.metadynamics import CVSpec, MetadynamicsMD
from autopath.metadynamics.report import walker_traces


def _toy(tmp_path):
    top = Topology()
    top.addAtom("C", Element.getBySymbol("C"), top.addResidue("UNK", top.addChain()))
    top.setPeriodicBoxVectors(np.eye(3) * 3.0 * u.nanometer)
    pdb = str(tmp_path / "one.pdb")
    with open(pdb, "w") as fh:
        PDBFile.writeFile(top, [[1.0, 1.5, 1.5]] * u.nanometer, fh)
    s = mm.System()
    s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.0))
    f = mm.CustomExternalForce("6*cos(2*3.14159*x/0.45) + 0.5*41840*(step(0.6-x)*(0.6-x)^2"
                               "+ step(x-1.5)*(x-1.5)^2) + 1000*((y-1.5)^2+(z-1.5)^2)")
    f.addParticle(0, [])
    s.addForce(f)
    return top, s, pdb


def _spec():
    cv = mm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return CVSpec(cv=cv, grid_min=0.5, grid_max=1.6, hill_width=0.03, grid_points=111, name="x")


@pytest.mark.parametrize("reweight", [True, False])
def test_rows_follow_rct_ustride_and_map_onto_dcd_frames(tmp_path, reweight):
    top, s, pdb = _toy(tmp_path)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[0],
                        platform="Reference", out_dir=str(tmp_path))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=[_spec()], mMD_time=0.2,
           bias_factor=10, hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"),
           reweight=reweight, rct_ustride=1, dcd_interval=10.0)
    colvar = np.load(tmp_path / "COLVAR_w.npy")[:, 0]
    frames = np.load(tmp_path / "FRAMES_w.npz")
    assert len(colvar) == 101                                  # every hill (2 ps) + t=0
    assert np.allclose(frames["time_ns"], np.arange(101) * 0.002)
    assert os.path.exists(tmp_path / "REWEIGHT_w.npz") == reweight
    if reweight:
        rw = np.load(tmp_path / "REWEIGHT_w.npz")
        assert len(rw["bias_kj"]) == len(rw["rct_kj"]) == 101
        assert rw["bias_kj"][0] == 0.0 and rw["rct_kj"][-1] > 0.0
    traj = mda.Universe(pdb, str(tmp_path / "WTMetaD_w.dcd"))
    assert len(traj.trajectory) == 20                          # one frame per 10 ps
    rows = np.flatnonzero(frames["dcd_frame"] >= 0)
    assert list(rows) == list(range(5, 101, 5))
    for k in rows:
        traj.trajectory[int(frames["dcd_frame"][k])]
        assert traj.atoms.positions[0, 0] / 10.0 == pytest.approx(colvar[k], abs=1e-4)


def test_report_reads_row_times_from_frames(tmp_path):
    np.save(tmp_path / "COLVAR_w.npy", np.linspace(0.0, 1.0, 11)[:, None])
    np.savez(tmp_path / "FRAMES_w.npz", time_ns=np.arange(11) * 0.002)
    (tr,) = walker_traces(str(tmp_path), ["w"], None, (0.0, 1.0))
    assert tr.t_ns[-1] == pytest.approx(0.02)
