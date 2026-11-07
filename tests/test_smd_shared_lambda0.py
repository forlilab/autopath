"""SteeredMD.run equilibrates every replica with the spring at one shared lambda0."""
import numpy as np
import pandas as pd
import pytest
import openmm
import openmm.app as app
import openmm.unit as u

from autopath.pulling.SMDData import SMDData
from autopath.pulling.steered_md import SteeredMD

BOX_NM = 5.0
R_BOUND = 1.0


def _two_particle_system(tmp_path):
    """Two carbon particles held near R_BOUND by a weak bond, in a periodic box."""
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, R_BOUND, 500.0)
    bond.setUsesPeriodicBoundaryConditions(True)
    system.addForce(bond)
    box = [openmm.Vec3(BOX_NM, 0, 0), openmm.Vec3(0, BOX_NM, 0), openmm.Vec3(0, 0, BOX_NM)]
    system.setDefaultPeriodicBoxVectors(*box)

    top = app.Topology()
    chain = top.addChain()
    for name in ("LIG", "PCK"):
        res = top.addResidue(name, chain)
        top.addAtom("C1", app.element.carbon, res)
    top.setPeriodicBoxVectors(box)

    c = BOX_NM / 2
    pos = [openmm.Vec3(c + R_BOUND, c, c), openmm.Vec3(c, c, c)] * u.nanometers
    pdb = tmp_path / "start.pdb"
    with open(pdb, "w") as fh:
        app.PDBFile.writeFile(top, pos, fh)
    return system, top, str(pdb)


def _smd(tmp_path, system, top):
    return SteeredMD(system=system, topology=top, groupA_atoms=[0], groupB_atoms=[1],
                     restart_velocities=True, timestep=0.002, autostop_nc=None,
                     out_dir=str(tmp_path / "out"), platform="CPU", dx_per_move=0.001,
                     max_displacement=0.02, sMD_spring_cte=3765.6, save_freq=5)


def _read(smd, run_id):
    path = f"{smd.out_dir}/sMD_{run_id}.dat"
    return pd.read_csv(path, comment="#"), SMDData.parse_log_metadata(path)


def test_replicas_share_lambda0_and_schedule(tmp_path):
    system, top, pdb = _two_particle_system(tmp_path)
    smd = _smd(tmp_path, system, top)
    ids = [smd.run(pulling_speed=0.01, run_id=str(i), pdb_file=pdb) for i in (1, 2)]
    (d1, m1), (d2, m2) = (_read(smd, i) for i in ids)

    # lambda0 is the COM distance of the shared input structure, identical for both replicas
    assert m1["lambda0_nm"] == pytest.approx(R_BOUND, abs=1e-5)
    assert m2["lambda0_nm"] == m1["lambda0_nm"]
    np.testing.assert_array_equal(d1.r_target.to_numpy(), d2.r_target.to_numpy())
    assert d1.r_target.iloc[0] == pytest.approx(m1["lambda0_nm"] + 0.001, abs=1e-9)
    # the 50 ps ran with independent velocities, so the replicas start at different distances
    assert m1["r_start_nm"] != m2["r_start_nm"]
    # and in the spring: within a few thermal widths of lambda0
    sigma = np.sqrt(0.0083144626 * 300 / 3765.6)
    for m in (m1, m2):
        assert abs(m["r_start_nm"] - m["lambda0_nm"]) < 5 * sigma


def test_explicit_lambda0(tmp_path):
    system, top, pdb = _two_particle_system(tmp_path)
    smd = _smd(tmp_path, system, top)
    rid = smd.run(pulling_speed=0.01, run_id="1", pdb_file=pdb, lambda0_nm=1.03)
    d, m = _read(smd, rid)
    assert m["lambda0_nm"] == pytest.approx(1.03, abs=1e-12)
    assert d.r_target.iloc[0] == pytest.approx(1.031, abs=1e-9)
    # equilibrated in bond (500) + spring (3765.6 at 1.03): within 5 sd of that Gaussian
    k_tot = 500.0 + 3765.6
    mean = (500.0 * R_BOUND + 3765.6 * 1.03) / k_tot
    assert abs(m["r_start_nm"] - mean) < 5 * np.sqrt(0.0083144626 * 300 / k_tot)
