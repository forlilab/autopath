"""SteeredMD projection mode: pull z = n·(COM_ligand − COM_pocket) in the protein frame."""
import logging
import re

import numpy as np
import openmm
import pandas as pd
import pytest

from autopath.pulling.SMDData import SMDData
from autopath.pulling.steered_md import SteeredMD
from tests import _projection_toys as toy

DX = 0.001
K = 3765.6


@pytest.fixture(autouse=True)
def _one_cpu_thread():
    # an 8-atom toy runs ~30x slower when the CPU platform spreads it over every core
    cpu = openmm.Platform.getPlatformByName("CPU")
    old = cpu.getPropertyDefaultValue("Threads")
    cpu.setPropertyDefaultValue("Threads", "1")
    yield
    cpu.setPropertyDefaultValue("Threads", old)


def _smd(tmp_path, system, top, proj=None, lateral=False, max_displacement=0.02, **kw):
    return SteeredMD(system=system, topology=top, groupA_atoms=toy.LIGAND, groupB_atoms=toy.POCKET,
                     restart_velocities=True, timestep=0.002, autostop_nc=None,
                     out_dir=str(tmp_path / "out"), platform="CPU", dx_per_move=DX,
                     max_displacement=max_displacement, sMD_spring_cte=K, save_freq=5,
                     projection_params=proj, lateral_restraint=lateral, **kw)


def _read(smd, run_id):
    path = f"{smd.out_dir}/sMD_{run_id}.dat"
    return pd.read_csv(path, comment="#"), SMDData.parse_log_metadata(path)


def _header(smd, run_id):
    with open(f"{smd.out_dir}/sMD_{run_id}.dat") as fh:
        return [line.strip() for line in fh if line.startswith("#")]


def test_projection_replicas_share_schedule(tmp_path):
    system, top, pdb = toy.smd_system(tmp_path)
    smd = _smd(tmp_path, system, top, proj=toy.params())
    ids = [smd.run(pulling_speed=0.01, run_id=str(i), pdb_file=pdb) for i in (1, 2)]
    (d1, m1), (d2, m2) = (_read(smd, i) for i in ids)
    assert m1["cv"] == "projection" and m2["cv"] == "projection"
    # lambda0 is z of the shared input structure (PDB rounding: 1e-4 nm)
    assert m1["lambda0_nm"] == pytest.approx(toy.z_input(), abs=1e-3)
    assert m2["lambda0_nm"] == m1["lambda0_nm"]
    np.testing.assert_array_equal(d1.r_target.to_numpy(), d2.r_target.to_numpy())
    assert d1.r_target.iloc[0] == pytest.approx(m1["lambda0_nm"] + DX, abs=1e-9)
    assert len(d1) == 20
    sigma = np.sqrt(0.0083144626 * 300 / K)
    for m in (m1, m2):
        assert abs(m["r_start_nm"] - m["lambda0_nm"]) < 5 * sigma


def test_explicit_lambda0_below_input(tmp_path):
    system, top, pdb = toy.smd_system(tmp_path)
    smd = _smd(tmp_path, system, top, proj=toy.params())
    lam0 = toy.z_input() - 0.15
    rid = smd.run(pulling_speed=0.01, run_id="1", pdb_file=pdb, lambda0_nm=lam0)
    d, m = _read(smd, rid)
    assert m["lambda0_nm"] == pytest.approx(lam0, abs=1e-9)
    assert d.r_target.iloc[0] == pytest.approx(lam0 + DX, abs=1e-9)
    # equilibrated in the spring at lam0: z is pulled down from the input value
    assert m["r_start_nm"] < toy.z_input() - 0.075


def test_work_is_spring_only_with_restraint(tmp_path):
    system, top, pdb = toy.smd_system(tmp_path)
    smd = _smd(tmp_path, system, top, proj=toy.params(R_cylinder_ang=0.5, alpha_deg=10.0), lateral=True)
    rid = smd.run(pulling_speed=0.01, run_id="1", pdb_file=pdb)
    d, m = _read(smd, rid)
    lam_new = d.r_target.to_numpy()
    lam_old = np.concatenate([[m["lambda0_nm"]], lam_new[:-1]])
    r = d.r_before.to_numpy()
    expected = 0.5 * K * ((r - lam_new) ** 2 - (r - lam_old) ** 2)
    np.testing.assert_allclose(d.dW_protocol.to_numpy(), expected, rtol=0, atol=1e-6)


def test_header_restraint_fields(tmp_path):
    system, top, pdb = toy.smd_system(tmp_path)
    p = toy.params(R_cylinder_ang=1.5, z_cc_ang=7.0, alpha_deg=30.0, k_xy=5.0)
    smd = _smd(tmp_path, system, top, proj=p, lateral=True, max_displacement=0.005)
    m = _read(smd, smd.run(pulling_speed=0.01, run_id="1", pdb_file=pdb))[1]
    assert m["cv"] == "projection"
    assert m["restraint"] == "funnel"
    np.testing.assert_allclose([float(x) for x in m["axis_coeffs"].split(",")], p["axis_coeffs"], atol=1e-9)
    assert m["R_cylinder_ang"] == pytest.approx(1.5)
    assert m["z_cc_ang"] == pytest.approx(7.0)
    assert m["alpha_deg"] == pytest.approx(30.0)
    assert m["k_xy_kcal_per_mol_per_ang2"] == pytest.approx(5.0)

    smd = _smd(tmp_path / "free", system, top, proj=p, max_displacement=0.005)
    rid = smd.run(pulling_speed=0.01, run_id="1", pdb_file=pdb)
    m = _read(smd, rid)[1]
    assert m["restraint"] == "none"
    assert not any(k in m for k in ("R_cylinder_ang", "z_cc_ang", "alpha_deg", "k_xy_kcal_per_mol_per_ang2"))


def test_backward_projection_not_supported(tmp_path):
    system, top, pdb = toy.smd_system(tmp_path)
    smd = _smd(tmp_path, system, top, proj=toy.params())
    with pytest.raises(ValueError, match="^not supported:"):
        smd.run(pulling_speed=0.01, pulling_direction="backward", run_id="1", pdb_file=pdb)


@pytest.mark.parametrize("ligand, pocket", [([2], toy.POCKET), (toy.LIGAND, [0, 1, 4])])
def test_mismatched_groups_not_supported(tmp_path, ligand, pocket):
    system, top, _ = toy.smd_system(tmp_path)
    with pytest.raises(ValueError, match="^not supported:"):
        SteeredMD(system=system, topology=top, groupA_atoms=ligand, groupB_atoms=pocket,
                  out_dir=str(tmp_path / "out"), platform="CPU", projection_params=toy.params())


def test_single_atom_group_not_supported(tmp_path):
    # SteeredMD drops mass weighting for one-atom groups; the projection geometry is mass-weighted
    system, top, _ = toy.smd_system(tmp_path)
    p = dict(toy.params(), guest_index=[2])
    with pytest.raises(ValueError, match="^not supported:"):
        SteeredMD(system=system, topology=top, groupA_atoms=[2], groupB_atoms=toy.POCKET,
                  out_dir=str(tmp_path / "out"), platform="CPU", projection_params=p)


def test_lab_frame_params_not_supported(tmp_path):
    system, top, _ = toy.smd_system(tmp_path)
    lab = {k: v for k, v in toy.params().items() if k not in ("frame_b_index", "frame_c_index", "axis_coeffs")}
    with pytest.raises(ValueError, match="^not supported:"):
        _smd(tmp_path, system, top, proj=lab)


def test_lateral_restraint_without_params_not_supported(tmp_path):
    system, top, _ = toy.smd_system(tmp_path)
    with pytest.raises(ValueError, match="^not supported:"):
        _smd(tmp_path, system, top, lateral=True)


def test_pbc_cap_along_axis(tmp_path, caplog):
    system, top, pdb = toy.smd_system(tmp_path)
    L, safety, max_disp = toy.BOX_NM, 2.5, 2.4
    # the protein tumbles in the 50 ps, so n is unknown: min_i (L/2)/|n_i| lies in [L/2, sqrt(3) L/2]
    assert toy.z_input() + max_disp > np.sqrt(3) * L / 2 - safety
    smd = _smd(tmp_path, system, top, proj=toy.params(), max_displacement=max_disp, pbc_safety_nm=safety)
    with caplog.at_level(logging.WARNING, logger="autopath"):
        rid = smd.run(pulling_speed=0.25, run_id="1", pdb_file=pdb)
    d, m = _read(smd, rid)
    msg = [r.getMessage() for r in caplog.records if "Capping sMD_moves" in r.getMessage()]
    assert len(msg) == 1
    z_end_max = float(re.search(r"limit \(([-0-9.]+) nm\)", msg[0]).group(1))
    assert L / 2 - safety <= z_end_max <= np.sqrt(3) * L / 2 - safety
    assert len(d) < round(max_disp / DX)
    assert d.r_target.iloc[-1] < z_end_max
    assert d.r_target.iloc[-1] > z_end_max - 2 * DX


def test_distance_mode_header(tmp_path):
    system, top, pdb = toy.smd_system(tmp_path)
    smd = _smd(tmp_path, system, top, max_displacement=0.005)
    rid = smd.run(pulling_speed=0.01, run_id="1", pdb_file=pdb)
    d, m = _read(smd, rid)
    assert m["cv"] == "distance"
    assert "# cv=projection" not in _header(smd, rid)
    assert list(d.columns) == ["step", "time", "r_target", "r_before", "r_after", "force", "force_sem",
                               "U_cvpack", "dW_protocol"]
    # distance mode: lambda0 is the COM-COM distance of the input structure
    p = toy.positions()
    r_in = np.linalg.norm(toy.com(p, toy.LIGAND) - toy.com(p, toy.POCKET))
    assert m["lambda0_nm"] == pytest.approx(r_in, abs=1e-3)
