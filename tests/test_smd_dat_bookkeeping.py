import glob
from datetime import datetime

import pandas as pd
import pytest

from autopath.pulling.SMDData import SMDData
from autopath.pulling.steered_md import SteeredMD
from tests.test_smd_shared_lambda0 import _two_particle_system


@pytest.fixture
def run(tmp_path):
    system, top, pdb = _two_particle_system(tmp_path)
    smd = SteeredMD(system=system, topology=top, groupA_atoms=[0], groupB_atoms=[1],
                    restart_velocities=True, timestep=0.002, autostop_nc=None,
                    out_dir=str(tmp_path / "o"), platform="Reference", dx_per_move=0.001,
                    max_displacement=0.02, sMD_spring_cte=3765.6, save_freq=5, seed=1)
    smd.run(pulling_speed=0.01, run_id="a", pdb_file=pdb)
    return smd


def _dat(smd):
    return glob.glob(f"{smd.out_dir}/sMD_*.dat")[0]


def test_dat_header_has_start_time(run):
    smd = run
    datetime.fromisoformat(SMDData.parse_log_metadata(_dat(smd))["start_time"])


def test_dat_trailer_records_stop(run):
    smd = run
    t = SMDData.parse_log_trailer(_dat(smd))
    assert (t["stop_reason"], t["final_move"], t["n_moves"]) == ("end", smd.sMD_moves, smd.sMD_moves)
    assert t["final_displacement_nm"] > 0 and t["wall_minutes"] >= 0


def test_dat_still_reads_with_comment(run):
    smd = run
    df = pd.read_csv(_dat(smd), comment="#")
    assert len(df) == smd.sMD_moves
    assert list(df.columns) == ["step", "time", "r_target", "r_before", "r_after",
                                "force", "force_sem", "U_cvpack", "dW_protocol"]


def test_trailer_absent_on_old_log(tmp_path):
    fn = tmp_path / "old.dat"
    fn.write_text("# spring_constant_kJ_mol_nm2=1\nstep,time\n0,0.0\n1,0.1\n")
    assert SMDData.parse_log_trailer(str(fn)) == {}
