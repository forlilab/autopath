"""SteeredMD(seed=...) makes pulls reproducible per (seed, run_id)."""
import numpy as np
import pandas as pd

from autopath.pulling.steered_md import SteeredMD
from tests.test_smd_shared_lambda0 import _two_particle_system


def _pull(tmp_path, system, top, pdb, seed, run_id, tag):
    smd = SteeredMD(system=system, topology=top, groupA_atoms=[0], groupB_atoms=[1],
                    restart_velocities=True, timestep=0.002, autostop_nc=None,
                    out_dir=str(tmp_path / tag), platform="Reference", dx_per_move=0.001,
                    max_displacement=0.02, sMD_spring_cte=3765.6, save_freq=5, seed=seed)
    rid = smd.run(pulling_speed=0.01, run_id=run_id, pdb_file=pdb)
    d = pd.read_csv(f"{smd.out_dir}/sMD_{rid}.dat", comment="#")
    return np.cumsum(d.dW_protocol.to_numpy()), d.r_after.to_numpy()


def test_same_seed_identical_different_seed_differs(tmp_path):
    system, top, pdb = _two_particle_system(tmp_path)
    w1, r1 = _pull(tmp_path, system, top, pdb, 7, "1", "a")
    w2, r2 = _pull(tmp_path, system, top, pdb, 7, "1", "b")
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(r1, r2)
    w3, _ = _pull(tmp_path, system, top, pdb, 8, "1", "c")
    w4, _ = _pull(tmp_path, system, top, pdb, 7, "2", "d")   # same seed, another replica
    assert not np.array_equal(w1, w3)
    assert not np.array_equal(w1, w4)


def test_run_seed_is_valid_openmm_seed():
    smd = SteeredMD.__new__(SteeredMD)
    smd.seed = None
    assert smd.run_seed("replica-1_v0.01_forward") is None
    smd.seed = 0
    seeds = {smd.run_seed(f"replica-{i}_v0.01_forward") for i in range(200)}
    assert len(seeds) == 200 and min(seeds) >= 1 and max(seeds) < 2**31
