import pytest

from autopath.pulling.steered_md import STOP_LABELS, pull_summary


def test_pull_summary_format():
    assert pull_summary("replica-001_v0.015_forward", "autostop", 1375, 2534, 1.3756, 2.14) == (
        "[replica-001_v0.015_forward] autostop at move 1375/2534, displacement 1.376 nm, 2.1 min")


@pytest.mark.parametrize("reason,label", [
    ("autostop", "autostop"), ("pbc_guard", "PBC guard stop"),
    ("end", "reached end"), ("r0_reached", "r0 reached 0")])
def test_pull_summary_labels(reason, label):
    assert STOP_LABELS[reason] == label
    assert f"] {label} at move" in pull_summary("r", reason, 1, 2, 0.1, 0.1)


def test_pull_summary_unknown_reason():
    with pytest.raises(KeyError):
        pull_summary("r", "nope", 1, 2, 0.1, 0.1)


def test_run_logs_single_summary_and_counts(tmp_path, caplog):
    import logging
    from autopath.pulling.steered_md import SteeredMD
    from tests.test_smd_shared_lambda0 import _two_particle_system

    system, top, pdb = _two_particle_system(tmp_path)
    smd = SteeredMD(system=system, topology=top, groupA_atoms=[0], groupB_atoms=[1],
                    restart_velocities=True, timestep=0.002, autostop_nc=None,
                    out_dir=str(tmp_path / "o"), platform="Reference", dx_per_move=0.001,
                    max_displacement=0.02, sMD_spring_cte=3765.6, save_freq=5, seed=1)
    with caplog.at_level(logging.DEBUG, logger="autopath"):
        smd.run(pulling_speed=0.01, run_id="a", pdb_file=pdb)
    n = smd.sMD_moves
    info = [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "] reached end at move" in r.getMessage()]
    assert len(info) == 1 and f"reached end at move {n}/{n}," in info[0]
    assert smd.stop_counts[0.01]["end"] == 1
