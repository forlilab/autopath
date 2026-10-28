import os

from autopath.pulling.SMDData import SMDData


def test_log_to_traj_file_keeps_directories():
    p = "/data/analog_series/logs/sMD_replica-123_v0.01_forward.dat"
    assert SMDData._log_to_traj_file(p) == "/data/analog_series/logs/sMD_replica-123_v0.01_forward_aligned.dcd"


def test_log_to_traj_file_maps_log_in_file_name():
    p = os.path.join("runs", "sMD_log_123_v0.01_forward.dat")
    assert SMDData._log_to_traj_file(p) == os.path.join("runs", "sMD_traj_123_v0.01_forward_aligned.dcd")


def test_traj_and_log_names_round_trip():
    p = "/data/catalog/sMD_replica-7_v0.005_forward.dat"
    traj = SMDData._log_to_traj_file(p)
    assert SMDData._traj_to_log_name(traj) == "sMD_replica-7_v0.005_forward"
