from autopath.pulling.SMDData import SMDData, resolve_smd_dcd


def test_prefers_aligned_then_raw(tmp_path):
    stem = tmp_path / "sMD_replica-120000_v0.001_forward"
    assert resolve_smd_dcd(str(stem)) == f"{stem}_aligned.dcd"  # neither: legacy name
    (tmp_path / f"{stem.name}.dcd").write_bytes(b"")
    assert resolve_smd_dcd(str(stem)) == f"{stem}.dcd"
    (tmp_path / f"{stem.name}_aligned.dcd").write_bytes(b"")
    assert resolve_smd_dcd(str(stem)) == f"{stem}_aligned.dcd"


def test_log_to_traj_raw_roundtrip(tmp_path):
    log = tmp_path / "sMD_replica-120000_v0.001_forward.dat"
    raw = tmp_path / "sMD_replica-120000_v0.001_forward.dcd"
    raw.write_bytes(b"")
    assert SMDData._log_to_traj_file(str(log)) == str(raw)
    assert SMDData._traj_to_log_name(str(raw)) == log.stem
