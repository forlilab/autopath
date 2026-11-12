"""A new pull never reuses the time-of-day replica ID of an existing pull (no overwrite)."""
from datetime import datetime

import autopath.pulling.steered_md as smd_mod
from autopath.pulling.steered_md import SteeredMD


def test_new_run_id_skips_existing(tmp_path, monkeypatch):
    times = iter([datetime(2026, 11, 10, 8, 23, 29), datetime(2026, 11, 10, 8, 23, 29),
                  datetime(2026, 11, 10, 8, 23, 30)])

    class _Clock:
        @staticmethod
        def now():
            return next(times)

    monkeypatch.setattr(smd_mod, "datetime", _Clock)
    monkeypatch.setattr(smd_mod.time, "sleep", lambda s: None)
    (tmp_path / "sMD_replica-082329_v0.01_forward.dat").write_text("old pull")
    smd = SteeredMD.__new__(SteeredMD)
    smd.out_dir = str(tmp_path)
    assert smd._new_run_id(0.01, "forward") == "replica-082330_v0.01_forward"


def test_other_speed_may_share_the_second(tmp_path, monkeypatch):
    monkeypatch.setattr(smd_mod, "datetime", type("C", (), {"now": staticmethod(lambda: datetime(2026, 11, 10, 8, 23, 29))}))
    (tmp_path / "sMD_replica-082329_v0.01_forward.dat").write_text("old pull")
    smd = SteeredMD.__new__(SteeredMD)
    smd.out_dir = str(tmp_path)
    assert smd._new_run_id(0.005, "forward") == "replica-082329_v0.005_forward"
