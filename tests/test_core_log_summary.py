from collections import Counter

from autopath.autopath_core import end_summary, smd_speed_summary


def test_smd_speed_summary_capped():
    assert smd_speed_summary(0.015, Counter(autostop=46, pbc_guard=4), 2534) == (
        "sMD v=0.015 nm/ps: 50 pulls (46 autostop, 4 PBC guard stop), moves capped to 2534")


def test_smd_speed_summary_uncapped():
    assert smd_speed_summary(0.02, Counter(end=5), None) == "sMD v=0.02 nm/ps: 5 pulls (5 reached end)"


def test_end_summary():
    assert end_summary(150, 7, 312.44) == "== done: 150 sMD pulls, 7 warnings, 312.4 min =="
    assert end_summary(None, 0, 5.24) == "== done: 0 warnings, 5.2 min =="
