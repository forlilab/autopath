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
