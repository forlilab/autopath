"""Options set in the JSON config reach AutoPath through the CLI entry point."""
import json
import sys

from autopath.cli import run_AutoPath

NON_DEFAULT = {
    "sMD_conv_window": 7,
    "sMD_conv_streak": 4,
    "sMD_autostop_estimator": "jarzynski",
    "sMD_alternate_speeds": True,
}


def test_autostop_options_are_forwarded_to_autopath(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self, lig):
            seen["ran"] = lig

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"sMD": NON_DEFAULT}))
    lig = tmp_path / "lig.sdf"
    lig.write_text("")
    monkeypatch.setattr(run_AutoPath, "AutoPath", Recorder)
    monkeypatch.setattr(sys, "argv", ["run_AutoPath", "-c", str(cfg), "-l", str(lig)])

    run_AutoPath.main()

    assert seen["ran"] == str(lig)
    assert {k: seen.get(k) for k in NON_DEFAULT} == NON_DEFAULT
