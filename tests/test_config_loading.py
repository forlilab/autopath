import json

import pytest

from autopath.config import Config


def _load(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return Config.from_config(str(p))


def test_flat_config_sets_scalars(tmp_path):
    c = _load(tmp_path, {"temperature": 310.0, "sMD_conv_window": 7})
    assert c.temperature == 310.0 and c.sMD_conv_window == 7


def test_flat_config_keeps_dict_valued_parameters(tmp_path):
    c = _load(tmp_path, {"sMD_pulling_speeds": {"0.002": 4}})
    assert c.sMD_pulling_speeds == {0.002: 4}


def test_flat_config_maps_legacy_keys(tmp_path):
    c = _load(tmp_path, {"sMD_replicas": 3})
    assert c.sMD_pulling_speeds == {0.001: 3}


def test_nested_config_still_works(tmp_path):
    c = _load(tmp_path, {"General": {"temperature": 310.0},
                         "sMD": {"sMD_pulling_speeds": {"0.002": 4}}})
    assert c.temperature == 310.0 and c.sMD_pulling_speeds == {0.002: 4}


def test_mixed_flat_and_nested(tmp_path):
    c = _load(tmp_path, {"temperature": 310.0, "sMD": {"sMD_conv_streak": 4}})
    assert c.temperature == 310.0 and c.sMD_conv_streak == 4


def test_unknown_flat_key_raises(tmp_path):
    with pytest.raises(ValueError, match="not_a_param"):
        _load(tmp_path, {"not_a_param": 1})
