"""Every ``forces`` list in a protocol must be parallel to ``components_lookup``.

Pinned bug. ``run_restrained_minimization`` and ``run_restrained_md`` both pair
components with force constants using ``zip``, which stops at the shorter of the two
and reports nothing. A stage that declares too few constants therefore leaves the
trailing components at whatever value they last held, silently:

    >>> update_force_constants(sim, {"solute": 50.0, "second": 50.0})
    >>> run_restrained_minimization(sim, ["solute", "second"], [{"forces": [0.0]}])
    {'solute': 0.0, 'second': 50.0}

If that happens on the final minimization stage, a component stays pinned at
50 kcal/mol/A^2 all the way into the warm-up with nothing in the log to say so --
the same silent-restraint failure mode as the reinitialize bug pinned in
``test_warmup_restraints.py``, reached by a different route.

The protocol is therefore validated at load time, before any GPU work happens.
"""

import json

import pytest

from autopath.equilibration import Equilibration

PROTOCOLS = [
    "autopath/data/eq_lig-prot_5ns_4fs.json",
    "autopath/data/eq_lig-prot-memb_10ns_4fs.json",
]


def _load(protocol):
    with open(protocol) as f:
        return json.load(f)


def _write(tmp_path, protocol, name="protocol.json"):
    fname = tmp_path / name
    fname.write_text(json.dumps(protocol))
    return str(fname)


def _build(tmp_path, protocol_fname):
    return Equilibration(
        out_dir=str(tmp_path), protocol_fname=protocol_fname, platform="CPU"
    )


# --------------------------------------------------------------------------- #
# positive control: the shipped protocols are internally consistent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_shipped_protocols_are_parallel(protocol, tmp_path):
    raw = _load(protocol)
    n = len(raw["components_lookup"])
    for section in ("minimization", "equilibration"):
        for stage in raw[section]:
            assert len(stage["forces"]) == n, f"{protocol}: {section} {stage['name']}"
    assert len(raw["warmup"]["forces"]) == n

    _build(tmp_path, protocol)  # must not raise


# --------------------------------------------------------------------------- #
# each section is checked
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("section", ["minimization", "equilibration"])
def test_short_stage_forces_raise(section, tmp_path):
    protocol = _load(PROTOCOLS[0])
    protocol[section][1]["forces"] = protocol[section][1]["forces"][:-1]

    with pytest.raises(ValueError) as excinfo:
        _build(tmp_path, _write(tmp_path, protocol))

    message = str(excinfo.value)
    assert section in message, message
    assert "Stage 2" in message, message  # names the offending stage


@pytest.mark.parametrize("section", ["minimization", "equilibration"])
def test_long_stage_forces_raise(section, tmp_path):
    protocol = _load(PROTOCOLS[0])
    protocol[section][0]["forces"] = protocol[section][0]["forces"] + [1.0]

    with pytest.raises(ValueError, match=section):
        _build(tmp_path, _write(tmp_path, protocol))


def test_warmup_forces_length_mismatch_raises(tmp_path):
    protocol = _load(PROTOCOLS[0])
    protocol["warmup"]["forces"] = [15.0]

    with pytest.raises(ValueError, match="warmup"):
        _build(tmp_path, _write(tmp_path, protocol))


def test_warmup_forces_default_to_15_when_absent(tmp_path):
    """Protocols predating the warmup 'forces' key keep the documented default."""
    protocol = _load(PROTOCOLS[0])
    protocol["warmup"].pop("forces", None)

    equil = _build(tmp_path, _write(tmp_path, protocol))
    assert equil.warmup_forces == [15.0] * len(equil.components_lookup)


def test_validation_happens_at_load_not_at_run(tmp_path):
    """The error must surface from the constructor, before any simulation is built."""
    protocol = _load(PROTOCOLS[0])
    protocol["minimization"][0]["forces"] = [50.0]

    with pytest.raises(ValueError):
        Equilibration(
            out_dir=str(tmp_path),
            protocol_fname=_write(tmp_path, protocol),
            platform="CPU",
        )
