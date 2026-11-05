"""A pull that crashes mid-run must not leave an ``sMD_*.dat`` that counts as a replica."""
import glob
from types import SimpleNamespace

import numpy as np
import openmm.unit as openmmunit
import pytest

from autopath.pulling.steered_md import SteeredMD


class _Crash(RuntimeError):
    pass


def _raise(*a, **k):
    raise _Crash("NaN in pull")


def test_crashed_pull_leaves_no_dat(tmp_path, monkeypatch):
    monkeypatch.setattr("autopath.pulling.steered_md.add_reporters", lambda *a, **k: None)
    box = [[x * openmmunit.nanometers for x in row] for row in np.diag([9.0, 9.0, 9.0])]
    positions = np.array([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]]) * openmmunit.nanometers
    state = SimpleNamespace(getPeriodicBoxVectors=lambda: box,
                            getPositions=lambda **kw: positions,
                            getTime=lambda: 0.0 * openmmunit.picoseconds)
    sim = SimpleNamespace(context=SimpleNamespace(setParameter=lambda *a, **k: None,
                                                  getState=lambda **kw: state))
    smd = SimpleNamespace(
        out_dir=str(tmp_path),
        com_dist=SimpleNamespace(getValue=lambda *a, **k: 1.5 * openmmunit.nanometers),
        com_force=SimpleNamespace(getValue=_raise),
        sMD_moves=10, steps_per_move=1, save_freq=1, sMD_max_r_offset=0.0,
        pbc_safety_nm=0.5, groupA_atoms=[0], groupB_atoms=[1],
        dx_per_move=0.001 * openmmunit.nanometers,
        autostop_nc=None, autostop_nc_window=2, subset_protein_CA=None,
        _resolve_geom_features=lambda: [], _GEOM_POCKET=set(),
        force_n_samples=1, force_sample_stride=1,
        sMD_spring_cte=100.0 * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2,
    )
    with pytest.raises(_Crash):
        SteeredMD.pull_single_direction(smd, sim, run_id="replica-1_v0.01_forward",
                                        direction="forward")
    assert glob.glob(f"{tmp_path}/sMD_*.dat") == []
