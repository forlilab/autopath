"""Regression test for the cvpack/geometry r0 cross-check in
SteeredMD.pull_single_direction.

Reproduces the intermittent bug where cvpack's CentroidFunction CV
(com_dist.getValue) returns a value several nm off the true minimum-image
group-centroid distance (observed: 7.6 nm vs a true 1.7 nm). The old code
only logged a WARNING on disagreement and then kept using the (wrong)
cvpack scalar for the move cap, which made r_max - initial_r0_nm go
negative and floored sMD_moves to 1. The fix must raise instead of
silently proceeding.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import openmm.unit as openmmunit

from autopath.pulling.steered_md import SteeredMD


def test_raises_when_cvpack_cv_disagrees_with_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr("autopath.pulling.steered_md.add_reporters", lambda *a, **k: None)

    box_vecs = [[v * openmmunit.nanometers for v in row]
                for row in np.diag([12.7, 13.3, 10.0])]
    positions = np.array([[1.743, 0.0, 0.0], [0.0, 0.0, 0.0]]) * openmmunit.nanometers
    state = SimpleNamespace(
        getPeriodicBoxVectors=lambda: box_vecs,
        getPositions=lambda **kw: positions,
    )
    simulation = SimpleNamespace(
        context=SimpleNamespace(
            setParameter=lambda *a, **k: None,
            getState=lambda **kw: state,
        ),
    )
    smd = SimpleNamespace(
        out_dir=str(tmp_path),
        use_GReweighting=False,
        com_dist=SimpleNamespace(getValue=lambda *a, **k: 7.612 * openmmunit.nanometers),
        sMD_moves=2700,
        steps_per_move=1,
        save_freq=1,
        sMD_max_r_offset=3.0,
        projection_params=None,
        pbc_safety_nm=0.5,
        groupA_atoms=[0],
        groupB_atoms=[1],
        dx_per_move=0.001 * openmmunit.nanometers,
    )

    with pytest.raises(RuntimeError, match="differs from cvpack"):
        SteeredMD.pull_single_direction(smd, simulation, run_id=1, direction="forward",
                                        lambda0=1.743 * openmmunit.nanometers)
