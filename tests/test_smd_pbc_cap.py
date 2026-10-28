import logging
import re
from types import SimpleNamespace

import numpy as np
import openmm.unit as openmmunit
import pytest

from autopath.pulling.pbc import lattice_pbc_limit, triclinic_minimum_image
from autopath.pulling.steered_md import SteeredMD


def _dodecahedron(d):
    # OpenMM's square-xy rhombic dodecahedron, reduced form.
    return np.array([[d, 0, 0], [0, d, 0], [d / 2, d / 2, d * np.sqrt(2) / 2]])


def _old_cap(v, L):
    abs_v = np.abs(v)
    with np.errstate(divide="ignore"):
        per = np.where(abs_v > 1e-6, (L / 2.0) * np.linalg.norm(v) / abs_v, np.inf)
    return float(per.min())


def test_orthorhombic_cap_equals_old_formula():
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = rng.uniform(4.0, 15.0, 3)
        v = rng.normal(size=3)
        v[rng.integers(0, 3)] *= rng.choice([1.0, 0.0])
        lim, _ = lattice_pbc_limit(v / np.linalg.norm(v), np.diag(L))
        assert lim == pytest.approx(_old_cap(v, L), rel=1e-12)


def test_orthorhombic_minimum_image_matches_old():
    rng = np.random.default_rng(1)
    L = np.array([7.3, 9.1, 11.7])
    d = rng.uniform(-30, 30, size=(500, 3))
    np.testing.assert_allclose(triclinic_minimum_image(d, np.diag(L)), d - np.round(d / L) * L, atol=1e-12)


def test_dodecahedron_lattice_cap_exceeds_old_diagonal_cap():
    d = 8.0
    box = _dodecahedron(d)
    n = np.array([0.0, 0.0, 1.0])
    lim, ijk = lattice_pbc_limit(n, box)
    T = np.asarray(ijk, float) @ box
    assert lim == pytest.approx(T @ T / (2 * abs(n @ T)))
    assert lim == pytest.approx(d / np.sqrt(2))
    assert lim > _old_cap(n, np.diag(box)) + 1.0


def test_dodecahedron_reduction_recovers_in_cell_vectors():
    box = _dodecahedron(8.0)
    half = np.diag(box) / 2
    rng = np.random.default_rng(2)
    for _ in range(200):
        v = rng.uniform(-0.99, 0.99, 3) * half
        far = v + np.array(rng.integers(-2, 3, 3), float) @ box
        np.testing.assert_allclose(triclinic_minimum_image(far, box), v, atol=1e-9)


class _Stop(Exception):
    pass


class _FakeSMD(SimpleNamespace):
    @property
    def autostop_nc_window(self):
        # Halts pull_single_direction right after the move cap.
        raise _Stop


def _run_cap(tmp_path, box, pos, n_moves, monkeypatch, caplog):
    monkeypatch.setattr("autopath.pulling.steered_md.add_reporters", lambda *a, **k: None)
    box_vecs = [[x * openmmunit.nanometers for x in row] for row in box]
    positions = np.asarray(pos) * openmmunit.nanometers
    r = float(np.linalg.norm(triclinic_minimum_image(np.subtract(*pos), box)))
    state = SimpleNamespace(getPeriodicBoxVectors=lambda: box_vecs, getPositions=lambda **kw: positions)
    sim = SimpleNamespace(context=SimpleNamespace(setParameter=lambda *a, **k: None,
                                                  getState=lambda **kw: state))
    smd = _FakeSMD(out_dir=str(tmp_path), use_GReweighting=False,
                   com_dist=SimpleNamespace(getValue=lambda *a, **k: r * openmmunit.nanometers),
                   sMD_moves=n_moves, steps_per_move=1, save_freq=1, sMD_max_r_offset=100.0,
                   pbc_safety_nm=0.5, groupA_atoms=[0], groupB_atoms=[1],
                   dx_per_move=0.001 * openmmunit.nanometers)
    with caplog.at_level(logging.WARNING, logger="autopath"), pytest.raises(_Stop):
        SteeredMD.pull_single_direction(smd, sim, run_id=1, direction="forward",
                                        lambda0=r * openmmunit.nanometers)
    return r


@pytest.mark.parametrize("wrapped", [False, True])
def test_pull_cap_uses_lattice_limit_in_dodecahedron(tmp_path, monkeypatch, caplog, wrapped):
    d = 8.0
    box = _dodecahedron(d)
    # COM-COM vector along z; wrapped stores it across the c boundary to exercise the minimum image.
    pos = [[0.1, 0.2, 1.5], [0.1, 0.2, 0.0]]
    if wrapped:
        pos[0] = list(np.asarray(pos[0]) + box[2] + box[1])
    r = _run_cap(tmp_path, box, pos, 10000, monkeypatch, caplog)
    assert r == pytest.approx(1.5)
    want = int((d / np.sqrt(2) - 0.5 - r) / 0.001)
    m = re.search(r"Capping sMD_moves 10000 → (\d+)", caplog.text)
    assert m and int(m.group(1)) == want
    assert "non-orthorhombic" not in caplog.text
    assert "lattice vector" in caplog.text


def test_pull_cap_unchanged_in_orthorhombic_box(tmp_path, monkeypatch, caplog):
    L = np.array([7.0, 9.0, 11.0])
    v = np.array([1.0, 0.6, 0.3])
    r = _run_cap(tmp_path, np.diag(L), [list(v), [0.0, 0.0, 0.0]], 10000, monkeypatch, caplog)
    want = int((_old_cap(v, L) - 0.5 - r) / 0.001)
    m = re.search(r"Capping sMD_moves 10000 → (\d+)", caplog.text)
    assert m and int(m.group(1)) == want
