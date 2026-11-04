"""Contact-weighted ligand RMSF and the unrestrained-window helper."""
import json

import numpy as np
import pandas as pd
import pytest
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from autopath.ap_PLIP import ligand_rmsf_contacts, cw_rmsf
from autopath.equilibration import free_window_start


def _universe(coords):
    """Protein (2 ALA heavy atoms) + ligand UNK (N, C, H) over the given frames."""
    u = mda.Universe.empty(5, n_residues=3, atom_resindex=[0, 1, 2, 2, 2],
                           residue_segindex=[0, 0, 0], trajectory=True)
    u.add_TopologyAttr('name', ['CA', 'CA', 'N1', 'C1', 'H1'])
    u.add_TopologyAttr('element', ['C', 'C', 'N', 'C', 'H'])
    u.add_TopologyAttr('resname', ['ALA', 'ALA', 'UNK'])
    u.add_TopologyAttr('resid', [1, 2, 3])
    u.load_new(np.asarray(coords, dtype=np.float32), format=MemoryReader)
    return u


def _frames():
    """Protein fixed at x=0 and x=20. Ligand N sits 3 A from the first CA and
    jitters along y; ligand C sits 4.5 A away (inside the 5 A pocket shell, outside
    the 3.5 A contact cutoff) except in the last two frames, where it is at 3 A."""
    frames = []
    for i, dy in enumerate([0.0, 0.4, -0.4, 0.8, -0.8, 0.0]):
        cx = 3.0 if i >= 4 else 4.5
        frames.append([[0, 0, 0], [20, 0, 0], [3, dy, 0], [cx, 0, 0.5 * (i % 2)], [3, 1, 0]])
    return frames


def test_per_atom_matches_hand_calculation():
    coords = np.array(_frames(), dtype=float)
    pa = ligand_rmsf_contacts(_universe(coords), 'resname UNK')
    assert list(pa.atom_name) == ['N1', 'C1'], 'hydrogens are excluded'
    lig = coords[:, 2:4]
    rmsf = np.sqrt(((lig - lig.mean(0)) ** 2).sum(-1).mean(0))
    np.testing.assert_allclose(pa.rmsf, rmsf, atol=1e-4)
    d = np.linalg.norm(lig - coords[:, :1], axis=-1)
    np.testing.assert_allclose(pa.contact_freq, (d < 3.5).mean(0))


def test_window_defines_the_pocket_on_its_first_frame():
    """Frame 0 puts the second CA beyond 5 A; from frame 2 on it sits next to the
    ligand C. A window starting at frame 2 must count those contacts."""
    f = _frames()
    for i in range(2, 6):
        f[i][1] = [f[i][3][0] + 3.0, 0, f[i][3][2]]
    u = _universe(f)
    whole, win = ligand_rmsf_contacts(u, 'resname UNK'), ligand_rmsf_contacts(u, 'resname UNK', start=2)
    c_whole, c_win = whole.set_index('atom_name').contact_freq, win.set_index('atom_name').contact_freq
    assert c_whole['C1'] == pytest.approx(2 / 6), 'whole run: second CA not in the frame-0 pocket'
    assert c_win['C1'] == pytest.approx(1.0)
    lig = np.array(f, dtype=float)[2:, 2:4]
    np.testing.assert_allclose(win.rmsf, np.sqrt(((lig - lig.mean(0)) ** 2).sum(-1).mean(0)), atol=1e-4)


def test_cw_rmsf_is_the_contact_weighted_mean():
    pa = pd.DataFrame({'element': ['N', 'O', 'C', 'CL', 'C'], 'rmsf': [1.0, 3.0, 2.0, 4.0, 9.0],
                       'contact_freq': [0.75, 0.25, 0.5, 0.5, 0.0]})
    assert cw_rmsf(pa, 'polar') == pytest.approx((0.75 * 1 + 0.25 * 3) / 1.0)
    assert cw_rmsf(pa, 'nonpolar') == pytest.approx((0.5 * 2 + 0.5 * 4) / 1.0), 'CL counts as non-polar'
    assert cw_rmsf(pa, 'all') == pytest.approx((0.75 + 0.75 + 1 + 2) / 2.0)
    assert cw_rmsf(pa, ['C']) == pytest.approx(2.0)


def test_cw_rmsf_without_contacts_is_not_supported():
    pa = pd.DataFrame({'element': ['N', 'C'], 'rmsf': [1.0, 2.0], 'contact_freq': [0.0, 1.0]})
    with pytest.raises(ValueError, match='no polar'):
        cw_rmsf(pa, 'polar')
    with pytest.raises(ValueError, match='Unknown atom set'):
        cw_rmsf(pa, 'aromatic')


def _protocol(tmp_path, lig_forces):
    p = {'components_lookup': {'protein': 'protein', 'ligand': 'resname UNK'},
         'warmup': {'forces': [15, 15], 'nsteps': 1000, 'stepsize': 0.001},
         'equilibration': [{'forces': [5, f], 'nsteps': 250, 'stepsize': 0.004} for f in lig_forces]}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(p))
    return str(path)


def _state_csv(tmp_path, times):
    path = tmp_path / 'state.csv'
    pd.DataFrame({'#"Progress (%)"': 0, 'Step': 0, 'Time (ps)': times}).to_csv(path, index=False)
    return str(path)


def test_free_window_starts_after_the_last_ligand_restraint(tmp_path):
    # stages of 1 ps; ligand free in the last two (and in an earlier one, which
    # does not count because it is followed by a restrained stage)
    proto = _protocol(tmp_path, [1.0, 0.0, 0.5, 0.0, 0.0])
    csv = _state_csv(tmp_path, np.arange(1, 7) * 1.0)       # frames at 1..6 ps
    assert free_window_start(csv, proto) == 4                # t >= 6 - 2 ps
    assert free_window_start(csv, json.load(open(proto)), n_frames=6) == 4
    with pytest.raises(ValueError, match='frames'):
        free_window_start(csv, proto, n_frames=7)


def test_free_window_needs_an_unrestrained_tail(tmp_path):
    csv = _state_csv(tmp_path, [1.0, 2.0])
    with pytest.raises(ValueError, match='restrained until the end'):
        free_window_start(csv, _protocol(tmp_path, [0.0, 1.0]))
    with pytest.raises(ValueError, match='not in the protocol'):
        free_window_start(csv, _protocol(tmp_path, [0.0]), component='membrane')


def test_cw_rmsf_scores_reports_undefined_sets_only_when_asked():
    from autopath.ap_PLIP import cw_rmsf_scores
    pa = pd.DataFrame({'element': ['N', 'C'], 'rmsf': [1.0, 2.0], 'contact_freq': [0.0, 1.0]})
    with pytest.raises(ValueError, match='no polar'):
        cw_rmsf_scores(pa)
    s = cw_rmsf_scores(pa, on_undefined='nan')
    assert np.isnan(s['cw_rmsf_polar'])
    assert s['cw_rmsf_nonpolar'] == pytest.approx(2.0) and s['cw_rmsf_all'] == pytest.approx(2.0)
