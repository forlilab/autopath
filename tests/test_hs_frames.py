import numpy as np
import pytest

from autopath.pulling import frame_steps, hs_frame_weights

from .hs_toys import flat, simulate_pulls, write_dat


def _write_dcds(dat_files, save_freq):
    from MDAnalysis.lib.formats.libdcd import DCDFile
    import pandas as pd

    for fn in dat_files:
        n_rows = len(pd.read_csv(fn, comment="#"))
        out = fn.replace("log", "traj").replace(".dat", "_aligned.dcd")
        with DCDFile(out, "w") as f:
            f.write_header(remarks="t", natoms=1, istart=0, nsavc=1, delta=1.0, is_periodic=False)
            for _ in range(n_rows // save_freq):
                f.write(np.zeros((1, 3), dtype=np.float32))


def test_frame_steps_mapping():
    np.testing.assert_array_equal(frame_steps(1201, 240, 5)[:3], [4, 9, 14])
    assert frame_steps(1496, 299, 5)[-1] == 1494


def test_frame_count_mismatch_is_not_supported():
    with pytest.raises(ValueError, match="not supported"):
        frame_steps(1201, 241, 5)


def test_frame_weights_match_row_weights(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=6, speed=0.01, length_nm=0.1,
                          stop_steps=[60] + [None] * 5, seed=14)
    files = write_dat(rows, tmp_path)
    _write_dcds(files, save_freq=5)
    fw = hs_frame_weights(files, save_freq=5)
    assert fw.attrs["n_frames_excluded"] > 0
    assert set(fw.step) <= set(range(4, 61, 5))
    np.testing.assert_allclose(np.exp(fw.logw).sum(), 1.0)
