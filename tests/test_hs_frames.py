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


def test_frame_weights_equal_frame_slice_estimate(tmp_path):
    import pandas as pd
    from scipy.special import logsumexp

    from autopath.pulling import SMDData
    from autopath.pulling.hummer_szabo import f_jarzynski, hs_log_weights, slice_table, slice_weights

    rows = simulate_pulls(*flat(), n_pulls=6, speed=0.01, length_nm=0.1,
                          stop_steps=[60] + [None] * 5, seed=14)
    files = write_dat(rows, tmp_path)
    _write_dcds(files, save_freq=5)
    fw = hs_frame_weights(files, save_freq=5)

    # Independent: HS on the frame-step rows/slices only, within the common support.
    smd = SMDData(files, temperature=300.0)
    raw = smd.raw_data
    fr = raw[((raw.step + 1) % 5 == 0) & (raw.step <= 60)].reset_index(drop=True)
    sl = slice_table(fr, smd.protocol_grids, smd.beta)
    lw = hs_log_weights(fr, sl, f_jarzynski(sl, smd.beta), slice_weights(sl, "ess"),
                        smd.spring_constant, smd.beta)
    ref = fr[["trajname", "step"]].assign(ref=lw - logsumexp(lw))
    m = fw.merge(ref, on=["trajname", "step"], how="outer")
    assert len(m) == len(fw) == len(ref)
    np.testing.assert_allclose(m.logw.to_numpy(), m.ref.to_numpy(), rtol=0, atol=1e-10)


def test_dropped_pull_frames_are_counted(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=5, speed=0.01, length_nm=0.1, seed=15)
    bad = rows.trajname.unique()[2]
    # SMDData drops a forward pull whose distance falls below completion_threshold_nm.
    rows.loc[rows.trajname == bad, "r_after"] -= 0.95
    files = write_dat(rows, tmp_path)
    _write_dcds(files, save_freq=5)
    fw = hs_frame_weights(files, save_freq=5)
    # SMDData names a pull after its log file.
    assert fw.attrs["pulls_dropped"] == [f"sMD_{bad}_forward"]
    assert fw.trajname.nunique() == 4 and not fw.trajname.str.contains(bad + "_").any()
    assert fw.attrs["n_frames_excluded"] == 100 // 5
    np.testing.assert_allclose(np.exp(fw.logw).sum(), 1.0)


def test_step_gap_is_not_supported(tmp_path):
    import pandas as pd

    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.1, seed=16)
    files = write_dat(rows, tmp_path)
    with open(files[1]) as f:
        header = f.readline()
    d = pd.read_csv(files[1], comment="#").drop(index=37)
    with open(files[1], "w") as f:
        f.write(header)
        d.to_csv(f, index=False)
    _write_dcds(files, save_freq=5)
    with pytest.raises(ValueError, match="not supported: .*step"):
        hs_frame_weights(files, save_freq=5)
