import numpy as np
import pandas as pd
import pytest

from autopath.metadynamics.learned.select import (fisher, hidden_variable_p, labels_from_r,
                                                   prune, pull_frames)


def test_pull_frames_uses_own_frame_count(tmp_path):
    p = tmp_path / "sMD_replica-1_v0.01_forward.dat"
    t = np.linspace(0, 100, 201)
    pd.DataFrame({"step": np.arange(201), "time": t, "r_after": 0.8 + 0.01 * t,
                  "dW_protocol": np.full(201, 0.5)}).to_csv(p, index=False)
    a, b = pull_frames(str(p), 11), pull_frames(str(p), 6)
    assert a.t_ps.iloc[-1] == pytest.approx(100) and b.t_ps.iloc[-1] == pytest.approx(100)
    assert a.r.iloc[5] == pytest.approx(0.8 + 0.01 * 50)
    assert a.W.iloc[-1] == pytest.approx(0.5 * 201)


def test_labels_from_r():
    first = np.array([0.79, 0.80, 0.81])
    lab = labels_from_r(np.array([0.80, 1.2, 2.0, 2.2]), first, r_max=2.2)
    assert list(lab) == ["bound", "vestibule", "bulk", "bulk"]


def test_prune_keeps_best_of_correlated_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    V = np.c_[a, a + 0.01 * rng.normal(size=500), rng.normal(size=500)]
    kept = prune(V, ["a", "a2", "b"], score=np.array([1.0, 2.0, 0.5]))
    assert sorted(kept) == [1, 2]


def test_fisher_separated_vs_overlapping():
    y = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    assert fisher(np.r_[np.zeros(100), np.ones(100) * 5], y) > 1e3
    rng = np.random.default_rng(0)
    assert fisher(rng.normal(size=200), y) < 0.1


def _pulls(n_pull=60, n_frame=40, planted=True, speeds=(0.01, 0.02), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in speeds:
        for p in range(n_pull):
            hidden = rng.normal()
            r = np.linspace(0.8, 2.0, n_frame)
            W = np.cumsum(np.ones(n_frame)) + (3.0 * hidden * (r > 1.0) if planted else 0)
            W = W + rng.normal(scale=0.5, size=n_frame)
            x = hidden + rng.normal(scale=0.3, size=n_frame)
            rows.append(pd.DataFrame({"r": r, "W": W, "x": x, "pull": f"{v}_{p}", "speed": v}))
    return pd.concat(rows, ignore_index=True)


def test_hidden_variable_finds_planted_feature():
    df = _pulls(planted=True)
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(1))
    assert p < 0.05 and rho > 0


def test_hidden_variable_noise_rate_is_nominal():
    hits = 0
    for s in range(40):
        df = _pulls(planted=False, seed=100 + s)
        p, _ = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                                 df.speed.values, n_perm=100, rng=np.random.default_rng(s))
        hits += p < 0.05
    assert hits <= 6                              # nominal 5% of 40 = 2


def test_hidden_variable_single_speed_uses_split_halves():
    df = _pulls(planted=True, speeds=(0.01,), n_pull=80)
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(2))
    assert p < 0.05 and rho > 0


def test_hidden_variable_skips_bins_where_feature_is_constant():
    df = _pulls(planted=True)
    df.loc[df.r > 1.6, "x"] = 0.0
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(3))
    assert p < 0.05 and rho > 0


def test_hidden_variable_constant_feature_is_null():
    df = _pulls(planted=True)
    df["x"] = 0.0
    assert hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                             df.speed.values, n_perm=50, rng=np.random.default_rng(4)) == (1.0, 0.0)


def test_hidden_variable_sign_disagreement_across_speeds_is_null():
    df = _pulls(planted=True)
    df.loc[df.speed == 0.02, "x"] *= -1.0
    p, _ = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                             df.speed.values, n_perm=200, rng=np.random.default_rng(5))
    assert p >= 0.05


def _one_bin_pulls(n_pull=45, speeds=(0.01, 0.02, 0.04), c=0.15, seed=0):
    """x tracks W only in the r bin around 1.5 nm (per-speed rho ~0.4); noise elsewhere."""
    rng = np.random.default_rng(seed)
    rows = []
    r = np.linspace(0.8, 2.0, 49)                 # 0.025 nm spacing, ~2 frames per bin
    hot = np.round(r / 0.05) == 30
    for v in speeds:
        for p in range(n_pull):
            h = rng.normal()
            W = np.cumsum(np.ones(len(r))) + rng.normal(scale=0.5, size=len(r)) + c * h * hot
            x = np.where(hot, h, 0.0) + rng.normal(scale=0.3, size=len(r))
            rows.append(pd.DataFrame({"r": r, "W": W, "x": x, "pull": f"{v}_{p}", "speed": v}))
    return pd.concat(rows, ignore_index=True)


def test_hidden_variable_pools_a_moderate_effect_across_speeds():
    df = _one_bin_pulls()
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=500, rng=np.random.default_rng(6))
    assert p < 0.05 and rho > 0


def test_hidden_variable_speed_without_usable_bins_abstains():
    df = _pulls(planted=True)
    few = _pulls(planted=True, speeds=(0.04,), n_pull=10, seed=7)
    df = pd.concat([df, few], ignore_index=True)
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(8))
    assert p < 0.05 and rho > 0


def _r_proxy_pulls(n_pull=45, speeds=(0.01, 0.02, 0.04), seed=0):
    """Pulls with per-pull r jitter; W rises steeply with r; x is a noisy copy of r only."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in speeds:
        for p in range(n_pull):
            r = np.linspace(0.8, 2.0, 49) + rng.uniform(0, 0.05)
            W = 100.0 * r + rng.normal(scale=0.5, size=len(r))
            x = r + rng.normal(scale=0.005, size=len(r))
            rows.append(pd.DataFrame({"r": r, "W": W, "x": x, "pull": f"{v}_{p}", "speed": v}))
    return pd.concat(rows, ignore_index=True)


def test_hidden_variable_ignores_a_feature_that_only_tracks_r():
    df = _r_proxy_pulls()
    p, _ = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                             df.speed.values, n_perm=200, rng=np.random.default_rng(9))
    assert p >= 0.05


def test_select_caps_kept_features_and_keeps_f3():
    from autopath.metadynamics.learned.select import MAX_KEPT, select
    rng = np.random.default_rng(10)
    n, k = 400, 40
    labels = np.array(["bound"] * 150 + ["vestibule"] * 150 + ["bulk"] * 100, dtype=object)
    shift = (labels == "bound").astype(float)
    V = shift[:, None] * rng.uniform(1, 3, k) + rng.normal(size=(n, k))
    fams = ["F3"] + ["F7"] * (k - 1)
    frames = pd.DataFrame({"W": rng.normal(size=n), "r": np.linspace(0.8, 2.0, n),
                           "pull": np.arange(n) % 10, "speed": 0.01})
    out = select(V, [f"f{i}" for i in range(k)], fams, frames, labels, n_perm=10,
                 rng=np.random.default_rng(0))
    assert (out["fisher"][1:] >= 0.5).sum() > MAX_KEPT
    assert len(out["kept"]) <= MAX_KEPT == 30 and 0 in out["kept"]


def test_select_keeps_fisher_feature_and_drops_work_only_feature():
    from autopath.metadynamics.learned.select import select
    df = _pulls(planted=True)
    r = df.r.values
    labels = np.where(r <= 1.0, "bound", np.where(r >= 1.8, "bulk", "vestibule")).astype(object)
    rng = np.random.default_rng(1)
    n = len(df)
    V = np.c_[rng.normal(size=n), 3.0 * (labels == "bound") + rng.normal(size=n), df.x.values]
    out = select(V, ["F3_com", "F7_cls", "F7_work"], ["F3", "F7", "F7"], df, labels, n_perm=200,
                 rng=np.random.default_rng(0))
    assert out["fisher"][1] > 0.5 and out["fisher"][2] < 0.5
    assert out["kept"] == [0, 1]
    assert out["p_hidden"][2] < 0.05
    off = select(V, ["a", "b", "c"], ["F3", "F7", "F7"], df, labels, hidden=False)
    assert off["kept"] == [0, 1]
    assert np.isnan(off["p_hidden"]).all() and np.isnan(off["rho_hidden"]).all()


from autopath.metadynamics.learned.reduce import HLDA, lda_reference


def _three_class(seed=0):
    rng = np.random.default_rng(seed)
    means = {"bound": [0, 0, 0], "vestibule": [2, 0, 0], "bulk": [4, 3, 0]}
    X, y = [], []
    for k, m in means.items():
        X.append(rng.normal(size=(300, 3)) * [0.5, 0.5, 2.0] + m)
        y += [k] * 300
    return np.vstack(X), np.array(y)


def test_hlda_two_class_matches_closed_form():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 2)) * [1, 0.3]
    B = rng.normal(size=(400, 2)) * [1, 0.3] + [0.5, 1.0]
    X, y = np.vstack([A, B]), np.array(["bound"] * 400 + ["bulk"] * 400)
    w = HLDA(n_components=1, ridge=1e-9).fit(X, y, ["bound", "bulk"])[:, 0]
    ref = (np.linalg.inv(np.cov(A.T)) + np.linalg.inv(np.cov(B.T))) @ (B.mean(0) - A.mean(0))
    assert abs(w @ (ref / np.linalg.norm(ref))) == pytest.approx(1.0, abs=1e-6)


def test_hlda_ignores_the_noise_axis_and_signs_bound_low():
    X, y = _three_class()
    W = HLDA(n_components=2).fit(X, y, ["bound", "vestibule", "bulk"])
    assert np.all(np.abs(W[2]) < 0.1)                 # 3rd feature carries no class information
    s = X @ W[:, 0]
    assert s[y == "bound"].mean() < s[y == "bulk"].mean()


def test_hlda_survives_a_feature_constant_in_one_class():
    X, y = _three_class()
    rng = np.random.default_rng(1)
    z = rng.normal(size=len(y))
    z[y == "bulk"] = 0.3 + 1e-3 * rng.normal(size=(y == "bulk").sum())
    X = np.c_[X, z]
    X = (X - X.mean(0)) / X.std(0)                    # pooled standardisation
    W = HLDA(n_components=2).fit(X, y, ["bound", "vestibule", "bulk"])
    assert np.all(np.isfinite(W))
    assert abs(W[3, 0]) < 0.5                         # near-constant bulk feature must not hijack CV1
    s = X @ W[:, 0]
    assert s[y == "bulk"].mean() - s[y == "bound"].mean() > s.std()


def test_lda_reference_shape_and_sign():
    X, y = _three_class()
    W = lda_reference(X, y, 2)
    s = X @ W[:, 0]
    assert W.shape == (3, 2) and s[y == "bound"].mean() < s[y == "bulk"].mean()


import os

TRAJ = "examples/04_Full-Pipeline/3ptb/sMD/trajectories"
PDB3 = "examples/04_Full-Pipeline/3ptb/system.pdb"


def _3ptb_args():
    import MDAnalysis as mda
    u = mda.Universe(PDB3)
    lig = list(u.select_atoms("resname UNK").indices)
    pocket = list(u.select_atoms("protein and around 6 resname UNK").residues.atoms.indices)
    return (PDB3, "examples/data/3ptb.sdf", lig, pocket, TRAJ)


@pytest.fixture(scope="module")
def learned_3ptb(tmp_path_factory):
    from autopath.metadynamics.learned.pipeline import learn_cv
    out = str(tmp_path_factory.mktemp("learned3ptb"))
    path = learn_cv(*_3ptb_args(), out, stride=4, n_perm=50, hidden=False)
    return path, out


needs_traj = pytest.mark.skipif(not os.path.isdir(TRAJ), reason="3ptb sMD data not on disk")


@pytest.mark.learned_slow
@needs_traj
def test_learn_cv_3ptb_keeps_an_asp189_or_pocket_water_term(learned_3ptb):
    path, out = learned_3ptb
    d = np.load(path, allow_pickle=False)
    kept_names = [str(n) for n in d["names"][d["kept"]]]
    assert "F3_com" in kept_names
    assert any(n.startswith("F5") or (n.startswith("F4") and n.endswith("_172"))
               or (n.startswith("F1") and n.endswith("_172")) for n in kept_names), kept_names
    assert d["weights"].shape[0] == len(d["kept"])
    assert float(d["monotonic_rho"]) > 0.8 and np.isfinite(float(d["f3_acc"]))
    assert np.isnan(d["p_hidden"]).all() and np.isnan(d["rho_hidden"]).all()
    assert os.path.exists(os.path.join(out, "report.png"))


@pytest.mark.learned_slow
@needs_traj
def test_learned_cv_spec_builds_and_evaluates(learned_3ptb):
    import openmm as mm
    from autopath.metadynamics.learned.forces import learned_cv_spec
    path, _ = learned_3ptb
    system = mm.XmlSerializer.deserialize(open("examples/04_Full-Pipeline/3ptb/system.xml").read())
    specs = learned_cv_spec(path, system, n_components=1)
    assert len(specs) == 1 and specs[0].name == "learned1"
    cv = specs[0].resolve(None, system.getNumParticles(), None).cv
    assert cv is not None and specs[0].grid_min < specs[0].grid_max


@pytest.mark.learned_slow
@needs_traj
def test_learn_cv_second_call_reuses_feature_cache(learned_3ptb, tmp_path, caplog):
    import logging
    import shutil
    from autopath.metadynamics.learned.pipeline import learn_cv
    path, out = learned_3ptb
    work = str(tmp_path / "again")
    os.makedirs(work)
    shutil.copy(os.path.join(out, "features.npz"), work)
    with caplog.at_level(logging.INFO, logger="autopath.metadynamics.learned"):
        path2 = learn_cv(*_3ptb_args(), work, stride=4, n_perm=50, hidden=False)
    assert "reusing feature cache" in caplog.text
    a, b = np.load(path, allow_pickle=False), np.load(path2, allow_pickle=False)
    assert list(a["kept"]) == list(b["kept"])


def _fake_pull(d, name="sMD_replica-1_v0.01_forward"):
    dat, dcd = str(d / f"{name}.dat"), str(d / f"{name}_aligned.dcd")
    for f in (dat, dcd):
        open(f, "w").close()
    return dat, dcd


def test_cache_key_and_validity(tmp_path):
    from autopath.metadynamics.learned.pipeline import _cache_key, _cache_valid
    pull = _fake_pull(tmp_path)
    os.utime(pull[0], (1000, 1000))
    os.utime(pull[1], (1000, 1000))
    (tmp_path / "t.pdb").write_text("pdb")
    (tmp_path / "l.sdf").write_text("sdf")
    args = ([pull], 4, "forward", str(tmp_path / "t.pdb"), str(tmp_path / "l.sdf"), [3, 1], [5, 2])
    key = _cache_key(*args, code_hash="a")
    assert key == _cache_key(*args[:5], [1, 3], [2, 5], code_hash="a")  # index order irrelevant
    assert key != _cache_key(*args, code_hash="b")                      # feature code changed
    assert _cache_key(*args) == _cache_key(*args)                       # default hash is stable
    (tmp_path / "l.sdf").write_text("other")
    assert key != _cache_key(*args, code_hash="a")                      # sdf content changed
    (tmp_path / "l.sdf").write_text("sdf")
    (tmp_path / "t.pdb").write_text("other")
    assert key != _cache_key(*args, code_hash="a")                      # pdb content changed
    (tmp_path / "t.pdb").write_text("pdb")
    assert key == _cache_key(*args, code_hash="a")
    for alt in (_cache_key(args[0], 2, *args[2:], code_hash="a"),
                _cache_key(*args[:5], [1], [2, 5], code_hash="a"),
                _cache_key(*args[:2], "reverse", *args[3:], code_hash="a")):
        assert alt != key
    npz = str(tmp_path / "features.npz")
    assert not _cache_valid(npz, key)                                   # missing
    np.savez(npz, V=np.zeros(1), cache_key=np.array(key))
    assert _cache_valid(npz, key)
    assert not _cache_valid(npz, key + " ")                             # key mismatch
    os.utime(pull[1], (os.path.getmtime(npz) + 10,) * 2)                # DCD touched -> key differs
    assert not _cache_valid(npz, _cache_key(*args, code_hash="a"))
    np.savez(npz, V=np.zeros(1))                                        # legacy npz without key
    assert not _cache_valid(npz, key)


def test_standardise_guards_zero_variance_columns():
    from autopath.metadynamics.learned.pipeline import _standardise
    V = np.c_[np.arange(5.0), np.zeros(5), np.ones(5)]
    mu, sd = _standardise(V, np.array([0, 1]), ["F1", "F3", "F5"])
    assert sd.tolist() == [np.arange(5.0).std(), 1.0]
    with pytest.raises(ValueError, match="zero variance"):
        _standardise(V, np.array([0, 2]), ["F1", "F3", "F5"])
