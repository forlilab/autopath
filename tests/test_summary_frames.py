import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from autopath.metadynamics.summary.analysis import _pool, bands, log_weights
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.frames import basin_frames, extract_fe_frames, fes_minima, write_aligned_pdb
from ._summary_fixtures import make_run, tiny_system, write_tiny_md

CVS = ("funnel_z", "learned_sw")


def _models(path):
    """Positions and names of each MODEL block of a PDB."""
    out, cur = [], None
    for ln in open(path):
        if ln.startswith("MODEL"):
            cur = []
        elif ln.startswith(("ATOM", "HETATM")) and cur is not None:
            cur.append((ln[17:20].strip(), ln[12:16].strip(), float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        elif ln.startswith("ENDMDL"):
            out.append(cur)
            cur = None
    return out


# ---------------------------------------------------------------- fes_minima

def _dw1(x, a=0.5, b=2.0, d=1.0):
    return 3.0 * ((x - a) * (x - b)) ** 2 / ((b - a) / 2) ** 4 + d * (x > 1.25)


def test_minima_1d_double_well():
    x = np.linspace(0, 2.5, 251)
    F = _dw1(x)
    F -= F.min()
    m = fes_minima(F, [x], size=5, depth_kcal=0.5, min_separation=0.1)
    assert len(m) == 2
    assert m[0]["coords"][0] == pytest.approx(0.5, abs=0.02)
    assert m[1]["coords"][0] == pytest.approx(2.0, abs=0.02)
    assert m[0]["F"] == pytest.approx(0.0, abs=1e-9) and m[1]["F"] == pytest.approx(1.0, abs=0.05)
    assert fes_minima(F, [x], size=5, f_cut_kcal=0.5)[0]["coords"][0] == pytest.approx(0.5, abs=0.02)
    assert len(fes_minima(F, [x], size=5, f_cut_kcal=0.5)) == 1


def test_minima_2d_double_well_and_masks():
    x, y = np.linspace(-2, 2, 81), np.linspace(-1.5, 1.5, 61)
    X, Y = np.meshgrid(x, y, indexing="ij")
    F = 4.0 * (X ** 2 - 1) ** 2 + 6.0 * Y ** 2 + 0.8 * (X > 0)
    F -= F.min()
    m = fes_minima(F, [x, y], size=5)
    assert len(m) == 2
    assert m[0]["coords"] == pytest.approx((-1.0, 0.0), abs=0.06)
    assert m[1]["coords"] == pytest.approx((1.0, 0.0), abs=0.06)
    assert m[0]["index"] == (int(np.argmin(np.abs(x + 1))), 30)
    Fm = F.copy()
    Fm[:5] = np.nan
    assert len(fes_minima(Fm, [x, y], size=5)) == 2
    assert len(fes_minima(F, [x, y], size=5, f_cut_kcal=0.5)) == 1


def test_minima_merge_shallow_and_close():
    x = np.linspace(0, 3, 301)
    F = 2.0 * (x - 1.5) ** 2 + 0.2 * np.cos(2 * np.pi * x / 0.5)   # ripples: barriers < 0.5
    F -= F.min()
    assert len(fes_minima(F, [x], size=5, min_separation=0.01, depth_kcal=0.5, f_cut_kcal=10)) == 1
    many = fes_minima(F, [x], size=5, min_separation=0.01, depth_kcal=0.0, f_cut_kcal=10)
    assert len(many) > 1
    assert len(fes_minima(F, [x], size=5, min_separation=2.0, depth_kcal=0.0, f_cut_kcal=10)) == 1
    with pytest.raises(ValueError, match="not supported"):
        fes_minima(np.zeros((3, 3, 3)), [np.arange(3)] * 3)


# ---------------------------------------------------------------- basin_frames

@pytest.fixture(scope="module")
def run2(tmp_path_factory):
    return load_run(make_run(str(tmp_path_factory.mktemp("bf") / "r"), cvs=CVS, monitors=(), extend=True))


def _expected_weights(run, method, burn=0.2):
    vals, lws, keys = [], [], []
    for w in run.reweighted():
        lw = log_weights(run, w, method)
        k = np.arange(len(lw)) >= int(burn * len(lw))
        vals.append(np.flatnonzero(k)); lws.append(lw[k]); keys += [(w.rid, float(t)) for t in w.t[k]]
    _, W = _pool(vals, lws)
    return dict(zip(keys, W))


def test_basin_frames_ellipse_weights_seed(run2):
    c = (1.0, 1.0)
    sig = np.array([x.sigma for x in run2.cvs])
    for method in ("rw_ct", "rw_final"):
        bf = basin_frames(run2, run2.reweighted(), method, {"coords": c}, radius_hills=1.5, n_frames=8, seed=3)
        assert bf["frames"] and bf["n_available"] >= len(bf["frames"]) and bf["ess"] > 0
        exp = _expected_weights(run2, method)
        for f in bf["frames"]:
            v = np.array([f["cvs"][n] for n in CVS])
            assert np.sum(((v - c) / (1.5 * sig)) ** 2) <= 1.0
            assert f["frame"] >= 0
            assert f["weight"] == pytest.approx(exp[(f["rid"], f["time_ns"])], rel=1e-9)
            w = next(x for x in run2.walkers if x.rid == f["rid"])
            r = int(np.flatnonzero(np.isclose(w.t, f["time_ns"]))[0])
            assert w.record_frame[r] == f["frame"] and w.segment_of_record[r] == f["segment"]
            assert f["dcd"] == w.dcd_files[f["segment"]]
            assert all(abs(w.cvs[n][r] - f["cvs"][n]) < 1e-12 for n in CVS)
        again = basin_frames(run2, run2.reweighted(), method, {"coords": c}, radius_hills=1.5, n_frames=8, seed=3)
        assert [(f["rid"], f["time_ns"]) for f in again["frames"]] == [(f["rid"], f["time_ns"]) for f in bf["frames"]]
    a = basin_frames(run2, run2.reweighted(), "bias", {"coords": c}, radius_hills=1.5, n_frames=8, seed=3)
    b = basin_frames(run2, run2.reweighted(), "rw_final", {"coords": c}, radius_hills=1.5, n_frames=8, seed=3)
    assert a["weight_method"] == "rw_final" and a["frames"] == b["frames"]
    picks = [basin_frames(run2, run2.reweighted(), "rw_ct", {"coords": c}, radius_hills=3.0, n_frames=3, seed=k)
             for k in range(4)]
    assert picks[0]["n_available"] > 3
    assert len({tuple((f["rid"], f["time_ns"]) for f in p["frames"]) for p in picks}) > 1


def test_basin_frames_few_and_none(run2):
    all_ = basin_frames(run2, run2.reweighted(), "rw_ct", {"coords": (1.0, 1.0)}, radius_hills=0.3, n_frames=10 ** 6)
    assert len(all_["frames"]) == all_["n_available"] > 0
    none = basin_frames(run2, run2.reweighted(), "rw_ct", {"coords": (50.0, 50.0)})
    assert none["frames"] == [] and none["n_available"] == 0 and "no DCD frame" in none["reason"]


# ---------------------------------------------------------------- write_aligned_pdb

def test_write_aligned_pdb_images_and_fits(tmp_path):
    pdb, dcd = write_tiny_md(str(tmp_path / "t.pdb"), str(tmp_path / "t.dcd"), 6, seed=1)
    import MDAnalysis as mda
    raw = mda.Universe(pdb, dcd)
    lig = raw.select_atoms("resname UNK")
    broken = 0
    for ts in raw.trajectory:
        broken += np.max(np.linalg.norm(np.diff(lig.positions, axis=0), axis=1)) > 3.0
        broken += np.ptp(raw.select_atoms("protein").positions, axis=0).max() > 20.0
    assert broken > 0                                   # the DCD really is split across images
    fr = [dict(dcd=dcd, frame=i, pdb=pdb, ref_pdb=pdb) for i in range(6)]
    out = str(tmp_path / "o.pdb")
    res = write_aligned_pdb(fr, out)
    assert res["n_models"] == 6 and max(res["rmsd"]) < 1e-3
    from MDAnalysis.analysis.rms import rmsd
    rca = tiny_system().select_atoms("name CA").positions
    ms = _models(out)
    assert len(ms) == 6
    ca0 = np.array([a[2:] for a in ms[0] if a[1] == "CA"])
    for m in ms:
        ca = np.array([a[2:] for a in m if a[1] == "CA"])
        assert rmsd(ca, rca, center=True, superposition=True) < 0.01
        assert np.abs(ca - ca0).max() < 0.01                # every model in one frame
        lg = np.array([a[2:] for a in m if a[0] == "UNK"])
        assert len(lg) == 3 and np.max(np.linalg.norm(np.diff(lg, axis=0), axis=1)) < 1.7
        wat = np.array([a[2:] for a in m if a[0] == "HOH"])
        assert len(wat) % 3 == 0 and len(wat)
        d = np.linalg.norm(wat[::3, None] - lg[None], axis=2).min(1)
        assert d.min() < 5.0 + 1.0


# ---------------------------------------------------------------- extract_fe_frames

PAT = re.compile(r"^min\d{2}_funnel_z-?\d+\.\d{2}nm_learned_sw-?\d+\.\d{2}_F(-?\d+\.\d{2}kcal|na)\.pdb$")


def _csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_extract_shared_with_segments(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=CVS, monitors=(), extend=True, dcd=True))
    s = extract_fe_frames(run, bands(run), str(tmp_path / "out"), n_frames=6)
    assert s["pair"] == list(CVS)
    segs = set()
    # extension segments restart wall_s, so balanced reweighting is refused with a reason
    assert any(x.startswith("rw_balanced:") and "not supported" in x for x in s["skipped"])
    for m in ("rw_ct", "rw_final", "bias"):
        d = tmp_path / "out" / "fe_frames" / m
        assert (d / "minima.png").exists() and (d / "minima.csv").exists()
        mins, frs = _csv(d / "minima.csv"), _csv(d / "frames.csv")
        assert mins and len(mins) <= 6
        pdbs = sorted(p for p in os.listdir(d) if p.endswith(".pdb"))
        assert pdbs and all(PAT.match(p) for p in pdbs)
        for r in mins:
            if r["pdb"]:
                assert len(_models(d / r["pdb"])) == int(r["n_frames"])
                assert int(r["n_frames"]) == sum(f["k"] == r["k"] for f in frs)
        for f in frs:
            assert f["dcd"] == f"WTMetaD_{f['segment_rid']}.dcd"
            assert f["segment_rid"] == (f["rid"] if f["segment"] == "0" else f"{f['rid']}_x2")
            segs.add(f["segment"])
        assert [float(r["F_min"]) for r in mins] == sorted(float(r["F_min"]) for r in mins)
    assert segs == {"0", "1"}


def test_extract_isolated_and_1d_and_points(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", cvs=CVS, monitors=(), dcd=True))
    s = extract_fe_frames(run, bands(run), str(tmp_path / "out"), estimators=["rw_ct", "bias"], n_frames=3)
    for m in ("rw_ct", "bias"):
        assert set(s["estimators"][m]) == {"w0", "w1", "png"}
        for rid in ("w0", "w1"):
            d = tmp_path / "out" / "fe_frames" / m / rid
            assert (d / "minima.csv").exists()
            assert {f["rid"] for f in _csv(d / "frames.csv")} == {rid}
        assert not any(p.endswith(".pdb") for p in os.listdir(tmp_path / "out" / "fe_frames" / m))
    r1 = load_run(make_run(str(tmp_path / "r1"), cvs=("funnel_z",), monitors=(), dcd=True))
    extract_fe_frames(r1, bands(r1), str(tmp_path / "o1"), estimators=["rw_ct", "bias"], n_frames=2)
    for m in ("rw_ct", "bias"):
        p = [x for x in os.listdir(tmp_path / "o1" / "fe_frames" / m) if x.endswith(".pdb")]
        assert p and all(re.match(r"^min\d{2}_funnel_z-?\d+\.\d{2}nm_F", x) and "learned" not in x for x in p)
    s = extract_fe_frames(r1, bands(r1), str(tmp_path / "o2"), estimators=["rw_ct"], at=[[0.5], [1.5]], n_frames=2)
    rows = s["estimators"]["rw_ct"]["pooled"]["minima"]
    assert [r["label"] for r in rows] == ["pt", "pt"] and rows[0]["coords"] == (0.5,)
    assert all(x.startswith("pt") for x in os.listdir(tmp_path / "o2" / "fe_frames" / "rw_ct") if x.endswith(".pdb"))
