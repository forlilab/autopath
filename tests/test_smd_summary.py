import json

import numpy as np
import pandas as pd
import pytest

import autopath
from autopath.pulling.outputs import jarzynski_convergence, write_summary

from .smd_toy_run import analysed_dir

KT = 0.008314462618 * 300.0


def _tr(works, n_steps=5, speed=0.01, drop=None):
    rows = []
    for i, w in enumerate(works):
        m = n_steps - (drop if i == 0 and drop else 0)
        for s in range(m):
            rows.append(dict(trajname=f"p{i}", speed=speed, step=s, work=w, r_coord=0.1 * s))
    pulls = pd.DataFrame(dict(trajname=[f"p{i}" for i in range(len(works))],
                              pull_index=range(1, len(works) + 1)))
    return pd.DataFrame(rows), pulls


def _row(df, n):
    return df[df.n == n].iloc[0]


def test_jarzynski_equal_work():
    r = _row(jarzynski_convergence(*_tr([5.0] * 5), 300.0), 5)
    assert r.top_pull_weight == pytest.approx(0.2)
    assert r.ess_over_n == pytest.approx(1)
    assert r.tail_position == pytest.approx(0, abs=1e-9)
    assert r.beta_sigma_W == 0


def test_jarzynski_one_dominant_pull():
    r = _row(jarzynski_convergence(*_tr([0.0] + [100 * KT] * 4), 300.0), 5)
    assert r.top_pull_weight > 0.999
    assert r.ess_over_n == pytest.approx(0.2, abs=1e-3)
    assert r.tail_position == pytest.approx(1, abs=1e-3)


def test_jarzynski_order_follows_pull_index():
    tr, pu = _tr([0.0, 3.0, 7.0, 1.0])
    a = jarzynski_convergence(tr, pu, 300.0)
    b = jarzynski_convergence(tr, pu.assign(pull_index=pu.pull_index[::-1].to_numpy()), 300.0)
    assert not np.isclose(_row(a, 2).beta_sigma_W, _row(b, 2).beta_sigma_W)
    assert _row(a, 4).to_dict() == pytest.approx(_row(b, 4).to_dict())


def test_jarzynski_convergence_uneven_and_single_pull():
    tr, pu = _tr([0.0, 1.0, 2.0], n_steps=20, drop=10)
    tr2, pu2 = _tr([0.0], speed=0.02)
    tr2, pu2 = tr2.assign(trajname="q0"), pu2.assign(trajname="q0")
    out = jarzynski_convergence(pd.concat([tr, tr2]), pd.concat([pu, pu2]), 300.0)
    assert set(out.speed) == {0.01}
    short_last = tr[tr.trajname == "p0"].r_coord.max()
    assert out.r_nm.max() <= short_last + 1e-12


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    out = analysed_dir(tmp_path_factory.mktemp("sum"))
    vall = pd.DataFrame(dict(speed=[0.01, 0.01, 0.02], n_replicas=[5, 12, 12],
                             converged=[False, True, False]))
    vall.to_csv(out / "sMD_conv_vALL_metrics.csv", index=False)
    return out


META = dict(autopath_version=autopath.__version__, system="toy", ligand=dict(resname="UNK", heavy_atoms=9),
            temperature_K=300.0, spring_k_kJ_mol_nm2=3765.6, path_model="dtw", n_paths=2,
            analysis_warnings=1)


def test_summary_schema_on_toy_run(toy):
    s = write_summary(str(toy), META)
    assert set(s) == {"autopath_version", "system", "ligand", "temperature_K", "spring_k_kJ_mol_nm2",
                      "path_model", "n_paths", "analysis_warnings", "speeds", "readouts",
                      "jarzynski_diagnostics"}
    assert s["autopath_version"] == autopath.__version__
    sp = s["speeds"]["0.01"]
    assert sp["pulls"] == 12 and sp["stop"] == {"unknown": 12}
    assert sp["converged"] is True and s["speeds"]["0.02"]["converged"] is False
    assert sum(sp["path_weights"].values()) == pytest.approx(1, abs=1e-3)
    ro = {(r["estimator"], r["speed"]): r for r in s["readouts"]}
    assert ro[("force", 0.0)]["dG_kJ_mol"] is not None and ro[("force", 0.0)]["pkoff"] is not None
    assert ro[("cumulant", 0.0)]["dG_kJ_mol"] is None
    txt = (toy / "summary.json").read_text()
    assert "NaN" not in txt
    assert json.loads(txt) == s


def test_summary_refresh_keeps_meta(toy):
    write_summary(str(toy), META)
    pd.DataFrame(dict(speed=[0.01, 0.02], n_replicas=[12, 12], converged=[False, True])
                 ).to_csv(toy / "sMD_conv_vALL_metrics.csv", index=False)
    s = write_summary(str(toy))
    assert {k: s[k] for k in META} == META
    assert s["speeds"]["0.01"]["converged"] is False and s["speeds"]["0.02"]["converged"] is True


def test_summary_single_speed(tmp_path):
    out = analysed_dir(tmp_path, speeds=(0.01,), estimators=("cumulant", "jarzynski"))
    s = write_summary(str(out), META)
    assert "force" not in {r["estimator"] for r in s["readouts"]}
    assert list(s["jarzynski_diagnostics"]) == ["0.01"]
