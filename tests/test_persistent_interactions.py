"""get_persistent_interactions: aggregation, cutoff units, stride and fingerprint reuse.

ProLIF's Fingerprint is stubbed so the method's own logic runs for real on
deterministic occupancies; nothing else is mocked.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from autopath import ap_PLIP
from autopath.ap_PLIP import ProteinLigandAnalyzer


class _Selection:
    n_atoms = 1


class _Universe:
    """Duck-type of mda.Universe: a sliceable trajectory and select_atoms()."""

    def __init__(self, n_frames):
        self.trajectory = list(range(n_frames))

    def select_atoms(self, sel):
        return _Selection()


def _make_fingerprint_stub(occupancies):
    """Stub Fingerprint class. ``occupancies`` maps protein residue -> set of frames."""

    class _Fingerprint:
        instances = []

        def __init__(self, interactions=None):
            self.interactions = interactions
            self.frames = None
            self.from_cache = False
            _Fingerprint.instances.append(self)

        @classmethod
        def from_pickle(cls, path):
            fp = cls(interactions=["CACHED"])
            fp.from_cache = True
            fp.frames = [0]
            return fp

        def run(self, traj, lig, prot, n_jobs=1):
            self.frames = list(traj)
            return self

        def to_pickle(self, path):
            pass

        def plot_barcode(self):
            pass

        def to_dataframe(self):
            cols = pd.MultiIndex.from_tuples(
                [("LIG1", res, "HBond") for res in occupancies],
                names=["ligand", "protein", "interaction"],
            )
            data = [[f in occupancies[res] for res in occupancies] for f in self.frames]
            return pd.DataFrame(data, columns=cols)

    return _Fingerprint


def _analyzer(outdir, replicas):
    ana = ProteinLigandAnalyzer.__new__(ProteinLigandAnalyzer)
    ana.replicas = replicas
    ana.outdir = str(outdir)
    ana.protein_mda_selection = "protein"
    ana.ligand_mda_selection = "resname UNK"
    return ana


def test_every_replica_is_returned_not_just_the_last(tmp_path, monkeypatch):
    per_rep = {"repA": {"TYR10.A": set(range(10))},
               "repB": {"TRP20.A": set(range(10))}}

    class _Dispatch:
        """Hands each replica its own stub class, keyed by call order."""

        order = list(per_rep)
        calls = []

        def __new__(cls, interactions=None):
            name = cls.order[len(cls.calls)]
            cls.calls.append(name)
            return _make_fingerprint_stub(per_rep[name])(interactions=interactions)

    monkeypatch.setattr(ap_PLIP, "Fingerprint", _Dispatch)
    ana = _analyzer(tmp_path, {n: _Universe(10) for n in per_rep})

    resids, by_res, by_type = ana.get_persistent_interactions()

    assert resids == [10, 20]
    assert set(by_res["replica"]) == {"repA", "repB"}
    assert set(by_type["replica"]) == {"repA", "repB"}
    assert [i[0] for i in by_res.index] == ["TYR10.A", "TRP20.A"]


def test_frequency_cutoff_is_a_fraction(tmp_path, monkeypatch):
    # TYR10 present in 6/10 frames, TRP20 in 4/10
    occ = {"TYR10.A": set(range(6)), "TRP20.A": set(range(4))}
    monkeypatch.setattr(ap_PLIP, "Fingerprint", _make_fingerprint_stub(occ))
    ana = _analyzer(tmp_path, {"rep1": _Universe(10)})

    resids, by_res, _ = ana.get_persistent_interactions(frequency_cutoff=0.5)

    assert resids == [10]
    assert sorted(by_res["%"].round(6)) == [40.0, 60.0]


def test_stride_changes_the_frames_analysed(tmp_path, monkeypatch):
    # present on even frames only: 50% unstrided, 100% when every 2nd frame is taken
    occ = {"TYR10.A": {f for f in range(10) if f % 2 == 0}}

    def run(stride):
        stub = _make_fingerprint_stub(occ)
        monkeypatch.setattr(ap_PLIP, "Fingerprint", stub)
        outdir = tmp_path / f"s{stride}"
        outdir.mkdir()
        ana = _analyzer(outdir, {"rep1": _Universe(10)})
        _, by_res, _ = ana.get_persistent_interactions(stride=stride)
        return stub.instances[-1].frames, by_res["%"].iloc[0]

    frames1, pct1 = run(1)
    frames2, pct2 = run(2)

    assert len(frames1) == 10 and pct1 == pytest.approx(50.0)
    assert frames2 == [0, 2, 4, 6, 8] and pct2 == pytest.approx(100.0)

    with pytest.raises(ValueError):
        _analyzer(tmp_path, {}).get_persistent_interactions(stride=0)


def test_cached_fingerprint_does_not_leak_into_the_next_replica(tmp_path, monkeypatch):
    occ = {"TYR10.A": set(range(10))}
    stub = _make_fingerprint_stub(occ)
    monkeypatch.setattr(ap_PLIP, "Fingerprint", stub)
    (tmp_path / "FP_repA.pkl").write_text("")      # repA is cached, repB is not
    ana = _analyzer(tmp_path, {"repA": _Universe(10), "repB": _Universe(10)})

    ana.get_persistent_interactions(fp_interactions=["HBDonor"])

    computed = [fp for fp in stub.instances if not fp.from_cache]
    assert [fp.interactions for fp in computed] == [["HBDonor"]]
