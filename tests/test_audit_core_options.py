"""Orchestrator options: unsupported ones fail loudly, per-ligand output dirs don't nest."""
import pytest

import autopath.autopath_core as core


class _Stop(Exception):
    pass


@pytest.mark.parametrize("kw", [{"sMD_time": 5}, {"sMD_steps_per_move": 100}, {"VS_mode": True}])
def test_unsupported_options_raise(tmp_path, kw):
    rec = tmp_path / "rec.pdb"
    rec.write_text("END\n")
    with pytest.raises(ValueError, match="not supported"):
        core.AutoPath(pdb_path=str(rec), do_fix_pdb=False, **kw)


def test_smd_outdir_does_not_nest_across_ligands(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdb"
    rec.write_text("END\n")
    ap = core.AutoPath(pdb_path=str(rec), do_fix_pdb=False, sMD_outdir="sMD")

    def stop(*a, **k):
        raise _Stop          # halts run() right after the output dir is set
    monkeypatch.setattr(core.os, "makedirs", stop)
    seen = []
    for lig in ("a.sdf", "b.sdf", "c.sdf"):
        with pytest.raises(_Stop):
            ap.run(ligand_file=lig)
        seen.append(ap.sMD_outdir)
    assert seen == ["a/sMD", "b/sMD", "c/sMD"]


def test_missing_aligned_traj_raises(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdb"
    rec.write_text("END\n")
    monkeypatch.chdir(tmp_path)
    ap = core.AutoPath(pdb_path=str(rec), do_fix_pdb=False, run_preparation=False,
                       run_equilibration=False)
    monkeypatch.setattr(core, "load_system", lambda p: None)
    monkeypatch.setattr(core, "PDBFile", lambda p: type("P", (), {"topology": None})())
    with pytest.raises(FileNotFoundError, match="_aligned.dcd"):
        ap.run(ligand_file="a.sdf")
