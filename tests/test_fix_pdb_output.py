"""do_fix_pdb writes <stem>_fixed<ext> next to the input, atomically, without touching directory names."""
import glob
from types import SimpleNamespace

import autopath.autopath_core as core


def test_fixed_pdb_name_and_no_temp_left(tmp_path, monkeypatch):
    d = tmp_path / "set.pdbs"            # a directory name containing ".pdb"
    d.mkdir()
    rec = d / "rec.pdb"
    rec.write_text("END\n")
    fixed = SimpleNamespace(topology=None, positions=None)
    monkeypatch.setattr(core, "PDBPreprocessor",
                        lambda p: SimpleNamespace(fix=lambda **k: fixed))
    monkeypatch.setattr(core, "save_pdb", lambda top, pos, out: open(out, "w").write("FIXED\n"))

    ap = core.AutoPath(pdb_path=str(rec), do_fix_pdb=True)

    assert ap.protein_file == str(d / "rec_fixed.pdb")
    assert (d / "rec_fixed.pdb").read_text() == "FIXED\n"
    assert glob.glob(f"{d}/*tmp*") == []
