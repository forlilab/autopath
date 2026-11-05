import logging

from autopath.utils import build_residue_mapping


def _ca_pdb(path, residues, start=1):
    lines = []
    for i, resn in enumerate(residues):
        rid = start + i
        lines.append(
            f"ATOM  {i + 1:5d}  CA  {resn} A{rid:4d}    "
            f"{3.8 * i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_mapping_warns_on_count_and_name_mismatch(tmp_path, caplog):
    orig = _ca_pdb(tmp_path / "orig.pdb", ["ALA", "MSE", "GLY"], start=10)
    sys_ = _ca_pdb(tmp_path / "sys.pdb", ["ALA", "MET", "GLY", "SER"], start=1)
    with caplog.at_level(logging.WARNING, logger="autopath.utils"):
        mapping = build_residue_mapping(orig, sys_)
    assert mapping == {10: 1, 11: 2, 12: 3}
    text = caplog.text
    assert "CA count mismatch" in text
    assert "Residue name mismatches" in text
