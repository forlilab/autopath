"""output_format='none' turns the unbinding-paths figure off without reading any trajectory."""
import pytest

from autopath.pulling.Diagnostics import make_unbinding_paths_visualization
from autopath.pulling.PathModel import DTWPathModel


def test_none_writes_nothing(tmp_path):
    out = tmp_path / "path_analysis"
    res = make_unbinding_paths_visualization(
        paths={"path-0": [("missing.pdb", "missing.dcd")]}, reference_pdb="missing.pdb",
        ligand_select="resname UNK", outdir=str(out), output_format="none")
    assert res is None
    assert not out.exists()


def test_bad_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="'none'"):
        make_unbinding_paths_visualization(paths={}, reference_pdb="x.pdb", ligand_select="all",
                                           outdir=str(tmp_path), output_format="png")


def test_path_model_carries_format():
    assert DTWPathModel(viz_output_format="none").viz_output_format == "none"
