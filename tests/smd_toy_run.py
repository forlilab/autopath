"""Shared helper: toy pulls pushed through the real sMD analysis."""
from pathlib import Path

import pandas as pd

from autopath.pulling import SMDData
from autopath.pulling.AnalysisSMD import SMDAnalysis

from .hs_toys import double_well, simulate_pulls, write_dat


def toy_smd(root: Path, speeds=(0.01, 0.02), n_pulls=12) -> SMDData:
    parts = []
    for i, speed in enumerate(speeds):
        r = simulate_pulls(*double_well(), n_pulls=n_pulls, speed=speed, length_nm=0.8, seed=i + 1)
        r["trajname"] = r["trajname"].str.replace("replica-", f"replica-{i + 1}0")
        parts.append(r)
    rows = pd.concat(parts, ignore_index=True)
    rows.attrs.update(parts[0].attrs)
    files = write_dat(rows, Path(root) / "pulls")
    return SMDData(files, temperature=300.0, completion_threshold_nm=0.0)


def analysed_dir(root: Path, speeds=(0.01, 0.02), path_model="dtw", n_paths=2,
                 estimators=("cumulant", "jarzynski", "force"), **analysis_kw) -> Path:
    """Run the analysis on 12 toy pulls per speed and return its output dir."""
    out = Path(root) / "out"
    smd = toy_smd(root, speeds)
    SMDAnalysis(do_plots=False, outdir=str(out), path_model=path_model, n_paths=n_paths,
                estimators=list(estimators), **analysis_kw).run(smd, cluster_to_boundary=False)
    return out
