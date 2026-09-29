"""Regression tests for the milestone bugs found in the final branch review."""
import os
import numpy as np
import pytest
import MDAnalysis as mda

from autopath.utils import _reimaging_ligand
from autopath.clustering import order_milestones_bound_to_unbound


def _lig_universe():
    """One protein CA plus a 2-atom ligand residue (heavy atom + its hydrogen)."""
    u = mda.Universe.empty(3, n_residues=2, atom_resindex=[0, 1, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "C1", "H1"])
    u.add_TopologyAttr("resname", ["ALA", "LIG"])
    u.add_TopologyAttr("resid", [1, 2])
    u.dimensions = [40, 40, 40, 90, 90, 90]
    # ligand sits a box length away from the protein, so re-imaging must fire
    u.atoms.positions = np.array([[0., 0., 0.], [34., 0., 0.], [35., 0., 0.]])
    return u


def test_reimaging_moves_ligand_hydrogens_with_their_heavy_atoms():
    """A heavy-atom selection must not strand the ligand's hydrogens.

    Callers pass "(ligand) and not name H*". Translating only the selected atoms
    left the H a box length behind, giving a 41 A C-H bond in the structure that
    then gets serialized and minimised.
    """
    u = _lig_universe()
    assert _reimaging_ligand(u, "resname LIG and not name H*") is True

    c1, h1 = u.atoms[1].position, u.atoms[2].position
    assert np.linalg.norm(h1 - c1) == pytest.approx(1.0, abs=1e-4)
    # and it really did move, i.e. the test is not passing because nothing happened
    assert c1[0] == pytest.approx(-6.0, abs=1e-4)
    assert h1[0] == pytest.approx(-5.0, abs=1e-4)


def test_reimaging_leaves_protein_untouched():
    u = _lig_universe()
    _reimaging_ligand(u, "resname LIG and not name H*")
    np.testing.assert_allclose(u.atoms[0].position, [0., 0., 0.], atol=1e-4)


def test_reimaging_noop_when_ligand_already_at_minimum_image():
    u = _lig_universe()
    u.atoms.positions = np.array([[0., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
    assert _reimaging_ligand(u, "resname LIG and not name H*") is False
    np.testing.assert_allclose(u.atoms[1].position, [3., 0., 0.], atol=1e-4)


# --- milestone direction anchoring -----------------------------------------

def _write_state_pair(tmp_path, seps):
    """Write one .pdb per requested pocket-ligand separation (in nm).

    Atom 0 is the pocket, atom 1 the ligand; PDB coordinates are Angstrom.
    """
    paths = []
    for i, sep in enumerate(seps):
        p = tmp_path / f"milestone_{i + 1}_frame_{i}.pdb"
        with open(p, "w") as f:
            f.write(f"ATOM      1  C1  POC A   1    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            f.write(f"ATOM      2  C2  LIG A   2    {sep * 10:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            f.write("END\n")
        paths.append(str(p))
    return paths


def test_bound_to_unbound_order_is_left_alone(tmp_path):
    paths = _write_state_pair(tmp_path, [0.4, 0.9, 1.5])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == paths


def test_unbound_to_bound_order_is_reversed(tmp_path):
    """PC1's sign is arbitrary, so the centroid order is as likely to be reversed."""
    paths = _write_state_pair(tmp_path, [1.5, 0.9, 0.4])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == list(reversed(paths))


def test_single_milestone_is_returned_unchanged(tmp_path):
    paths = _write_state_pair(tmp_path, [0.7])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == paths


# --- sigma='auto' ----------------------------------------------------------

def _auto_sigma(com_dists):
    """The sigma='auto' arithmetic from pathCV_cv, against which we assert."""
    steps = np.diff(np.asarray(com_dists, dtype=float))
    gaps = np.abs(steps)
    usable = gaps[gaps > 1e-6]
    if usable.size == 0:
        raise ValueError("no usable spacing")
    return 0.5 * float(usable.min())


def test_auto_sigma_is_positive_for_a_non_monotone_milestone_list():
    """A backwards step must not produce a negative sigma.

    cvpack accepts a negative sigma without complaint, so this failed silently.
    """
    sigma = _auto_sigma([0.40, 1.20, 0.50, 0.90, 1.10])
    assert sigma > 0
    assert sigma == pytest.approx(0.5 * 0.20, abs=1e-9)


def test_auto_sigma_matches_half_the_minimum_gap_when_monotone():
    assert _auto_sigma([0.4, 0.9, 1.5]) == pytest.approx(0.25, abs=1e-9)


def test_auto_sigma_rejects_all_identical_milestones():
    with pytest.raises(ValueError):
        _auto_sigma([0.5, 0.5, 0.5])


def test_pathcv_source_guards_sigma_and_warns_on_non_monotone():
    """Pin the guard in the real code, since the arithmetic above is a copy."""
    import inspect
    from autopath.metadynamics import CV
    src = inspect.getsource(CV.pathCV_cv)
    assert "np.abs(steps)" in src
    assert "cannot auto-tune sigma" in src
    assert "not monotonically increasing" in src


# --- no loadState anywhere -------------------------------------------------

def test_no_module_calls_load_state():
    """loadState restores global parameters by name, so it silently keeps stale
    values on a system whose forces were stripped and re-added."""
    import autopath
    root = os.path.dirname(os.path.abspath(autopath.__file__))
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                for lineno, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    if ".loadState(" in code:
                        offenders.append(f"{os.path.relpath(path, root)}:{lineno}")
    assert offenders == [], f"loadState is forbidden; found at {offenders}"


# --- medoid_info.json lookup ------------------------------------------------

def test_core_looks_for_medoid_info_under_path_analysis():
    """PathModel writes medoid_info.json into path_analysis/.

    Reading it from the analysis dir instead silently fell back to using every
    sMD trajectory rather than the medoids, with only a warning.
    """
    import inspect
    from autopath import autopath_core
    from autopath.pulling import PathModel as pm

    writer = inspect.getsource(pm)
    assert '"path_analysis", "medoid_info.json"' in writer

    reader = inspect.getsource(autopath_core)
    assert 'path_analysis/medoid_info.json' in reader


# --- bias plot glob (isolated-walker mode) ---------------------------------

def test_plot_bias_skips_walker_bias_directories(tmp_path, monkeypatch):
    """With mMD_multiple_walkers=False each walker gets its own bias_<run_id>/
    directory inside out_dir, so a bare `bias_*` glob hands np.load a directory."""
    import numpy as np
    from autopath.metadynamics.Analysis import plot_bias

    # plot_bias takes sys_name from Path(out_dir).parts[0], so out_dir must be
    # relative the way the pipeline passes it.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "3ptb" / "metadynamics_funnel"
    out.mkdir(parents=True)
    # the layout the pipeline actually produces in isolated mode:
    # directories named bias_<run_id>, each holding the real arrays
    for run_id in ("milestone_1_frame_0", "milestone_2_frame_145"):
        d = out / f"bias_{run_id}"
        d.mkdir()
        np.save(d / "bias_12345_7.npy", np.linspace(0.0, 5.0, 25))
    # must not raise IsADirectoryError, and must produce the plot
    plot_bias("3ptb/metadynamics_funnel", x_min=0.0, x_max=1.0, grid_points=25,
              colvar_name="PathCV_progress")
    assert (out / "3ptb_BIAS.png").exists()


def test_plot_bias_returns_quietly_when_no_bias_arrays(tmp_path, monkeypatch):
    from autopath.metadynamics.Analysis import plot_bias
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "sys" / "metadynamics"
    out.mkdir(parents=True)
    plot_bias("sys/metadynamics", x_min=0.0, x_max=1.0, grid_points=25,
              colvar_name="PathCV_progress")
    assert not (out / "sys_BIAS.png").exists()


def test_load_bias_files_finds_isolated_walker_subdirs(tmp_path):
    """Isolated mode hides each walker's bias inside bias_<run_id>/.

    A flat os.listdir of out_dir matches nothing there (the entries are
    directories named bias_<run_id>, not bias_<int>_<int>.npy), so the post-run
    diagnosis got an empty dict and died on np.stack([]).
    """
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis

    out = tmp_path / "metadynamics_funnel"
    out.mkdir()
    (out / "bias_milestone_1_frame_0").mkdir()
    np.save(out / "bias_milestone_1_frame_0" / "bias_874600475_101.npy",
            np.zeros(125))
    (out / "bias_milestone_2_frame_145").mkdir()
    np.save(out / "bias_milestone_2_frame_145" / "bias_525115211_17.npy",
            np.ones(125))

    biases = MetadynamicsAnalysis.load_bias_files(str(out))
    assert len(biases) == 2, f"expected both walkers, got {list(biases)}"
    assert all(b.shape == (125,) for b in biases.values())


def test_load_bias_files_still_reads_flat_shared_bias_layout(tmp_path):
    """Shared-bias mode must keep working: highest index wins per walker."""
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis

    out = tmp_path / "metadynamics"
    out.mkdir()
    np.save(out / "bias_7_1.npy", np.zeros(125))
    np.save(out / "bias_7_9.npy", np.ones(125))
    np.save(out / "bias_8_3.npy", np.full(125, 2.0))

    biases = MetadynamicsAnalysis.load_bias_files(str(out))
    assert sorted(biases) == [7, 8]
    assert biases[7][0] == 1.0, "should keep the highest save index for walker 7"


# --- metadynamics context rebuild + state-xml restart ----------------------

def test_metad_rebuilds_context_instead_of_reinitialising():
    """The funnel and bias forces are added to the System AFTER the Context is
    built. reinitialize(preserveState=True) restores globals by name, so a newly
    added force's default can be overwritten by a stale value and the force
    silently does nothing. equilibration.py and relax_md.py already use
    rebuild_simulation for this; MetadynamicsMD must too.
    """
    import inspect
    from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD

    src = inspect.getsource(MetadynamicsMD.run)
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    assert "reinitialize" not in code
    assert "rebuild_simulation(simulation, system)" in code


def test_metad_attaches_reporters_after_the_rebuild():
    """rebuild_simulation returns a NEW Simulation, so reporters attached before
    it would be silently dropped and the run would write no output."""
    import inspect
    from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD

    src = inspect.getsource(MetadynamicsMD.run)
    assert src.index("rebuild_simulation(simulation, system)") < src.index("add_reporters(")


def test_core_seeds_metadynamics_from_state_xml_not_checkpoint():
    """A checkpoint is only valid for an identical Context, and metadynamics adds
    the funnel and bias forces on top of the relaxed system."""
    import inspect
    from autopath import autopath_core

    src = inspect.getsource(autopath_core)
    assert "_relax_state.xml" in src
    # the metadynamics seeds must not come from a relax checkpoint any more
    assert "_relax_checkpoint.chk" not in src


# --- funnel cylinder radius ------------------------------------------------

def _funnel_universe(n_frames=400, r_bound=0.5, r_cyl=4.0, seed=0):
    """Ligand walks out along +x; radial spread is SMALL while bound and LARGE
    once it is in the cylinder region. That is the real shape of an unbinding
    path, and the case a bound-state-derived radius gets wrong."""
    import numpy as np
    import MDAnalysis as mda
    rng = np.random.default_rng(seed)
    u = mda.Universe.empty(2, n_residues=2, atom_resindex=[0, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "C1"])
    u.add_TopologyAttr("resname", ["ALA", "LIG"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("masses", [12.0, 12.0])  # center_of_mass needs masses
    coords = np.zeros((n_frames, 2, 3))
    z = np.linspace(0.0, 25.0, n_frames)
    half = n_frames // 2
    rad = np.concatenate([rng.normal(0, r_bound, half),
                          rng.normal(0, r_cyl, n_frames - half)])
    coords[:, 1, 0] = z
    coords[:, 1, 1] = rad
    u.load_new(coords, order="fac")
    u.dimensions = [80, 80, 80, 90, 90, 90]
    return u


def test_cylinder_radius_is_derived_from_the_cylinder_region():
    """A hardcoded 2 A wall that the reference path itself violates does not
    guide sampling, it just injects force."""
    from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
    import numpy as np

    u = _funnel_universe()
    p = generate_funnel_parameters_from_trajectory(
        u, host_selection="resname ALA", guest_selection="resname LIG",
        stride=1, verbose=False)

    from openmm import unit as ommunit
    R = p["R_cylinder"].value_in_unit(ommunit.angstrom)
    r = np.asarray(p["radial_distances"])
    z = np.abs(np.asarray(p["axial_distances"]))
    zcc = p["z_cc"].value_in_unit(ommunit.angstrom)
    outside = float((r[z >= zcc] > R).mean())
    # the whole point: the derived wall must not exclude most of the path
    assert outside < 0.25, f"R_cylinder={R:.2f} excludes {100*outside:.0f}% of the path"
    assert R > 2.0, f"derived radius {R:.2f} collapsed to the old hardcoded value"


def test_explicit_cylinder_radius_still_overrides():
    from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
    u = _funnel_universe()
    p = generate_funnel_parameters_from_trajectory(
        u, host_selection="resname ALA", guest_selection="resname LIG",
        R_cylinder_ang=1.5, stride=1, verbose=False)
    from openmm import unit as ommunit
    assert p["R_cylinder"].value_in_unit(ommunit.angstrom) == 1.5


# --- per-path metadynamics grouping ---------------------------------------

def _write_processed_csv(path, rows, with_path_ok=True):
    import pandas as pd
    cols = ["step", "trajname", "path"] + (["path_ok"] if with_path_ok else [])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)


def test_read_trajname_to_path_keeps_only_surviving_rows(tmp_path):
    from autopath.utils import read_trajname_to_path

    _write_processed_csv(
        tmp_path / "sMD_processed_data.csv",
        [[0, "sMD_replica-1_v0.01_forward", "path-0_v0.01", True],
         [1, "sMD_replica-1_v0.01_forward", "path-0_v0.01", True],
         [0, "sMD_replica-2_v0.01_forward", "path-1_v0.01", False]],
    )
    mapping = read_trajname_to_path(str(tmp_path))
    assert mapping == {"sMD_replica-1_v0.01_forward": "path-0_v0.01"}


def test_read_trajname_to_path_treats_missing_path_ok_as_all_kept(tmp_path):
    from autopath.utils import read_trajname_to_path

    _write_processed_csv(
        tmp_path / "sMD_processed_data.csv",
        [[0, "a", "path-0"], [0, "b", "path-1"]],
        with_path_ok=False,
    )
    assert read_trajname_to_path(str(tmp_path)) == {"a": "path-0", "b": "path-1"}


def test_read_trajname_to_path_is_empty_when_the_table_is_missing(tmp_path):
    from autopath.utils import read_trajname_to_path

    assert read_trajname_to_path(str(tmp_path)) == {}


def test_grouping_splits_subdirectory_milestones_by_path(tmp_path):
    """per_path extraction writes one subdir per path; each must become its own
    group so it gets its own path CV and its own funnel."""
    from autopath.autopath_core import group_milestones_by_path

    root = tmp_path / "milestones"
    files = []
    for path_label in ("path-1_v0.02", "path-0_v0.02"):
        (root / path_label).mkdir(parents=True)
        for i in (1, 0):
            f = root / path_label / f"milestone_{path_label}_{i}_frame_{i}.xml"
            f.write_text("")
            files.append(str(f))

    groups = group_milestones_by_path(files, str(root))

    assert [k for k, _ in groups] == ["path-0_v0.02", "path-1_v0.02"]
    assert all(len(v) == 2 for _, v in groups)
    for key, members in groups:
        assert all(os.path.basename(os.path.dirname(m)) == key for m in members)


def test_grouping_keeps_the_flat_layout_as_one_unlabelled_group(tmp_path):
    """all_medoids (the default) writes milestones flat. That layout must stay a
    single group keyed None, or the default pipeline changes its output paths."""
    from autopath.autopath_core import group_milestones_by_path

    root = tmp_path / "milestones"
    root.mkdir()
    files = []
    for i in range(3):
        f = root / f"milestone_{i}_frame_{i}.xml"
        f.write_text("")
        files.append(str(f))

    groups = group_milestones_by_path(files, str(root))

    assert len(groups) == 1
    assert groups[0][0] is None
    assert sorted(groups[0][1]) == sorted(files)


def test_grouping_puts_the_unlabelled_group_first(tmp_path):
    from autopath.autopath_core import group_milestones_by_path

    root = tmp_path / "milestones"
    (root / "path-0").mkdir(parents=True)
    flat = root / "milestone_0_frame_0.xml"
    flat.write_text("")
    nested = root / "path-0" / "milestone_1_frame_1.xml"
    nested.write_text("")

    groups = group_milestones_by_path([str(nested), str(flat)], str(root))
    assert [k for k, _ in groups] == [None, "path-0"]
