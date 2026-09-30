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


# --- medoid filtering keys on the trajectory, not the stale label ----------

def test_medoid_filter_keeps_a_medoid_whose_label_is_stale():
    """medoid_info.json is written before the quality filters run and can label a
    trajectory differently from the final assignment. Filtering on the LABEL drops
    medoids whose trajectory actually passed QC."""
    from autopath.utils import filter_medoids_by_surviving_trajectories

    medoid_to_path = {"repA": "path-0_v0.02", "repB": "path-1_v0.02"}
    surviving = {"repA": "path-0_v0.02", "repB": "path-0_v0.02"}  # repB really passed

    kept, dropped, disagree = filter_medoids_by_surviving_trajectories(
        medoid_to_path, surviving)

    assert dropped == {}, "a medoid whose trajectory passed QC must not be dropped"
    assert kept == {"repA": "path-0_v0.02", "repB": "path-0_v0.02"}
    assert disagree == {"repB": ("path-1_v0.02", "path-0_v0.02")}


def test_medoid_filter_drops_a_medoid_whose_trajectory_failed_qc():
    from autopath.utils import filter_medoids_by_surviving_trajectories

    kept, dropped, disagree = filter_medoids_by_surviving_trajectories(
        {"repA": "path-0", "repGone": "path-1"}, {"repA": "path-0"})
    assert kept == {"repA": "path-0"}
    assert dropped == {"repGone": "path-1"}
    assert disagree == {}


def test_medoid_filter_is_a_noop_without_analysis_output():
    """An empty surviving map means 'unknown', never 'nothing survived'."""
    from autopath.utils import filter_medoids_by_surviving_trajectories

    m = {"repA": "path-0", "repB": "path-1"}
    kept, dropped, disagree = filter_medoids_by_surviving_trajectories(m, {})
    assert kept == m and dropped == {} and disagree == {}


# --- funnel z_max must be a wall, not a switch -----------------------------

def _funnel_energy(params, r_z, r_xy):
    """Funnel energy (kJ/mol) for a guest at axial r_z, radial r_xy (Angstrom)."""
    import openmm as mm
    from openmm import unit as ommunit
    from autopath.metadynamics.Funnel import create_funnel_force_from_trajectory_analysis

    sysm = mm.System()
    sysm.addParticle(1.0)
    sysm.addParticle(1.0)
    force = create_funnel_force_from_trajectory_analysis(params)
    sysm.addForce(force)
    ctx = mm.Context(sysm, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName("Reference"))
    # axis is +x, so r_z is along x and r_xy along y
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(r_z / 10.0, r_xy / 10.0, 0)]
                     * ommunit.nanometer)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        ommunit.kilojoule_per_mole)
    del ctx
    return e


def _funnel_params(z_cc=8.0, z_max=20.0, R=4.0, k=10.0):
    from openmm import unit as ommunit
    import numpy as np
    return {
        "host_index": [0], "guest_index": [1],
        "k_xy": k * ommunit.kilocalorie_per_mole / ommunit.angstrom ** 2,
        "z_cc": z_cc * ommunit.angstrom,
        "z_max": z_max * ommunit.angstrom,
        "alpha": 45.0 * ommunit.degrees,
        "R_cylinder": R * ommunit.angstrom,
        "unbinding_axis": np.array([1.0, 0.0, 0.0]),
    }


def test_funnel_is_continuous_across_z_max():
    """Gating the cylinder on step(z_max - |r_z|) zeroes the whole potential past
    z_max, so the energy drops discontinuously and the guest is released."""
    p = _funnel_params()
    r_xy = 10.0  # well outside R_cylinder, so the radial wall is active
    inside = _funnel_energy(p, 19.99, r_xy)
    outside = _funnel_energy(p, 20.01, r_xy)
    assert inside > 0, "radial wall should be active just inside z_max"
    assert abs(outside - inside) < 1.0, (
        f"energy jumps {inside:.1f} -> {outside:.1f} kJ/mol across z_max")


def test_funnel_still_confines_radially_beyond_z_max():
    """Past z_max the guest must not be radially free."""
    p = _funnel_params()
    assert _funnel_energy(p, 30.0, 12.0) > _funnel_energy(p, 30.0, 4.0)


def test_funnel_pushes_back_axially_beyond_z_max():
    """There must be an axial restoring term past z_max, growing with distance."""
    p = _funnel_params()
    on_axis = [_funnel_energy(p, z, 0.0) for z in (20.0, 22.0, 25.0)]
    assert on_axis[0] == pytest.approx(0.0, abs=1e-6), "wall starts at z_max"
    assert on_axis[1] > 0 and on_axis[2] > on_axis[1], "wall must grow outward"


def test_funnel_unchanged_inside_the_cone_and_cylinder():
    """The fix must not perturb the region the funnel already handled."""
    p = _funnel_params()
    # inside the cone, within R_funnel -> free
    assert _funnel_energy(p, 2.0, 1.0) == pytest.approx(0.0, abs=1e-6)
    # inside the cylinder, within R_cylinder -> free
    assert _funnel_energy(p, 15.0, 3.0) == pytest.approx(0.0, abs=1e-6)
    # inside the cylinder, outside R_cylinder -> radial wall
    assert _funnel_energy(p, 15.0, 6.0) > 0


# --- z_cc must put the binding site inside the cone ------------------------

def _unbinding_universe(n_frames=400, seed=1):
    """Ligand leaves along +x. Radial spread is LARGE while bound (the pocket
    lets it rattle) and small far out, which is the shape that exposes a z_cc
    placed at the median: the site ends up in the cylinder."""
    import numpy as np
    import MDAnalysis as mda
    rng = np.random.default_rng(seed)
    u = mda.Universe.empty(2, n_residues=2, atom_resindex=[0, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "C1"])
    u.add_TopologyAttr("resname", ["ALA", "LIG"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("masses", [12.0, 12.0])
    z = np.linspace(1.0, 20.0, n_frames)
    rad = np.where(z < 8.0, rng.normal(0, 3.0, n_frames), rng.normal(0, 1.0, n_frames))
    coords = np.zeros((n_frames, 2, 3))
    coords[:, 1, 0] = z
    coords[:, 1, 1] = rad
    u.load_new(coords, order="fac")
    u.dimensions = [80, 80, 80, 90, 90, 90]
    return u


def _cone_radius_at(params, z_ang):
    from openmm import unit as ommunit
    import numpy as np
    zcc = params["z_cc"].value_in_unit(ommunit.angstrom)
    R = params["R_cylinder"].value_in_unit(ommunit.angstrom)
    a = params["alpha"].value_in_unit(ommunit.degrees)
    return R + max(0.0, zcc - z_ang) * np.tan(np.radians(a))


def test_default_zcc_keeps_the_binding_site_inside_the_cone():
    """Limongelli 2013 requires z_cc > the site's outer edge, so that exploring
    the site is unaffected by the restraint."""
    from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
    from openmm import unit as ommunit
    import numpy as np

    u = _unbinding_universe()
    p = generate_funnel_parameters_from_trajectory(
        u, host_selection="resname ALA", guest_selection="resname LIG",
        stride=1, verbose=False)

    z = np.abs(np.asarray(p["axial_distances"]))
    z_site = float(np.percentile(z, 10.0))
    zcc = p["z_cc"].value_in_unit(ommunit.angstrom)
    assert zcc > z_site, f"z_cc={zcc:.2f} does not clear the site edge {z_site:.2f}"
    # and the cone must be wide enough there to not squeeze the bound ligand
    r_bound = float(np.percentile(np.asarray(p["radial_distances"])[z <= z_site], 95))
    assert _cone_radius_at(p, z_site) > r_bound


def test_median_zcc_would_put_the_site_in_the_cylinder():
    """Pins WHY the default matters: percentile_z=50 reproduces the old bug."""
    from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
    from openmm import unit as ommunit
    import numpy as np

    u = _unbinding_universe()
    bad = generate_funnel_parameters_from_trajectory(
        u, host_selection="resname ALA", guest_selection="resname LIG",
        percentile_z=50.0, stride=1, verbose=False)
    good = generate_funnel_parameters_from_trajectory(
        u, host_selection="resname ALA", guest_selection="resname LIG",
        stride=1, verbose=False)

    z = np.abs(np.asarray(bad["axial_distances"]))
    z_site = float(np.percentile(z, 10.0))
    r_bound = float(np.percentile(np.asarray(bad["radial_distances"])[z <= z_site], 95))

    assert _cone_radius_at(bad, z_site) < _cone_radius_at(good, z_site)
    assert bad["z_cc"].value_in_unit(ommunit.angstrom) < good["z_cc"].value_in_unit(ommunit.angstrom)
    # the default must be the one that leaves the bound ligand unrestrained
    assert _cone_radius_at(good, z_site) > r_bound


def test_core_does_not_force_a_median_zcc():
    import inspect
    from autopath import autopath_core
    src = inspect.getsource(autopath_core)
    assert "percentile_z=50" not in src.replace(" ", "")


# --- walker biases: sum when shared, average when isolated -----------------

def test_openmm_writes_self_bias_and_sums_to_get_the_total():
    """Pins WHY shared-bias files must be summed.

    OpenMM's Metadynamics saves each walker's SELF bias and rebuilds the total by
    adding every walker's file, so one file is a partial contribution -- reading
    only the last walker would give a fraction of the total.
    """
    import inspect
    import openmm.app.metadynamics as m

    src = inspect.getsource(m.Metadynamics._syncWithDisk)
    assert "self._selfBias" in src, "walkers save their own bias, not the total"
    total = inspect.getsource(m.Metadynamics)
    assert "self._totalBias = np.copy(self._selfBias)" in total
    assert "self._totalBias += bias.bias" in total


def test_isolated_walkers_are_averaged_not_summed():
    """Each per-walker biasDir holds a COMPLETE bias, so summing n of them scales
    the free energy by n."""
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis

    one = np.linspace(0.0, 10.0, 25)
    arrays = [one, one, one]
    _, fes_mean = MetadynamicsAnalysis.compute_fes(
        arrays, temperature=300.0, bias_factor=15.0, combine="mean")
    _, fes_one = MetadynamicsAnalysis.compute_fes(
        [one], temperature=300.0, bias_factor=15.0, combine="mean")
    np.testing.assert_allclose(fes_mean, fes_one)

    _, fes_sum = MetadynamicsAnalysis.compute_fes(
        arrays, temperature=300.0, bias_factor=15.0, combine="sum")
    np.testing.assert_allclose(fes_sum, 3.0 * fes_one)


def test_shared_walkers_are_summed():
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis

    a, b = np.full(25, 1.0), np.full(25, 2.0)
    _, fes = MetadynamicsAnalysis.compute_fes(
        [a, b], temperature=300.0, bias_factor=15.0, combine="sum")
    _, fes_total = MetadynamicsAnalysis.compute_fes(
        [a + b], temperature=300.0, bias_factor=15.0, combine="sum")
    np.testing.assert_allclose(fes, fes_total)


def test_compute_fes_rejects_an_unknown_combination_rule():
    import numpy as np, pytest as _pytest
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis
    with _pytest.raises(ValueError):
        MetadynamicsAnalysis.compute_fes(
            [np.zeros(5)], temperature=300.0, bias_factor=15.0, combine="median")


def test_bias_layout_detects_both_arrangements(tmp_path):
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis

    shared = tmp_path / "shared"; shared.mkdir()
    np.save(shared / "bias_7_1.npy", np.zeros(25))
    assert MetadynamicsAnalysis.bias_layout(str(shared)) == "shared"

    iso = tmp_path / "iso"; iso.mkdir()
    (iso / "bias_milestone_1_frame_0").mkdir()
    np.save(iso / "bias_milestone_1_frame_0" / "bias_99_3.npy", np.zeros(25))
    assert MetadynamicsAnalysis.bias_layout(str(iso)) == "isolated"
