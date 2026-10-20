"""Regression tests for milestone extraction, metadynamics setup and analysis."""
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
    """Re-imaging via a heavy-atom selection also moves the ligand's hydrogens."""
    u = _lig_universe()
    assert _reimaging_ligand(u, "resname LIG and not name H*") is True

    c1, h1 = u.atoms[1].position, u.atoms[2].position
    assert np.linalg.norm(h1 - c1) == pytest.approx(1.0, abs=1e-4)
    # the ligand actually moved
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

def _pathcv(tmp_path, seps, sigma="auto"):
    """Build the path CV through 2-atom milestones (pocket at 0, ligand at `seps` nm)."""
    import openmm as mm
    from autopath.metadynamics.CV import pathCV_cv
    system = mm.System()
    for _ in range(2):
        system.addParticle(12.0)
    spec = pathCV_cv(None, _write_state_pair(tmp_path, seps), [0], [1], system, sigma=sigma)
    return spec._factory(None, 2, None)


def test_auto_sigma_is_half_the_minimum_gap(tmp_path):
    cv = _pathcv(tmp_path, [0.4, 0.9, 1.5])
    assert cv._args["sigma"] == pytest.approx(0.25, abs=1e-4)


def test_auto_sigma_is_positive_and_warns_for_a_non_monotone_milestone_list(tmp_path, caplog):
    """A backwards step must not produce a negative sigma."""
    with caplog.at_level("WARNING", logger="autopath.cv"):
        cv = _pathcv(tmp_path, [0.40, 1.20, 0.50, 0.90, 1.10])
    assert cv._args["sigma"] == pytest.approx(0.5 * 0.20, abs=1e-4)
    assert "not monotonically increasing" in caplog.text


def test_auto_sigma_rejects_all_identical_milestones(tmp_path):
    with pytest.raises(ValueError, match="cannot auto-tune sigma"):
        _pathcv(tmp_path, [0.5, 0.5, 0.5])


def test_explicit_sigma_is_passed_through(tmp_path):
    cv = _pathcv(tmp_path, [0.4, 0.9, 1.5], sigma=0.07)
    assert cv._args["sigma"] == pytest.approx(0.07)


# --- no loadState anywhere -------------------------------------------------

def test_no_module_calls_load_state():
    """loadState is forbidden: it restores global parameters by name and keeps stale values."""
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


# --- funnel cylinder radius ------------------------------------------------

def _funnel_universe(n_frames=400, r_bound=0.5, r_cyl=4.0, seed=0):
    """Ligand exits along +x with small radial spread while bound and large spread in the cylinder."""
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
    """The derived cylinder radius contains most of the reference path's cylinder region."""
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
    """Each per-path milestone subdirectory becomes its own group."""
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
    """Flat milestones form a single group keyed None."""
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
    """Medoids are kept by surviving trajectory, even when their path label is stale."""
    from autopath.utils import filter_medoids_by_surviving_trajectories

    medoid_to_path = {"repA": "path-0_v0.02", "repB": "path-1_v0.02"}
    surviving = {"repA": "path-0_v0.02", "repB": "path-0_v0.02"}  # repB relabelled

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
    """Funnel energy is continuous across z_max."""
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
    """The cone and cylinder regions keep their expected free/walled behaviour."""
    p = _funnel_params()
    # inside the cone, within R_funnel -> free
    assert _funnel_energy(p, 2.0, 1.0) == pytest.approx(0.0, abs=1e-6)
    # inside the cylinder, within R_cylinder -> free
    assert _funnel_energy(p, 15.0, 3.0) == pytest.approx(0.0, abs=1e-6)
    # inside the cylinder, outside R_cylinder -> radial wall
    assert _funnel_energy(p, 15.0, 6.0) > 0


# --- z_cc must put the binding site inside the cone ------------------------

def _unbinding_universe(n_frames=400, seed=1):
    """Ligand exits along +x with large radial spread while bound and small spread far out."""
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
    """Default z_cc lies beyond the binding site, so the site is unrestrained (Limongelli 2013)."""
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
    """A median z_cc (percentile_z=50) narrows the cone at the site relative to the default."""
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


# --- funnel standard-state correction ---------------------------------------

def _flat_funnel_params(R=5.0, z_cc=8.0, z_max=20.0):
    from openmm import unit as ommunit
    return {"R_cylinder": R * ommunit.angstrom,
            "z_cc": z_cc * ommunit.angstrom,
            "z_max": z_max * ommunit.angstrom}


def _well(depth_kj=-40.0, n=125, width_frac=0.10):
    """FES with a square well of known width at the bound end, flat elsewhere."""
    import numpy as np
    fe = np.zeros(n)
    fe[:max(1, int(width_frac * n))] = depth_kj
    return fe


def test_correction_uses_cross_section_not_cylinder_volume():
    """The standard-state correction uses the cylinder cross-section, kT*ln(pi R^2 C0)."""
    import numpy as np, math
    from autopath.metadynamics.Analysis import correct_fe_for_funnel

    R, kT = 5.0, 8.314e-3 * 298.15
    r = correct_fe_for_funnel(_well(), _flat_funnel_params(R=R),
                              temperature=298.15, cv_length_per_unit_ang=10.0)
    # rebuild the expected value straight from Eq. 15
    fe = _well(); fe = fe - fe[-10:].mean()
    bound = fe[:max(1, int(0.20 * len(fe)))]
    d_cv = 1.0 / (len(fe) - 1)
    I = float(np.trapz(np.exp(-bound / kT), dx=d_cv) * 10.0)
    expected = -kT * math.log(I) - kT * math.log(math.pi * R ** 2 / 1660.0)
    assert r["dG_bind_std_kj_mol"] == pytest.approx(expected, abs=1e-6)


def test_correction_is_independent_of_the_cylinder_length():
    """z_max only sets the cylinder length, which Eq. 15 does not use."""
    from autopath.metadynamics.Analysis import correct_fe_for_funnel
    a = correct_fe_for_funnel(_well(), _flat_funnel_params(z_max=20.0),
                              temperature=298.15, cv_length_per_unit_ang=10.0)
    b = correct_fe_for_funnel(_well(), _flat_funnel_params(z_max=60.0),
                              temperature=298.15, cv_length_per_unit_ang=10.0)
    assert a["dG_bind_std_kj_mol"] == pytest.approx(b["dG_bind_std_kj_mol"])


def test_a_wider_bound_basin_binds_more_strongly():
    """The basin integral makes a wider bound basin bind more strongly."""
    from autopath.metadynamics.Analysis import correct_fe_for_funnel
    narrow = correct_fe_for_funnel(_well(width_frac=0.04), _flat_funnel_params(),
                                   temperature=298.15, cv_length_per_unit_ang=10.0)
    wide = correct_fe_for_funnel(_well(width_frac=0.16), _flat_funnel_params(),
                                 temperature=298.15, cv_length_per_unit_ang=10.0)
    assert wide["dG_bind_std_kj_mol"] < narrow["dG_bind_std_kj_mol"]


def test_correction_refuses_without_a_length_scale():
    """A dimensionless progress coordinate cannot supply the length Eq. 15 needs."""
    from autopath.metadynamics.Analysis import correct_fe_for_funnel
    with pytest.raises(ValueError, match="cv_length_per_unit_ang"):
        correct_fe_for_funnel(_well(), _flat_funnel_params(), temperature=298.15)


# --- convergence criterion -------------------------------------------------

def test_convergence_needs_recrossing_and_a_flat_plateau():
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA

    flat = np.concatenate([np.full(40, -40.0), np.zeros(85)])
    assert MA.assess_convergence([{"n_round_trips": 2}], flat)["converged"]

    # no recrossing
    v = MA.assess_convergence([{"n_round_trips": 0}], flat)
    assert not v["converged"] and "round trip" in " ".join(v["reasons"])

    # sloping unbound region: the paper's flatness test
    sloped = np.concatenate([np.full(40, -40.0), np.linspace(0, -30, 85)])
    v = MA.assess_convergence([{"n_round_trips": 5}], sloped)
    assert not v["converged"] and "plateau" in " ".join(v["reasons"])


def test_convergence_sums_round_trips_over_walkers():
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA
    flat = np.concatenate([np.full(40, -40.0), np.zeros(85)])
    v = MA.assess_convergence(
        [{"n_round_trips": 0}, {"n_round_trips": 1}], flat, min_round_trips=1)
    assert v["converged"] and v["n_round_trips"] == 1


# --- dG(t) convergence curve ----------------------------------------------

def test_dG_plateau_detects_a_settled_run():
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA
    t = np.linspace(0, 5, 20)
    assert MA.dG_plateau(t, np.full(20, -25.0))["settled"]


def test_dG_plateau_detects_a_drifting_run():
    """A drifting ΔG(t) is reported as not settled."""
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA
    t = np.linspace(0, 5, 20)
    drifting = np.linspace(-10.0, -45.0, 20)
    v = MA.dG_plateau(t, drifting)
    assert not v["settled"] and "still moves" in v["reason"]


def test_convergence_fails_when_dG_has_not_settled():
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA
    flat = np.concatenate([np.full(40, -40.0), np.zeros(85)])
    v = MA.assess_convergence(
        [{"n_round_trips": 3}], flat,
        dG_plateau_result={"settled": False, "reason": "drifting", "n_points": 9})
    assert not v["converged"] and "ΔG(t)" in " ".join(v["reasons"])


def test_a_flat_curve_without_both_basins_is_not_settled():
    """A flat ΔG(t) only counts as settled once both basins were visited."""
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA
    t = np.linspace(0, 5, 20)
    flat_but_empty = np.full(20, -6.0)          # flat, and nearly zero binding
    v = MA.dG_plateau(t, flat_but_empty, both_basins=False)
    assert not v["settled"]
    assert "did not visit both basins" in v["reason"]
    assert MA.dG_plateau(t, flat_but_empty, both_basins=True)["settled"]


# --- the gate belongs on the reported number, not on the curve -------------

def test_correction_flags_an_empty_basin_instead_of_raising_when_not_strict():
    """An empty bound basin raises when strict, and is flagged with a finite ΔG otherwise."""
    import numpy as np
    from autopath.metadynamics.Analysis import correct_fe_for_funnel
    empty = np.zeros(125)                       # no well anywhere
    with pytest.raises(ValueError, match="Bound state not sampled"):
        correct_fe_for_funnel(empty, _flat_funnel_params(), temperature=298.15,
                              cv_length_per_unit_ang=10.0)
    r = correct_fe_for_funnel(empty, _flat_funnel_params(), temperature=298.15,
                              cv_length_per_unit_ang=10.0, strict=False)
    assert r["bound_state_sampled"] is False
    assert np.isfinite(r["dG_bind_std_kj_mol"])
    # an empty site gives a small basin integral, hence an unfavourable ΔG
    assert r["dG_bind_std_kj_mol"] > 0
    assert correct_fe_for_funnel(_well(), _flat_funnel_params(), temperature=298.15,
                                 cv_length_per_unit_ang=10.0)["bound_state_sampled"]


def test_dG_plateau_will_not_settle_on_unsampled_snapshots():
    """The plateau test uses only snapshots marked as sampled."""
    import numpy as np
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis as MA
    t = np.linspace(0, 5, 20)
    g = np.full(20, -25.0)                      # perfectly flat
    sampled = np.zeros(20, dtype=bool)
    sampled[:4] = True                          # only the first few are real
    v = MA.dG_plateau(t, g, sampled=sampled)
    assert v["n_points"] == 4, "the mask, not finiteness, must select the points"
    assert not v["settled"] and not v["ends_resolved"]
    assert "not resolved at the end" in v["reason"]
    # without a mask the same curve settles
    assert MA.dG_plateau(t, g)["settled"]


# --- shared bias: sequential walkers form one curve ------------------------

def test_walker_seed_order_reverses_only_for_unbound_first():
    """Milestones are ordered bound -> unbound and reversed only for unbound_first."""
    from autopath.autopath_core import walker_seed_order
    m = ["bound", "mid", "unbound"]
    assert walker_seed_order(m, "unbound_first") == ["unbound", "mid", "bound"]
    assert walker_seed_order(m, "bound_first") == ["bound", "mid", "unbound"]
    # must not mutate the caller's list
    assert m == ["bound", "mid", "unbound"]


def test_walker_seed_order_rejects_an_unknown_order():
    from autopath.autopath_core import walker_seed_order
    with pytest.raises(ValueError, match="unbound_first"):
        walker_seed_order(["a", "b"], "middle_first")


# --- reaching bulk: the cylinder cap and the path's far end ----------------

def _two_atom_universe(guest_z=2.0):
    """One host atom at the origin, one guest atom up the z axis."""
    import MDAnalysis as mda
    u = mda.Universe.empty(2, n_residues=2, atom_resindex=[0, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr('name', ['P', 'G'])
    u.add_TopologyAttr('resname', ['PRO', 'LIG'])
    u.add_TopologyAttr('mass', [12.0, 12.0])
    u.atoms.positions = [[0.0, 0.0, 0.0], [0.0, 0.0, guest_z]]
    return u


def test_z_for_bulk_is_where_the_guest_clears_the_cutoff():
    """Returns the first axial z at which the guest clears the bulk cutoff."""
    import numpy as np
    from autopath.metadynamics.Funnel import z_where_guest_reaches_bulk
    u = _two_atom_universe(guest_z=2.0)
    guest = u.select_atoms('name G')
    z = z_where_guest_reaches_bulk(u, guest, np.zeros(3), np.array([0.0, 0.0, 1.0]),
                                   cutoff_ang=10.0, protein_selection='name P',
                                   step_ang=0.5)
    # starts 2 A out, needs >10 A, so the first 0.5 A step past 10 is 10.5
    assert z == pytest.approx(10.5, abs=1e-6)
    # and the search must not leave the guest displaced
    assert u.select_atoms('name G').positions[0][2] == pytest.approx(2.0)


def test_z_for_bulk_raises_when_the_axis_never_escapes():
    import numpy as np
    from autopath.metadynamics.Funnel import z_where_guest_reaches_bulk
    u = _two_atom_universe()
    with pytest.raises(ValueError, match="never cleared"):
        z_where_guest_reaches_bulk(u, u.select_atoms('name G'), np.zeros(3),
                                   np.array([0.0, 0.0, 1.0]), cutoff_ang=10.0,
                                   protein_selection='name P', max_extra_ang=1.0)


def test_extend_milestones_keeps_the_existing_spacing():
    """Extending the path adds milestones no wider apart than the existing ones."""
    import numpy as np
    from autopath.metadynamics.CV import _extend_milestones
    m = np.array([[0.70], [1.10], [1.50]])
    out = _extend_milestones(m, 2.70).flatten()
    assert out[:3].tolist() == [0.70, 1.10, 1.50]
    gaps = np.abs(np.diff(out))
    # every new gap stays within the range the original path already had
    assert gaps.max() <= np.abs(np.diff(m.flatten())).max() + 1e-9
    assert out[-1] == pytest.approx(2.70)


def test_extend_milestones_rejects_a_target_inside_the_path():
    import numpy as np
    from autopath.metadynamics.CV import _extend_milestones
    with pytest.raises(ValueError, match="must be beyond"):
        _extend_milestones(np.array([[0.7], [1.5]]), 1.2)


def test_com_distance_at_z_converts_the_wall_into_a_path_target():
    """Converts an axial wall position into the corresponding COM separation."""
    import numpy as np
    from autopath.metadynamics.Funnel import com_distance_at_z
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])   # host, guest (A)
    # move the guest to z = 10 A along z: COM separation is 10 A = 1.0 nm
    assert com_distance_at_z(pos, [1], [0], [0.0, 0.0, 1.0], 10.0) == pytest.approx(1.0)
    # an off-axis guest ends up further than z, since z is only the projection
    off = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 2.0]])
    d = com_distance_at_z(off, [1], [0], [0.0, 0.0, 1.0], 10.0)
    assert d > 1.0 and d == pytest.approx(np.hypot(3.0, 10.0) / 10.0)


def test_z_max_for_bulk_takes_the_worst_probed_frame():
    """The cap is the maximum required over all probed frames."""
    import numpy as np
    from autopath.metadynamics.Funnel import z_max_for_bulk
    u = _two_atom_universe(guest_z=2.0)
    # two probes of the same frame differ only in the z label, so the required
    # cap is identical; the point is that the max is what gets returned
    z = z_max_for_bulk(u, 'name G', 'name P', np.array([0.0, 0.0, 1.0]),
                       frame_z_pairs=[(0, 2.0), (0, 17.0)], cutoff_ang=10.0,
                       n_probe=2, protein_selection='name P', step_ang=0.5)
    assert z == pytest.approx(10.5, abs=1e-6)


# --- aligned trajectories must not gain atomic overlaps --------------------

def _triclinic_solvent_traj(tmp_path, n_wat=80, rot=0.7):
    """Two frames in a dodecahedron cell; in frame 1 only the solute is rotated, so
    superposing onto frame 0 rotates the solvent relative to the box."""
    import numpy as np, mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    # standard residues, since bonds must survive a PDB round-trip; they form one
    # molecule for image_molecules to anchor on
    ca = []
    bb_atoms = []
    for i in range(4):
        r = top.add_residue('ALA', ch)
        n = top.add_atom('N', md.element.nitrogen, r)
        c_a = top.add_atom('CA', md.element.carbon, r)
        c = top.add_atom('C', md.element.carbon, r)
        o = top.add_atom('O', md.element.oxygen, r)
        ca.append(c_a); bb_atoms.append((n, c_a, c, o))
    wat = []
    for i in range(n_wat):
        r = top.add_residue('HOH', ch)
        o = top.add_atom('O', md.element.oxygen, r)
        h1 = top.add_atom('H1', md.element.hydrogen, r)
        h2 = top.add_atom('H2', md.element.hydrogen, r)
        top.add_bond(o, h1); top.add_bond(o, h2)
        wat.append((o.index, h1.index, h2.index))
    L = 3.0
    box = np.array([[L, 0, 0], [0, L, 0], [L/2, L/2, L*0.7071]])
    n_at = top.n_atoms
    xyz = np.zeros((2, n_at, 3))
    # non-collinear solute, otherwise superpose has no unique rotation
    for k, (n, c_a, c, o) in enumerate(bb_atoms):
        base = np.array([1.20 + 0.30*np.cos(2.1*k),
                         1.50 + 0.30*np.sin(2.1*k),
                         0.80 + 0.22*k])
        xyz[:, n.index]   = base
        xyz[:, c_a.index] = base + [0.147, 0.0, 0.0]
        xyz[:, c.index]   = base + [0.147 + 0.153, 0.0, 0.0]
        xyz[:, o.index]   = base + [0.147 + 0.153, 0.123, 0.0]
    # solvent on a regular fractional lattice so the input has no clashes
    g = np.array([(i, j, k) for i in range(5) for j in range(4) for k in range(4)],
                 dtype=float)
    f = (g + 0.5) / np.array([5.0, 4.0, 4.0])
    for i, (o, h1, h2) in enumerate(wat):
        c = f[i] @ box
        xyz[:, o] = c
        xyz[:, h1] = c + [0.0957, 0, 0]
        xyz[:, h2] = c + [-0.024, 0.0927, 0]
    R = np.array([[np.cos(rot), -np.sin(rot), 0],
                  [np.sin(rot),  np.cos(rot), 0], [0, 0, 1]])
    idx = [a.index for grp in bb_atoms for a in grp]
    pivot = xyz[1, idx].mean(axis=0)
    xyz[1, idx] = (xyz[1, idx] - pivot) @ R.T + pivot     # solute rotates only
    t = md.Trajectory(xyz, top)
    t.unitcell_vectors = np.repeat(box[None, :, :], 2, axis=0)
    pdb = str(tmp_path / 'sys.pdb'); dcd = str(tmp_path / 'traj.dcd')
    t[0].save_pdb(pdb); t.save_dcd(dcd)
    return pdb, dcd, top


def _rigid_rmsd(A, B):
    """RMSD of B onto A after optimal rigid superposition (nm)."""
    import numpy as np
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Bc.T @ Ac)
    d = np.sign(np.linalg.det(V @ W))
    R = V @ np.diag([1, 1, d]) @ W
    return float(np.sqrt((((Bc @ R) - Ac) ** 2).sum(axis=1).mean()))


def test_alignment_applies_only_a_rigid_transform(tmp_path):
    """Alignment applies only a rigid transform; imaging after superposition would not,
    because the unit cell vectors are not rotated with the coordinates."""
    import copy
    import mdtraj as md
    from autopath.utils import wrap_align_save_traj
    pdb, dcd, top = _triclinic_solvent_traj(tmp_path)
    # baseline: imaging before superposition may shift molecules by lattice vectors
    src = md.load(dcd, top=pdb).center_coordinates().image_molecules(make_whole=True)

    # fixture check: imaging after the rotation must visibly damage the frame
    sel = src.topology.select('name CA')
    rot = copy.deepcopy(src).superpose(src[0], atom_indices=sel)
    buggy = copy.deepcopy(rot).image_molecules(make_whole=True)
    damage = max(_rigid_rmsd(rot.xyz[i], buggy.xyz[i]) for i in range(src.n_frames))
    assert damage > 0.05, (
        f"fixture does not exercise imaging-after-rotation (damage {damage*10:.3f} A); "
        "a passing result here would be meaningless")

    out = wrap_align_save_traj(dcd, pdb, remove_original=False,
                               align_select='name CA', out_dir=str(tmp_path))
    got = md.load(out[0], top=pdb)
    assert got.n_frames == src.n_frames
    worst = max(_rigid_rmsd(src.xyz[i], got.xyz[i]) for i in range(src.n_frames))
    assert worst < 1e-3, (
        f"alignment moved atoms by {worst*10:.2f} A beyond any rigid transform; "
        "imaging must happen before superposition, never after")


# --- cylinder cap always reaches bulk ----------------------------------------

def test_cylinder_cap_is_set_by_the_bulk_cutoff_not_the_trajectory_end():
    """The cylinder cap is set by bulk_cutoff_ang, not by where the trajectory ends."""
    from autopath.metadynamics.Funnel import generate_funnel_parameters_from_trajectory
    from openmm import unit as ommunit

    caps = {}
    for cut in (10.0, 35.0):
        p = generate_funnel_parameters_from_trajectory(
            _funnel_universe(), host_selection="resname ALA", guest_selection="resname LIG",
            bulk_cutoff_ang=cut, stride=1, verbose=False)
        caps[cut] = p["z_max"].value_in_unit(ommunit.angstrom)
    assert caps[35.0] >= 35.0 > 25.0          # beyond the 25 A trajectory end
    assert caps[35.0] > caps[10.0]
