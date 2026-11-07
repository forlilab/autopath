"""Per-component restraint types.

``components_lookup`` accepts either a bare selection string (harmonic, the legacy
form) or a dict carrying a ``type``. Lipid headgroups use ``harmonic_z`` so the
bilayer keeps its leaflet registry and thickness while lateral diffusion and
area-per-lipid stay free for the semi-isotropic barostat to relax.
"""

import json

import pytest

openmm = pytest.importorskip("openmm")

import openmm.unit as openmmunit
from openmm import LangevinMiddleIntegrator, Platform, System, Vec3
from openmm.app import Simulation, Topology, element

from autopath.customForces import _FG_COMPONENTS, add_harmonic_z_restraints
from autopath.equilibration import RESTRAINT_TYPES, Equilibration

MEMB_10NS = "autopath/data/eq_lig-prot-memb_10ns_4fs.json"
SOLUBLE_5NS = "autopath/data/eq_lig-prot_5ns_4fs.json"


def _one_particle_at_origin():
    system = System()
    system.setDefaultPeriodicBoxVectors(
        Vec3(6, 0, 0) * openmmunit.nanometer,
        Vec3(0, 6, 0) * openmmunit.nanometer,
        Vec3(0, 0, 6) * openmmunit.nanometer,
    )
    topology = Topology()
    residue = topology.addResidue("XXX", topology.addChain())
    system.addParticle(12.0 * openmmunit.dalton)
    topology.addAtom("C0", element.carbon, residue)

    positions = [Vec3(0.0, 0.0, 0.0)] * openmmunit.nanometer
    add_harmonic_z_restraints(
        system, positions, topology, [0], restraint_force=10,
        force_name="k_test", force_group=_FG_COMPONENTS,
    )
    integrator = LangevinMiddleIntegrator(
        300 * openmmunit.kelvin, 1 / openmmunit.picosecond, 0.001 * openmmunit.picoseconds
    )
    simulation = Simulation(
        topology, system, integrator, Platform.getPlatformByName("Reference")
    )
    return simulation


def _energy_at(simulation, x, y, z):
    simulation.context.setPositions([Vec3(x, y, z)] * openmmunit.nanometer)
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(openmmunit.kilojoule_per_mole)
    force = state.getForces().value_in_unit(
        openmmunit.kilojoule_per_mole / openmmunit.nanometer
    )[0]
    return energy, force


def test_z_restraint_is_free_in_xy():
    simulation = _one_particle_at_origin()
    energy, force = _energy_at(simulation, 1.5, -1.2, 0.0)
    assert energy == pytest.approx(0.0)
    assert force.x == pytest.approx(0.0) and force.y == pytest.approx(0.0)


def test_z_restraint_bites_in_z():
    simulation = _one_particle_at_origin()
    energy, force = _energy_at(simulation, 0.0, 0.0, 0.4)
    # U = k * dz^2, k = 10 kcal/mol/A^2 = 4184 kJ/mol/nm^2
    assert energy == pytest.approx(4184.0 * 0.4**2, rel=1e-6)
    assert force.z < 0  # pulls back toward z0
    assert force.x == pytest.approx(0.0) and force.y == pytest.approx(0.0)


def test_z_restraint_uses_minimum_image():
    """A particle near the far z face is pulled across the boundary, not through the box."""
    simulation = _one_particle_at_origin()
    near, across = _energy_at(simulation, 0, 0, 0.5)[0], _energy_at(simulation, 0, 0, 5.5)[0]
    assert across == pytest.approx(near, rel=1e-6)


def test_headgroups_are_z_only_and_everything_else_is_harmonic(tmp_path):
    raw = json.loads(open(MEMB_10NS).read())
    equil = Equilibration(
        out_dir=str(tmp_path), protocol_fname=MEMB_10NS, platform="CPU"
    )
    assert equil.component_specs["lipid_head"]["type"] == "harmonic_z"
    for name in raw["components_lookup"]:
        if name != "lipid_head":
            assert equil.component_specs[name]["type"] == "harmonic"


def test_bare_string_components_still_parse(tmp_path):
    """The soluble protocol uses the legacy string form throughout."""
    equil = Equilibration(
        out_dir=str(tmp_path), protocol_fname=SOLUBLE_5NS, platform="CPU"
    )
    assert all(s["type"] == "harmonic" for s in equil.component_specs.values())
    assert equil.components_lookup["ligand"] == "resname UNK and not name H*"


def test_unknown_restraint_type_raises(tmp_path):
    protocol = json.loads(open(MEMB_10NS).read())
    protocol["components_lookup"]["lipid_head"]["type"] = "springy"
    fname = tmp_path / "bad.json"
    fname.write_text(json.dumps(protocol))

    with pytest.raises(ValueError, match="springy"):
        Equilibration(out_dir=str(tmp_path), protocol_fname=str(fname), platform="CPU")


def test_every_declared_type_is_dispatchable():
    for protocol in (MEMB_10NS, SOLUBLE_5NS):
        for value in json.loads(open(protocol).read())["components_lookup"].values():
            kind = value["type"] if isinstance(value, dict) else "harmonic"
            assert kind in RESTRAINT_TYPES
