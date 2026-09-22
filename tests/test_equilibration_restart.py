"""Restart pair written at the end of equilibration: saved system + saved state must load together.

The restraint forces are stripped from the System before saving, but
``Context.reinitialize(preserveState=True)`` restores global parameters by name, so the
``k_*`` parameters of the removed forces survive in the context and ``saveState`` writes
them into the state XML. Loading that state against the restraint-free system then raises
"invalid parameter name". ``rebuild_simulation`` builds a fresh context from the stripped
system instead, so the pair round-trips.
"""

import numpy as np
import pytest
from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    NonbondedForce,
    Platform,
    System,
    Vec3,
)
from openmm.app import Element, Simulation, Topology
import openmm.unit as openmmunit

from autopath.customForces import remove_openmm_force
from autopath.utils import load_system, rebuild_simulation, save_simulation, save_system

PLATFORM = Platform.getPlatformByName("Reference")
N_PARTICLES = 6
RESTRAINT_NAMES = ("k_protein_BB", "k_protein_SC")


def _topology():
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("AR", chain)
    for _ in range(N_PARTICLES):
        topology.addAtom("Ar", Element.getBySymbol("Ar"), residue)
    topology.setPeriodicBoxVectors(
        (Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3)) * openmmunit.nanometer
    )
    return topology


def _positions():
    return [Vec3(0.3 * i, 0.3 * i, 0.3 * i) for i in range(N_PARTICLES)] * openmmunit.nanometer


def _restrained_system():
    """Argon box with a barostat and one restraint force per ``k_*`` parameter."""
    system = System()
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.0 * openmmunit.nanometer)
    for _ in range(N_PARTICLES):
        system.addParticle(39.95 * openmmunit.amu)
        nonbonded.addParticle(0.0, 0.34, 0.996)
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3))
    system.addForce(MonteCarloBarostat(1 * openmmunit.bar, 300 * openmmunit.kelvin))

    positions = _positions().value_in_unit(openmmunit.nanometer)
    for name in RESTRAINT_NAMES:
        force = CustomExternalForce(f"{name}*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.setName(name)
        force.addGlobalParameter(name, 10.0)
        for axis in ("x0", "y0", "z0"):
            force.addPerParticleParameter(axis)
        for idx in range(N_PARTICLES):
            force.addParticle(idx, list(positions[idx]))
        system.addForce(force)
    return system


def _equilibrated_simulation():
    """A simulation carrying restraints and a few steps of history, as run() leaves it."""
    system = _restrained_system()
    simulation = Simulation(
        _topology(), system, LangevinMiddleIntegrator(300, 1.0, 0.002), PLATFORM
    )
    simulation.context.setPositions(_positions())
    simulation.context.setVelocitiesToTemperature(300 * openmmunit.kelvin, 1234)
    simulation.step(5)
    return simulation, system


def _reload(system_path, state_path):
    """Rebuild a simulation from the saved pair, the way a restart would."""
    simulation = Simulation(
        _topology(),
        load_system(system_path),
        LangevinMiddleIntegrator(300, 1.0, 0.002),
        PLATFORM,
    )
    simulation.loadState(state_path)
    return simulation


def test_reinitialize_alone_leaves_stale_restraint_parameters(tmp_path):
    """reinitialize(preserveState=True) keeps the removed forces' parameters in the context."""
    simulation, system = _equilibrated_simulation()
    remove_openmm_force(system, "k_")
    simulation.context.reinitialize(preserveState=True)

    parameters = simulation.context.getState(getParameters=True).getParameters()
    assert set(RESTRAINT_NAMES) <= set(parameters)

    save_system(system, f"{tmp_path}/system_equil_run.xml")
    save_simulation(simulation, f"{tmp_path}/checkpoint_equil_run")
    with pytest.raises(Exception, match="invalid parameter name"):
        _reload(f"{tmp_path}/system_equil_run.xml", f"{tmp_path}/checkpoint_equil_run.xml")


def test_rebuilt_simulation_saves_a_loadable_pair(tmp_path):
    """The saved system and state form a restraint-free pair that restores the state."""
    simulation, system = _equilibrated_simulation()
    reference = simulation.context.getState(getPositions=True, getVelocities=True)

    remove_openmm_force(system, "k_")
    simulation = rebuild_simulation(simulation, system)

    parameters = simulation.context.getState(getParameters=True).getParameters()
    assert not [name for name in parameters if name.startswith("k_")]

    save_system(system, f"{tmp_path}/system_equil_run.xml")
    save_simulation(simulation, f"{tmp_path}/checkpoint_equil_run")

    saved_system = load_system(f"{tmp_path}/system_equil_run.xml")
    force_names = [f.getName() for f in saved_system.getForces()]
    assert not [name for name in force_names if name.startswith("k_")]
    assert any("Barostat" in name for name in force_names)

    restarted = _reload(f"{tmp_path}/system_equil_run.xml", f"{tmp_path}/checkpoint_equil_run.xml")
    state = restarted.context.getState(getPositions=True, getVelocities=True)
    np.testing.assert_allclose(
        state.getPositions(asNumpy=True).value_in_unit(openmmunit.nanometer),
        reference.getPositions(asNumpy=True).value_in_unit(openmmunit.nanometer),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        state.getVelocities(asNumpy=True).value_in_unit(openmmunit.nanometer / openmmunit.picosecond),
        reference.getVelocities(asNumpy=True).value_in_unit(openmmunit.nanometer / openmmunit.picosecond),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(state.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)),
        np.array(reference.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)),
        atol=1e-6,
    )


def test_rebuilt_simulation_keeps_running(tmp_path):
    """The rebuilt simulation is a working simulation, not just a state holder."""
    simulation, system = _equilibrated_simulation()
    remove_openmm_force(system, "k_")
    simulation = rebuild_simulation(simulation, system)
    simulation.step(5)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert np.isfinite(energy.value_in_unit(openmmunit.kilojoule_per_mole))
