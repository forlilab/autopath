"""Restart pair written at the end of relaxation MD: saved system + saved state must load together.

The temporary ligand restraints are stripped from the System before saving, but the context
that produced the state still defines their ``k_*`` global parameters, so ``saveState`` writes
them into the state XML. Loading that state against the restraint-free system then raises
"invalid parameter name". Rebuilding the simulation on the stripped system avoids it.
"""

import numpy as np
import pytest
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation
import openmm.unit as openmmunit

from autopath.customForces import add_harmonic_restraints, remove_openmm_force
from autopath.relax_md import RelaxMD
from autopath.utils import load_system, save_simulation, save_system
from tests.test_equilibration_restart import (
    N_PARTICLES,
    PLATFORM,
    _positions,
    _restrained_system,
    _topology,
)

RESTRAINT_NAME = "k_harmonic_restrain"
RUN_ID = "relax_run"


def _relax_stage(out_dir):
    """A relax run as it stands just before writing its outputs."""
    topology = _topology()
    # Same argon box and barostat, carrying the restraint relax_md adds instead.
    system = remove_openmm_force(_restrained_system(), "k_protein")
    ligand_atoms = list(range(N_PARTICLES))
    add_harmonic_restraints(
        system,
        _positions(),
        topology,
        ligand_atoms,
        restraint_force=5,
        force_name=RESTRAINT_NAME,
        force_group=19,
    )

    simulation = Simulation(
        topology, system, LangevinMiddleIntegrator(300, 1.0, 0.002), PLATFORM
    )
    simulation.context.setPositions(_positions())
    simulation.context.setVelocitiesToTemperature(300 * openmmunit.kelvin, 1234)
    simulation.step(5)

    runner = RelaxMD(
        topology=topology,
        ligand_atoms=ligand_atoms,
        pocket_atoms=[0],
        out_dir=str(out_dir),
        platform="Reference",
    )
    return runner, simulation, system


def _reload(topology, system_path, state_path):
    """Rebuild a simulation from the saved pair, the way a downstream stage would."""
    simulation = Simulation(
        topology,
        load_system(system_path),
        LangevinMiddleIntegrator(300, 1.0, 0.002),
        PLATFORM,
    )
    simulation.loadState(state_path)
    return simulation


def test_saving_without_rebuilding_leaves_a_stale_restraint_parameter(tmp_path):
    """Stripping the force alone keeps its parameter in the context and in the saved state."""
    runner, simulation, system = _relax_stage(tmp_path)
    remove_openmm_force(system, RESTRAINT_NAME)

    parameters = simulation.context.getState(getParameters=True).getParameters()
    assert RESTRAINT_NAME in parameters

    save_system(system, f"{tmp_path}/{RUN_ID}_relax_system.xml")
    save_simulation(simulation, f"{tmp_path}/{RUN_ID}_relax_checkpoint")
    with pytest.raises(Exception, match="invalid parameter name"):
        _reload(
            runner.topology,
            f"{tmp_path}/{RUN_ID}_relax_system.xml",
            f"{tmp_path}/{RUN_ID}_relax_checkpoint.xml",
        )


def test_relax_outputs_form_a_loadable_pair(tmp_path):
    """The system and state written at the end of relaxation restore the run."""
    runner, simulation, system = _relax_stage(tmp_path)
    reference = simulation.context.getState(getPositions=True, getVelocities=True)

    runner._save_outputs(simulation, system, RUN_ID)

    saved_system = load_system(f"{tmp_path}/{RUN_ID}_relax_system.xml")
    force_names = [f.getName() for f in saved_system.getForces()]
    assert not [name for name in force_names if name.startswith("k_")]
    assert any("Barostat" in name for name in force_names)

    restarted = _reload(
        runner.topology,
        f"{tmp_path}/{RUN_ID}_relax_system.xml",
        f"{tmp_path}/{RUN_ID}_relax_checkpoint.xml",
    )
    parameters = restarted.context.getState(getParameters=True).getParameters()
    assert not [name for name in parameters if name.startswith("k_")]

    state = restarted.context.getState(getPositions=True, getVelocities=True)
    np.testing.assert_allclose(
        state.getPositions(asNumpy=True).value_in_unit(openmmunit.nanometer),
        reference.getPositions(asNumpy=True).value_in_unit(openmmunit.nanometer),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        state.getVelocities(asNumpy=True).value_in_unit(
            openmmunit.nanometer / openmmunit.picosecond
        ),
        reference.getVelocities(asNumpy=True).value_in_unit(
            openmmunit.nanometer / openmmunit.picosecond
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(state.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)),
        np.array(reference.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)),
        atol=1e-6,
    )


def test_relax_writes_the_expected_output_files(tmp_path):
    """Checkpoint, state, system and PDB keep their names, and the PDB carries the box."""
    runner, simulation, system = _relax_stage(tmp_path)
    box = simulation.context.getState().getPeriodicBoxVectors()

    returned_simulation, returned_system = runner._save_outputs(simulation, system, RUN_ID)

    for suffix in ("_relax_checkpoint.chk", "_relax_checkpoint.xml", "_relax_system.xml", "_relax.pdb"):
        assert (tmp_path / f"{RUN_ID}{suffix}").exists()
    assert returned_system is system
    np.testing.assert_allclose(
        np.array(runner.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)),
        np.array(box.value_in_unit(openmmunit.nanometer)),
        atol=1e-6,
    )
    returned_simulation.step(5)
    energy = returned_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert np.isfinite(energy.value_in_unit(openmmunit.kilojoule_per_mole))
