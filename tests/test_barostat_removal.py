"""Barostats of every type are removed when a run switches to NVT."""

import pytest

openmm = pytest.importorskip("openmm")

import openmm.unit as openmmunit
from openmm import LangevinMiddleIntegrator, Platform, System, Vec3
from openmm.app import Simulation, Topology, element

from autopath.customForces import remove_openmm_force
from autopath.equilibration import run_restrained_md
from autopath.utils import add_barostat


def _simulation(n=4):
    system = System()
    system.setDefaultPeriodicBoxVectors(
        Vec3(5, 0, 0) * openmmunit.nanometer,
        Vec3(0, 5, 0) * openmmunit.nanometer,
        Vec3(0, 0, 5) * openmmunit.nanometer,
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    system.addForce(nb)  # a barostat needs a periodic force
    topology = Topology()
    residue = topology.addResidue("XXX", topology.addChain())
    for i in range(n):
        system.addParticle(12.0 * openmmunit.dalton)
        nb.addParticle(0.0, 0.3, 0.0)
        topology.addAtom(f"C{i}", element.carbon, residue)
    integrator = LangevinMiddleIntegrator(
        300 * openmmunit.kelvin, 1 / openmmunit.picosecond, 0.001 * openmmunit.picoseconds
    )
    sim = Simulation(topology, system, integrator, Platform.getPlatformByName("Reference"))
    sim.context.setPositions([Vec3(0.3 * i, 0, 0) for i in range(n)] * openmmunit.nanometer)
    return sim, system, integrator


def _n_barostats(system):
    return sum("Barostat" in type(system.getForce(i)).__name__ for i in range(system.getNumForces()))


@pytest.mark.parametrize("is_membrane", [False, True])
def test_remove_matches_every_barostat_type(is_membrane):
    sim, system, _ = _simulation()
    add_barostat(system, 300, is_membrane)
    assert _n_barostats(system) == 1
    remove_openmm_force(system, "MonteCarlo")
    assert _n_barostats(system) == 0


@pytest.mark.parametrize("is_membrane", [False, True])
def test_npt_nvt_npt_keeps_one_barostat(is_membrane):
    def stage(npt):
        return dict(name="s", forces=[], npt_flag=npt, nsteps=2, stepsize=0.001)

    sim, system, integrator = _simulation()
    run_restrained_md(sim, system, integrator, [], [stage(True), stage(False)], 300, is_membrane)
    assert _n_barostats(system) == 0

    sim, system, integrator = _simulation()
    run_restrained_md(
        sim, system, integrator, [], [stage(True), stage(False), stage(True)], 300, is_membrane
    )
    assert _n_barostats(system) == 1
