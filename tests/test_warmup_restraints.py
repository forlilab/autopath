"""Warm-up restraints must be live before the temperature ramp.

Pinned bug. ``Equilibration.run`` removes the minimization restraints and re-adds them
anchored to the minimized geometry with ``restraint_force=15``, then calls
``Context.reinitialize(preserveState=True)``. That call preserves state by writing an
internal checkpoint and reloading it, and a checkpoint stores global parameters **by
name** -- so ``k_<component>`` comes back holding whatever the previous context held
(zero, from the last minimization stage) and the re-added force's
``addGlobalParameter`` default is silently discarded. Re-adding the force object does
not help; the name collision is what carries the stale value.

The warm-up therefore ran completely unrestrained (restraint force group energy exactly
0.0 kJ/mol for all 100 ps), and the first equilibration stage then switched the
restraints on at full strength against coordinates the system had already drifted away
from. On membrane systems that is an immediate ``Particle coordinate is NaN`` inside
Stage 1.

Two invariants are pinned here:

1. Only an explicit push sets the warm-up constants. The checkpoint is snapshotted
   when ``reinitialize`` is called, so a push on either side of it survives -- but
   the force's constructor default never does.
2. Every shipped protocol must define warm-up force constants, and they must be at
   least as stiff as equilibration Stage 1 so the ladder never ramps *up* into MD.

No prmtop is needed: the context-level tests drive a 20-particle box through the exact
remove/re-add/reinitialize sequence used by ``Equilibration.run``.
"""

import json

import pytest

openmm = pytest.importorskip("openmm")

import openmm.unit as openmmunit
from openmm import LangevinMiddleIntegrator, Platform, System, Vec3
from openmm.app import Simulation, Topology, element

from autopath.customForces import (
    _FG_COMPONENTS,
    add_harmonic_restraints,
    remove_openmm_force,
)
from autopath.equilibration import (
    Equilibration,
    read_force_constants,
    update_force_constants,
    warm_up_system,
)

N_PARTICLES = 20
COMPONENTS = ["protein_BB", "protein_SC"]
# Component -> atom indices, standing in for the MDAnalysis selections in run().
IDX = {"protein_BB": list(range(0, 10)), "protein_SC": list(range(10, N_PARTICLES))}

PROTOCOLS = [
    "autopath/data/eq_lig-prot_5ns_4fs.json",
    "autopath/data/eq_lig-prot-memb_10ns_4fs.json",
]


def _build_simulation():
    """A 20-particle box with no forces, on the Reference platform."""
    system = System()
    system.setDefaultPeriodicBoxVectors(
        Vec3(5, 0, 0) * openmmunit.nanometer,
        Vec3(0, 5, 0) * openmmunit.nanometer,
        Vec3(0, 0, 5) * openmmunit.nanometer,
    )
    topology = Topology()
    residue = topology.addResidue("XXX", topology.addChain())
    for i in range(N_PARTICLES):
        system.addParticle(12.0 * openmmunit.dalton)
        topology.addAtom(f"C{i}", element.carbon, residue)

    positions = [Vec3(0.2 * i, 0.0, 0.0) for i in range(N_PARTICLES)] * openmmunit.nanometer
    integrator = LangevinMiddleIntegrator(
        300 * openmmunit.kelvin, 1 / openmmunit.picosecond, 0.001 * openmmunit.picoseconds
    )
    simulation = Simulation(
        topology, system, integrator, Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions(positions)
    return simulation, system, topology, positions


def _add_restraints(system, positions, topology, force=15):
    for num, name in enumerate(COMPONENTS):
        add_harmonic_restraints(
            system,
            positions,
            topology,
            IDX[name],
            restraint_force=force,
            force_name=f"k_{name}",
            force_group=_FG_COMPONENTS + num,
        )


def _minimized_state():
    """Drive a context up to the point ``run()`` re-adds the restraints.

    Returns the simulation with all ``k_*`` constants at zero, as the last
    minimization stage leaves them.
    """
    simulation, system, topology, positions = _build_simulation()
    _add_restraints(system, positions, topology)
    simulation.context.reinitialize(preserveState=True)

    # last minimization stage: everything released
    update_force_constants(simulation, {name: 0.0 for name in COMPONENTS})
    assert read_force_constants(simulation, COMPONENTS) == pytest.approx(
        {name: 0.0 for name in COMPONENTS}
    )
    return simulation, system, topology, positions


# --------------------------------------------------------------------------- #
# 1. the context-level trap
# --------------------------------------------------------------------------- #


def test_readd_default_does_not_survive_reinitialize():
    """Characterize the trap: the re-added force's default is discarded."""
    simulation, system, topology, positions = _minimized_state()

    system = remove_openmm_force(system, "k_")
    _add_restraints(system, positions, topology, force=15)
    simulation.context.reinitialize(preserveState=True)

    # The 15 kcal/mol/A^2 default passed to add_harmonic_restraints is gone.
    assert read_force_constants(simulation, COMPONENTS) == pytest.approx(
        {name: 0.0 for name in COMPONENTS}
    )


def test_constants_set_after_reinitialize_survive():
    """The fix: pushing the warm-up constants after the reinitialize sticks."""
    simulation, system, topology, positions = _minimized_state()

    system = remove_openmm_force(system, "k_")
    _add_restraints(system, positions, topology)
    simulation.context.reinitialize(preserveState=True)
    update_force_constants(simulation, {"protein_BB": 15.0, "protein_SC": 7.5})

    assert read_force_constants(simulation, COMPONENTS) == pytest.approx(
        {"protein_BB": 15.0, "protein_SC": 7.5}
    )

    # and the restraints actually contribute energy, which is the point
    simulation.context.setPositions(
        [Vec3(0.2 * i, 0.05, 0.0) for i in range(N_PARTICLES)] * openmmunit.nanometer
    )
    energy = simulation.context.getState(getEnergy=True, groups={_FG_COMPONENTS, _FG_COMPONENTS + 1}).getPotentialEnergy()
    assert energy.value_in_unit(openmmunit.kilojoule_per_mole) > 0.0


def test_checkpoint_captures_values_at_reinitialize_time():
    """The snapshot is taken when reinitialize is called, not at force construction.

    So an explicit push placed *before* the reinitialize survives it too. What is
    never recovered is the force's own constructor default: only an explicit push
    sets the value. ``run()`` pushes after the reinitialize because that placement
    does not depend on the outgoing context happening to carry the same parameter
    names, but either side is correct.
    """
    simulation, system, topology, positions = _minimized_state()

    system = remove_openmm_force(system, "k_")
    _add_restraints(system, positions, topology)
    update_force_constants(simulation, {"protein_BB": 15.0, "protein_SC": 7.5})
    simulation.context.reinitialize(preserveState=True)

    assert read_force_constants(simulation, COMPONENTS) == pytest.approx(
        {"protein_BB": 15.0, "protein_SC": 7.5}
    )


# --------------------------------------------------------------------------- #
# 2. protocol wiring
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_shipped_protocols_define_warmup_forces(protocol, tmp_path):
    """Shipped protocols must state the constants, not lean on the 15.0 fallback.

    The fallback exists only so protocols written before this key keep working;
    a shipped protocol that relied on it would be silently un-tunable.
    """
    with open(protocol) as f:
        raw = json.load(f)
    assert "forces" in raw["warmup"], f"{protocol}: 'warmup' has no explicit forces"

    equil = Equilibration(out_dir=str(tmp_path), protocol_fname=protocol, platform="CPU")
    assert equil.warmup_forces == raw["warmup"]["forces"]
    assert len(equil.warmup_forces) == len(equil.components_lookup)
    assert any(k > 0 for k in equil.warmup_forces)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_warmup_is_stiffer_than_first_equilibration_stage(protocol, tmp_path):
    """The ladder must never ramp up: warm-up >= equilibration Stage 1."""
    equil = Equilibration(out_dir=str(tmp_path), protocol_fname=protocol, platform="CPU")
    stage1 = equil.equilibration_scheme[0]["forces"]
    for name, warm, first in zip(equil.components_lookup, equil.warmup_forces, stage1):
        assert warm >= first, f"{protocol}: {name} warms up softer than Stage 1"


# --------------------------------------------------------------------------- #
# 3. the temperature ramp spends exactly the steps the protocol declares
# --------------------------------------------------------------------------- #


class _FakeIntegrator:
    def __init__(self):
        self.temperatures = []
        self.stepsize = None

    def setStepSize(self, stepsize):
        self.stepsize = stepsize

    def getStepSize(self):
        return self.stepsize

    def setTemperature(self, temperature):
        self.temperatures.append(temperature)


class _FakeContext:
    def reinitialize(self, preserveState=False):
        pass

    def setVelocitiesToTemperature(self, temperature):
        pass


class _FakeSimulation:
    def __init__(self):
        self.context = _FakeContext()
        self.steps = []

    def step(self, nsteps):
        assert nsteps >= 0
        self.steps.append(nsteps)


def _ramp(**kwargs):
    simulation, integrator = _FakeSimulation(), _FakeIntegrator()
    warm_up_system(simulation, integrator, **kwargs)
    return simulation, integrator


@pytest.mark.parametrize(
    "Tstart,Tend,Tstep,warming_steps",
    [
        (100, 310, 5, 100000),   # membrane protocol
        (50, 300, 5, 100000),    # soluble protocol
        (100, 310, 4, 100000),   # Tstep does not divide the range
        (100, 310, 5, 999),      # fewer steps than increments
        (300, 300, 5, 1000),     # degenerate: nothing to ramp
    ],
)
def test_ramp_spends_exactly_the_declared_steps(Tstart, Tend, Tstep, warming_steps):
    simulation, _ = _ramp(
        Tstart=Tstart, Tend=Tend, Tstep=Tstep, warming_steps=warming_steps
    )
    assert sum(simulation.steps) == warming_steps


@pytest.mark.parametrize("Tstep", [5, 4, 7])
def test_ramp_reaches_the_target_temperature_exactly(Tstep):
    _, integrator = _ramp(Tstart=100, Tend=310, Tstep=Tstep, warming_steps=10000)
    assert integrator.temperatures[0] == 100
    assert integrator.temperatures[-1] == 310
    assert integrator.temperatures == sorted(integrator.temperatures)
    assert max(integrator.temperatures) <= 310


def test_ramp_does_not_divide_by_zero_when_there_is_nothing_to_ramp():
    simulation, integrator = _ramp(Tstart=310, Tend=310, Tstep=5, warming_steps=500)
    assert sum(simulation.steps) == 500
    assert integrator.temperatures == [310]
