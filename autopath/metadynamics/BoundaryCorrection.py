"""McGovern–de Pablo boundary correction for grid metadynamics.

Near a hard edge part of every hill falls outside the sampled domain, so the
sum of hills there underestimates the bias (about half at a face). Each hill is
divided by N(s), the fraction of a Gaussian centred at s that lies inside the
domain (McGovern & de Pablo, J. Chem. Phys. 139, 084102, 2013). N is floored
smoothly below the in-domain minimum so the bias stays finite past the edge
(after Dama et al., J. Chem. Theory Comput. 11, 5638, 2015).

The factor depends only on s, so it is applied per hill: the self-bias files,
the shared total and getFreeEnergy() all carry the corrected bias.

The class also logs every hill and gives the bias at the walker and c(t)
(Tiwary & Parrinello, J. Phys. Chem. B 119, 736, 2015) for frame reweighting.
"""

import time
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmm.app.metadynamics import Metadynamics
from scipy.special import erf, logsumexp

MAX_CVS = 32                                        # OpenMM CustomCVForce limit


@dataclass(eq=False)
class LinearComposite:
    """Bias variable s = sum_k w_k (f_k - mu_k) / sd_k over feature forces f_k.

    Passed as a BiasVariable's force: the bias force takes the f_k directly as its
    collective variables (one level, no inner MetaCollectiveVariable). Composites
    with the same ``key`` share their features, which are then added once.
    """
    forces: list
    weights: Any
    mu: Any
    sd: Any
    key: Any = None

    def __post_init__(self):
        self.weights, self.mu, self.sd = (np.asarray(a, dtype=float).ravel()
                                          for a in (self.weights, self.mu, self.sd))
        n = len(self.forces)
        if not (len(self.weights) == len(self.mu) == len(self.sd) == n):
            raise ValueError("LinearComposite needs one weight, mu and sd per feature force")

    def value(self, f) -> float:
        f = np.asarray(f, dtype=float)
        return float(np.sum(self.weights * (f - self.mu) / self.sd))

    def expression(self, names) -> str:
        return " + ".join(f"({float(w)!r})*({n} - ({float(m)!r}))/({float(s)!r})"
                          for n, w, m, s in zip(names, self.weights, self.mu, self.sd))


def _copy_force(f):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f))


def composite_bias_force(variables):
    """CustomCVForce table(s...) whose LinearComposite variables are expanded over their features.

    Returns (force, cv_map): cv_map[i] is ("cv", j) for a plain variable (CV index j) or
    ("lc", composite, [j...]) for a composite (indices of its features).
    """
    cvs, args, defs, cv_map, groups = [], [], [], [], {}
    for i, v in enumerate(variables):
        c = v.force
        if isinstance(c, LinearComposite):
            if v.periodic:
                raise ValueError("a LinearComposite bias variable cannot be periodic")
            key = id(c) if c.key is None else c.key
            if key not in groups:
                g = len(groups)
                idx = list(range(len(cvs), len(cvs) + len(c.forces)))
                cvs += [(f"lf{g}_{k}", _copy_force(f)) for k, f in enumerate(c.forces)]
                groups[key] = idx
            idx = groups[key]
            if len(idx) != len(c.forces):
                raise ValueError("composites sharing a key must have the same number of features")
            defs.append(f"s{i} = {c.expression([cvs[j][0] for j in idx])}")
            args.append(f"s{i}")
            cv_map.append(("lc", c, idx))
        else:
            cv_map.append(("cv", len(cvs)))
            cvs.append((f"cv{i}", v.force))
            args.append(f"cv{i}")
    if len(cvs) > MAX_CVS:
        raise ValueError(f"{len(cvs)} collective variables exceed OpenMM CustomCVForce's limit of {MAX_CVS}")
    force = mm.CustomCVForce("; ".join([f"table({', '.join(args)})"] + defs))
    for name, f in cvs:
        force.addCollectiveVariable(name, f)
    return force, cv_map


def inside_fraction(grid, lo, hi, sigma) -> np.ndarray:
    """Fraction of a Gaussian (std ``sigma``) centred at each grid point inside [lo, hi]."""
    s = np.asarray(grid, dtype=float)
    k = np.sqrt(2.0) * sigma
    return 0.5 * (erf((hi - s) / k) - erf((lo - s) / k))


def smooth_floor(n, a) -> np.ndarray:
    """n for n >= a, else n + (a - n)^2 / (2a): C1 at a, a/2 at n = 0."""
    n = np.asarray(n, dtype=float)
    return np.where(n >= a, n, n + (a - n) ** 2 / (2.0 * a))


def boundary_factor(variables, bounds) -> np.ndarray:
    """1/f(N) on the bias grid (same axis order as Metadynamics._selfBias).

    ``bounds`` holds one (lo, hi) per variable in CV units, or None for no
    edge on that variable. Periodic variables are never corrected.
    """
    axes, n_edges = [], 0
    for v, b in zip(variables, bounds):
        if v.periodic or b is None:
            axes.append(np.ones(v.gridWidth))
            continue
        grid = np.linspace(v.minValue, v.maxValue, v.gridWidth)
        axes.append(inside_fraction(grid, b[0], b[1], v.biasWidth))
        n_edges += 1
    if n_edges == 0:
        return np.ones(tuple(v.gridWidth for v in reversed(variables)))
    n = axes[0] if len(axes) == 1 else reduce(np.multiply.outer, reversed(axes))
    # in-domain minimum: a corner where every corrected variable has an edge
    return 1.0 / smooth_floor(n, 0.5 ** n_edges)


class BoundaryCorrectedMetadynamics(Metadynamics):
    """OpenMM Metadynamics whose hills are divided by the in-domain fraction N(s).

    ``bounds=[None, ...]`` gives plain hills. ``hills`` holds one
    (step, cv..., height kJ/mol) row per deposited hill, ``hill_wall`` its
    time.time() at deposition.
    """

    def __init__(self, system, variables, temperature, biasFactor, height, frequency,
                 saveFrequency=None, biasDir=None, bounds=None):
        if bounds is None:
            bounds = [(v.minValue, v.maxValue) for v in variables]
        if len(bounds) != len(variables):
            raise ValueError("bounds needs one (lo, hi) or None per variable")
        self.bounds = list(bounds)
        self.boundaryFactor = boundary_factor(variables, bounds)
        self.hills = []
        self.hill_wall = []
        self._cv_map = None
        if any(isinstance(v.force, LinearComposite) for v in variables):
            self._init_composite(system, variables, temperature, biasFactor, height, frequency,
                                 saveFrequency, biasDir)
        else:
            super().__init__(system, variables, temperature, biasFactor, height, frequency,
                             saveFrequency=saveFrequency, biasDir=biasDir)

    def _init_composite(self, system, variables, temperature, biasFactor, height, frequency,
                        saveFrequency, biasDir):
        # Metadynamics.__init__ with the bias force from composite_bias_force
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(height):
            height = height * unit.kilojoules_per_mole
        if biasFactor <= 1.0:
            raise ValueError('biasFactor must be > 1')
        if (saveFrequency is None) != (biasDir is None):
            raise ValueError('Must specify both saveFrequency and biasDir')
        if saveFrequency is not None and (saveFrequency < frequency or saveFrequency % frequency != 0):
            raise ValueError('saveFrequency must be a multiple of frequency')
        if len(variables) not in (1, 2, 3):
            raise ValueError('Metadynamics requires 1, 2, or 3 collective variables')
        if any(v.periodic for v in variables):
            raise ValueError('composite bias variables cannot be mixed with periodic ones')
        self.variables = variables
        self.temperature = temperature
        self.biasFactor = biasFactor
        self.height = height
        self.frequency = frequency
        self.biasDir = biasDir
        self.saveFrequency = saveFrequency
        self._id = np.random.randint(0x7FFFFFFF)
        self._saveIndex = 0
        self._selfBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._totalBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._loadedBiases = {}
        self._syncWithDisk()
        self._deltaT = temperature * (biasFactor - 1)
        self._force, self._cv_map = composite_bias_force(variables)
        self._widths = [v.gridWidth for v in variables]
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        if len(variables) == 1:
            self._table = mm.Continuous1DFunction(self._totalBias.flatten(), *self._limits, False)
        elif len(variables) == 2:
            self._table = mm.Continuous2DFunction(*self._widths, self._totalBias.flatten(), *self._limits, False)
        else:
            self._table = mm.Continuous3DFunction(*self._widths, self._totalBias.flatten(), *self._limits, False)
        self._force.addTabulatedFunction('table', self._table)
        freeGroups = set(range(32)) - set(force.getForceGroup() for force in system.getForces())
        if len(freeGroups) == 0:
            raise RuntimeError('Cannot assign a force group to the metadynamics force. '
                               'The maximum number (32) of the force groups is already used.')
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _cv_values(self, context):
        """CV values at the walker; composites evaluated from their feature values."""
        vals = self._force.getCollectiveVariableValues(context)
        if self._cv_map is None:
            return vals
        return [vals[m[1]] if m[0] == "cv" else m[1].value([vals[j] for j in m[2]])
                for m in self._cv_map]

    def getCollectiveVariables(self, simulation):
        return self._cv_values(simulation.context)

    def step(self, simulation, steps):
        # Metadynamics.step with the CV values from _cv_values
        stepsToGo = steps
        forceGroup = self._force.getForceGroup()
        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.frequency == 0:
                nextSteps = min(nextSteps, self.frequency)
            else:
                nextSteps = min(nextSteps, self.frequency - simulation.currentStep % self.frequency)
            simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                position = self._cv_values(simulation.context)
                energy = simulation.context.getState(energy=True, groups={forceGroup}).getPotentialEnergy()
                height = self.height * np.exp(-energy / (unit.MOLAR_GAS_CONSTANT_R * self._deltaT))
                self._addGaussian(position, height, simulation.context)
            if self.saveFrequency is not None and simulation.currentStep % self.saveFrequency == 0:
                self._syncWithDisk()
            stepsToGo -= nextSteps

    def _addGaussian(self, position, height, context):
        axisGaussians = []
        for i, v in enumerate(self.variables):
            x = (position[i] - v.minValue) / (v.maxValue - v.minValue)
            if v.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0, 1.0, num=v.gridWidth) - x)
            if v.periodic:
                dist = np.min(np.array([dist, np.abs(dist - 1)]), axis=0)
                dist[-1] = dist[0]
            axisGaussians.append(np.exp(-0.5 * dist * dist / v._scaledVariance))
        if len(self.variables) == 1:
            gaussian = axisGaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axisGaussians))

        h = height.value_in_unit(unit.kilojoules_per_mole)
        self.hills.append((context.getStepCount(), *[float(p) for p in position], h))
        self.hill_wall.append(time.time())
        hill = h * gaussian * self.boundaryFactor
        self._selfBias += hill
        self._totalBias += hill
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._totalBias.flatten(), *self._limits)
        else:
            self._table.setFunctionParameters(*self._widths, self._totalBias.flatten(), *self._limits)
        self._force.updateParametersInContext(context)

    def bias_energy(self, simulation) -> float:
        """Current bias at the walker, V(s_t, t), in kJ/mol."""
        state = simulation.context.getState(getEnergy=True, groups={self._force.getForceGroup()})
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    def rct(self) -> float:
        """c(t) in kJ/mol: frame weights are exp[(V(s_t, t) - c(t)) / kT]."""
        kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoules_per_mole)
        a = self._totalBias.ravel() / (kt * (self.biasFactor - 1.0))
        return kt * (logsumexp(self.biasFactor * a) - logsumexp(a))
