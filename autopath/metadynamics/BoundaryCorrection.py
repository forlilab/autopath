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
from functools import reduce

import numpy as np
import openmm.unit as unit
from openmm.app.metadynamics import Metadynamics
from scipy.special import erf, logsumexp


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
        super().__init__(system, variables, temperature, biasFactor, height, frequency,
                         saveFrequency=saveFrequency, biasDir=biasDir)

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
