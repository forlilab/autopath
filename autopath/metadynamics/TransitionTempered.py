"""Transition-tempered metadynamics (Dama et al., J. Chem. Theory Comput. 10, 3626, 2014).

Hills keep their full height until the bias links a set of basin points, then
shrink as h exp(-max(V* - V_th, 0) / ΔT), where V* is the largest bias level
at which all basins are still connected through grid cells with V >= V*
(the minimum bias on the maximally biased path, Eq. 1). ΔT = (γ - 1) T as in
well-tempered metadynamics; the bias converges to -F.

With ``ttBiasFactor`` the local well-tempered factor exp(-V(s)/((γ - 1) kT)) is
kept and TT multiplies on top with its own factor (PLUMED BIASFACTOR + TTBIASFACTOR);
the bias then converges to the well-tempered -(γ - 1)/γ F.
"""

import numpy as np
import openmm.unit as unit
from scipy import ndimage
from scipy.special import logsumexp

from .BoundaryCorrection import BoundaryCorrectedMetadynamics


def _linked(mask, basins, periodic):
    labels, _ = ndimage.label(mask)
    if any(labels[b] == 0 for b in basins):
        return False
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for ax, per in enumerate(periodic):
        if not per:
            continue
        lo = np.take(labels, 0, axis=ax).ravel()
        hi = np.take(labels, -1, axis=ax).ravel()
        for a, b in zip(lo, hi):
            if a and b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    roots = {find(labels[b]) for b in basins}
    return len(roots) == 1


def vstar(bias, basins, periodic=None, lower=None) -> float:
    """Minimum bias on the maximally biased path linking all ``basins``.

    ``bias`` is an N-D grid, ``basins`` a list of index tuples into it and
    ``periodic`` one flag per grid axis (face-adjacent cells are neighbours).
    ``lower`` is a level known to link the basins (the previous V*: the bias
    only grows); it narrows the search. Exact: the result is a grid value.
    """
    bias = np.asarray(bias, dtype=float)
    basins = [tuple(int(i) for i in np.atleast_1d(b)) for b in basins]
    periodic = [False] * bias.ndim if periodic is None else list(periodic)
    top = min(bias[b] for b in basins)
    if _linked(bias >= top, basins, periodic):
        return float(top)
    lo = float(bias.min())                          # every cell: always linked
    if lower is not None and lo < lower < top and _linked(bias >= lower, basins, periodic):
        lo = float(lower)
    hi = float(top)                                 # invariant: linked at lo, not at hi
    for _ in range(60):                             # ties can keep the count high: cap it
        if np.count_nonzero((bias >= lo) & (bias < hi)) <= 64:
            break
        mid = 0.5 * (lo + hi)
        if _linked(bias >= mid, basins, periodic):
            lo = mid
        else:
            hi = mid
    levels = np.unique(bias[(bias >= lo) & (bias < hi)])
    i, j = 0, len(levels) - 1                       # levels[0] links: same set as level lo
    while i < j:
        mid = (i + j + 1) // 2
        if _linked(bias >= levels[mid], basins, periodic):
            i = mid
        else:
            j = mid - 1
    return float(levels[i])


class TransitionTemperedMetadynamics(BoundaryCorrectedMetadynamics):
    """Boundary-corrected metadynamics with transition-tempered hill heights.

    ``basins`` holds one CV point (in CV units, variable order) per basin;
    ``threshold`` is V_th in kJ/mol. ``vstar_log`` holds V* at every hill.
    ``ttBiasFactor`` keeps well tempering with ``biasFactor`` and tempers by V*
    with ``ttBiasFactor``; None (default) is pure TT with ``biasFactor``.
    """

    def __init__(self, system, variables, temperature, biasFactor, height, frequency,
                 saveFrequency=None, biasDir=None, bounds=None, basins=None, threshold=0.0,
                 ttBiasFactor=None):
        if not basins or len(basins) < 2:
            raise ValueError("transition tempering needs at least two basin points")
        super().__init__(system, variables, temperature, biasFactor, height, frequency,
                         saveFrequency=saveFrequency, biasDir=biasDir, bounds=bounds)
        self.basin_points = [list(np.atleast_1d(np.asarray(b, dtype=float))) for b in basins]
        self.basins = [self._grid_index(b) for b in basins]
        self.ttBiasFactor = ttBiasFactor
        self.threshold = float(threshold)
        self.vstar_log = []
        self.wellTempered = ttBiasFactor is not None
        if self.wellTempered and ttBiasFactor <= 1.0:
            raise ValueError("ttBiasFactor must be > 1")
        kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoules_per_mole)
        self._kdt_wt = kt * (biasFactor - 1.0)
        self._kdt = kt * ((ttBiasFactor if self.wellTempered else biasFactor) - 1.0)

    def _grid_index(self, point):
        point = np.atleast_1d(np.asarray(point, dtype=float))
        if len(point) != len(self.variables):
            raise ValueError(f"basin {list(point)} needs {len(self.variables)} coordinates")
        idx = []
        for x, v in zip(point, self.variables):
            if not v.periodic and not v.minValue <= x <= v.maxValue:
                raise ValueError(f"basin coordinate {x} outside the grid [{v.minValue}, {v.maxValue}]")
            f = (x - v.minValue) / (v.maxValue - v.minValue)
            idx.append(int(round((f % 1.0 if v.periodic else f) * (v.gridWidth - 1))))
        return tuple(reversed(idx))                 # grid axes run in reverse variable order

    def vstar(self) -> float:
        return vstar(self._totalBias, self.basins, [v.periodic for v in reversed(self.variables)],
                     lower=self.vstar_log[-1] if self.vstar_log else None)

    def step(self, simulation, steps):
        stepsToGo = steps
        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.frequency == 0:
                nextSteps = min(nextSteps, self.frequency)
            else:
                nextSteps = min(nextSteps, self.frequency - simulation.currentStep % self.frequency)
            simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                position = self._force.getCollectiveVariableValues(simulation.context)
                v = self.vstar()
                self.vstar_log.append(v)
                height = self.height * np.exp(-max(v - self.threshold, 0.0) / self._kdt)
                if self.wellTempered:
                    e = simulation.context.getState(
                        getEnergy=True, groups={self._force.getForceGroup()}).getPotentialEnergy()
                    height = height * np.exp(-e.value_in_unit(unit.kilojoules_per_mole) / self._kdt_wt)
                self._addGaussian(position, height, simulation.context)
            if self.saveFrequency is not None and simulation.currentStep % self.saveFrequency == 0:
                self._syncWithDisk()
            stepsToGo -= nextSteps

    def getFreeEnergy(self):
        if self.wellTempered:
            return super().getFreeEnergy()
        return -self._totalBias * unit.kilojoules_per_mole

    def rct(self) -> float:
        """c(t) for a bias converging to -F: kT ln <exp(V/kT)> over the grid."""
        if self.wellTempered:
            return super().rct()
        kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoules_per_mole)
        v = self._totalBias.ravel() / kt
        return kt * (logsumexp(v) - np.log(v.size))
