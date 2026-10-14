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
from types import SimpleNamespace
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmm.app.metadynamics import Metadynamics
from scipy.special import erf, logsumexp

from .learned.sw import z_spline_values

MAX_CVS = 32                                        # OpenMM CustomCVForce limit


@dataclass(eq=False)
class LinearComposite:
    """Bias variable s = sum_k w_k (f_k - mu_k) / sd_k over feature forces f_k.

    Passed as a BiasVariable's force: the bias force takes the f_k directly as its
    collective variables (one level, no inner MetaCollectiveVariable). Composites
    with the same ``key`` share their features, which are then added once.
    ``clamp=(lo, hi)`` limits the value used by the bias to [lo, hi]; its gradient is zero outside.
    """
    forces: list
    weights: Any
    mu: Any
    sd: Any
    key: Any = None
    clamp: Any = None

    def __post_init__(self):
        self.weights, self.mu, self.sd = (np.asarray(a, dtype=float).ravel()
                                          for a in (self.weights, self.mu, self.sd))
        n = len(self.forces)
        if not (len(self.weights) == len(self.mu) == len(self.sd) == n):
            raise ValueError("LinearComposite needs one weight, mu and sd per feature force")
        if self.clamp is not None:
            self.clamp = (float(self.clamp[0]), float(self.clamp[1]))
            if not self.clamp[1] > self.clamp[0]:
                raise ValueError(f"clamp {self.clamp} needs hi > lo")

    def _clamped(self, v: float) -> float:
        return v if self.clamp is None else min(max(v, self.clamp[0]), self.clamp[1])

    def _clamped_expr(self, expr: str) -> str:
        return expr if self.clamp is None else f"min(max({expr}, {self.clamp[0]!r}), {self.clamp[1]!r})"

    def value(self, f) -> float:
        f = np.asarray(f, dtype=float)
        return self._clamped(float(np.sum(self.weights * (f - self.mu) / self.sd)))

    def expression(self, names) -> str:
        return self._clamped_expr(" + ".join(f"({float(w)!r})*({n} - ({float(m)!r}))/({float(s)!r})"
                                             for n, w, m, s in zip(names, self.weights, self.mu, self.sd)))


def _copy_force(f):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f))


@dataclass(eq=False)
class ResidualComposite(LinearComposite):
    """s = sum_k w_k [(f_k - mu_k) / sd_k - g_k(z)] over feature forces f_k and a z force.

    g_k is the natural cubic spline through ``tables[k]`` on linspace(z_min, z_max), with z
    clamped to that range. ``key`` is ignored: features are never shared.
    """
    z_force: Any = None
    z_min: float = 0.0
    z_max: float = 1.0
    tables: Any = None

    def __post_init__(self):
        super().__post_init__()
        if self.z_force is None:
            raise ValueError("ResidualComposite needs z_force, the z collective variable")
        self.z_min, self.z_max = float(self.z_min), float(self.z_max)
        if not self.z_max > self.z_min:
            raise ValueError(f"z_max ({self.z_max}) must exceed z_min ({self.z_min})")
        self.tables = np.atleast_2d(np.asarray(self.tables, dtype=float))
        if self.tables.shape[0] != len(self.forces) or self.tables.shape[1] < 4:
            raise ValueError("ResidualComposite needs one table row (>= 4 points) per feature force")

    def g(self, z):
        """Spline values: (n_feat,) for scalar z, (n_feat, len(z)) otherwise."""
        return z_spline_values(self.tables, self.z_min, self.z_max, z).T

    def value(self, f) -> float:
        """s from the feature values followed by z."""
        f = np.asarray(f, dtype=float)
        n = len(self.forces)
        return self._clamped(float(np.sum(self.weights * ((f[:n] - self.mu) / self.sd - self.g(f[n])))))

    def expression(self, names, tables) -> str:
        """names: feature CV names then the z CV name; tables: one tabulated-function name per feature."""
        *feats, zn = names
        zc = f"min(max({zn}, {self.z_min!r}), {self.z_max!r})"
        return self._clamped_expr(" + ".join(
            f"({float(w)!r})*(({n} - ({float(m)!r}))/({float(s)!r}) - {t}({zc}))"
            for n, t, w, m, s in zip(feats, tables, self.weights, self.mu, self.sd)))


def composite_bias_force(variables):
    """CustomCVForce table(s...) whose LinearComposite variables are expanded over their features.

    Returns (force, cv_map): cv_map[i] is ("cv", j) for a plain variable (CV index j) or
    ("lc", composite, [j...]) for a composite (indices of its feature CVs, then of its
    z CV for a ResidualComposite).
    """
    cvs, args, defs, cv_map, groups, tabs = [], [], [], [], {}, []
    for i, v in enumerate(variables):
        c = v.force
        if isinstance(c, LinearComposite):
            if v.periodic:
                raise ValueError("a LinearComposite bias variable cannot be periodic")
            if isinstance(c, ResidualComposite):
                idx = list(range(len(cvs), len(cvs) + len(c.forces) + 1))
                cvs += [(f"rf{i}_{k}", _copy_force(f)) for k, f in enumerate(c.forces)]
                cvs.append((f"rz{i}", _copy_force(c.z_force)))
                tn = [f"rt{i}_{k}" for k in range(len(c.forces))]
                tabs += [(n, mm.Continuous1DFunction([float(x) for x in row], c.z_min, c.z_max, False))
                         for n, row in zip(tn, c.tables)]
                expr = c.expression([cvs[j][0] for j in idx], tn)
            else:
                key = id(c) if c.key is None else c.key
                if key not in groups:
                    groups[key] = list(range(len(cvs), len(cvs) + len(c.forces)))
                    cvs += [(f"lf{len(groups) - 1}_{k}", _copy_force(f)) for k, f in enumerate(c.forces)]
                idx = groups[key]
                if len(idx) != len(c.forces):
                    raise ValueError("composites sharing a key must have the same number of features")
                expr = c.expression([cvs[j][0] for j in idx])
            defs.append(f"s{i} = {expr}")
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
    for name, f in tabs:
        force.addTabulatedFunction(name, f)
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


def _padded_variable(v, pad):
    """Copy of ``v`` widened by ``pad`` grid points each side and declared NON-periodic.

    OpenMM's Metadynamics refuses mixed periodic/non-periodic variables because
    Continuous{1,2,3}DFunction carries ONE periodic flag for the whole table, not one per
    axis (PLUMED's Grid keeps a per-dimension ``pbc_`` vector, which is why it has no such
    restriction). Padding sidesteps that: the axis is widened, hills are deposited with
    minimum-image wrapping over the TRUE period so the padding holds the exact periodic
    continuation, and the table is then declared non-periodic. The CV only ever reports
    values inside the original range, so the spline at either true edge has real data on
    both sides and both the bias and its derivative are continuous across the seam.

    Declaring the axis non-periodic WITHOUT padding is not equivalent: the endpoint values
    match but a one-sided spline gives mismatched derivatives, so the force jumps at the seam.
    """
    from openmm.app.metadynamics import BiasVariable
    dx = (v.maxValue - v.minValue) / (v.gridWidth - 1)
    return BiasVariable(v.force, v.minValue - pad * dx, v.maxValue + pad * dx,
                        v.biasWidth, periodic=False, gridWidth=v.gridWidth + 2 * pad)


def _pad_width(v, n_sigma=4.0):
    """Grid points of padding needed to carry a hill ``n_sigma`` past the true edge."""
    dx = (v.maxValue - v.minValue) / (v.gridWidth - 1)
    return int(np.ceil(n_sigma * v.biasWidth / dx))


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
        # Mixed periodicity: pad every periodic axis and present NON-periodic copies to
        # Metadynamics.__init__, which would otherwise raise. Only this case is padded; a
        # wholly periodic or wholly non-periodic set keeps OpenMM's native table untouched,
        # so existing runs and their saved biases are bit-for-bit unchanged.
        n_per = sum(bool(v.periodic) for v in variables)
        self._mixed = 0 < n_per < len(variables)
        self._pad = [(_pad_width(v) if v.periodic else 0) for v in variables] if self._mixed \
            else [0] * len(variables)
        if any(isinstance(v.force, LinearComposite) for v in variables):
            if self._mixed:
                raise ValueError(
                    "mixed periodic/non-periodic variables are not supported alongside a "
                    "LinearComposite CV: the composite path allocates its own bias arrays and "
                    "does not pad. Make the periodic variable non-periodic, or drop the "
                    "composite.")
            self._init_composite(system, variables, temperature, biasFactor, height, frequency,
                                 saveFrequency, biasDir)
        else:
            init_vars = [_padded_variable(v, p) if p else v
                         for v, p in zip(variables, self._pad)] if self._mixed else variables
            super().__init__(system, init_vars, temperature, biasFactor, height, frequency,
                             saveFrequency=saveFrequency, biasDir=biasDir)
            if self._mixed:
                self._padded_variables = init_vars
                self.variables = list(variables)      # restore TRUE periodicity for deposition
        if self._mixed:
            # boundary_factor must see the PADDED geometry but the TRUE periodicity. The
            # padded copies are marked non-periodic only so OpenMM will build the table; a
            # periodic axis has no edges, so correcting it amplifies hills at the seam (up to
            # 1/0.125 = 8x here) and injects a force spike exactly where the walker wraps.
            shim = [SimpleNamespace(periodic=v.periodic, gridWidth=pv.gridWidth,
                                    minValue=pv.minValue, maxValue=pv.maxValue,
                                    biasWidth=v.biasWidth)
                    for v, pv in zip(variables, getattr(self, "_padded_variables", variables))]
            self.boundaryFactor = boundary_factor(shim, self.bounds)

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

    def _axis_pad(self):
        """Padding per ARRAY axis. Bias arrays run in reversed variable order."""
        return list(reversed(self._pad))

    def _bias_view(self, arr):
        """The unpadded part of a bias array: what the FES, c(t) and disk all use.

        Every consumer goes through here rather than slicing itself, so the padding can only
        leak in one place instead of six.
        """
        if not self._mixed:
            return arr
        sl = tuple(slice(p, arr.shape[a] - p) if p else slice(None)
                   for a, p in enumerate(self._axis_pad()))
        return arr[sl]

    def _pad_from_view(self, arr):
        """Rebuild the padded array from an unpadded one by exact periodic extension.

        Used when loading another walker's bias: the file holds the unpadded view, and for a
        periodic axis the padding is just its periodic images, so nothing is lost.
        """
        if not self._mixed:
            return arr
        out = arr
        for a, p in enumerate(self._axis_pad()):
            if not p:
                continue
            n = out.shape[a]
            # grid point n-1 coincides with point 0, so the period is n-1 distinct points
            idx = (np.arange(-p, n + p) - 0) % (n - 1)
            out = np.take(out, idx, axis=a)
        return out

    def getFreeEnergy(self):
        """Free energy on the UNPADDED grid, so the shape matches the declared variables."""
        return (-((self.temperature + self._deltaT) / self._deltaT)
                * self._bias_view(self._totalBias) * unit.kilojoules_per_mole)

    def _syncWithDisk(self):
        """As Metadynamics, but the file holds the unpadded view.

        Keeping the on-disk shape equal to the declared grid means every existing reader of
        bias_*.npy keeps working unchanged; the padding is reconstructed on load.
        """
        if self.biasDir is None:
            return
        import os, re as _re
        from openmm.app.metadynamics import _LoadedBias
        oldName = os.path.join(self.biasDir, 'bias_%d_%d.npy' % (self._id, self._saveIndex))
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, 'temp_%d_%d.npy' % (self._id, self._saveIndex))
        fileName = os.path.join(self.biasDir, 'bias_%d_%d.npy' % (self._id, self._saveIndex))
        np.save(tempName, self._bias_view(self._selfBias))
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)
        fileLoaded = False
        pattern = _re.compile(r'bias_(.*)_(.*)\.npy')
        for filename in os.listdir(self.biasDir):
            match = pattern.match(filename)
            if match is None:
                continue
            matchId, matchIndex = int(match.group(1)), int(match.group(2))
            if matchId != self._id and (matchId not in self._loadedBiases
                                        or matchIndex > self._loadedBiases[matchId].index):
                try:
                    data = self._pad_from_view(np.load(os.path.join(self.biasDir, filename)))
                    self._loadedBiases[matchId] = _LoadedBias(matchId, matchIndex, data)
                    fileLoaded = True
                except IOError:
                    pass
        if fileLoaded:
            self._totalBias = np.copy(self._selfBias)
            for bias in self._loadedBiases.values():
                self._totalBias += bias.bias

    def _addGaussian(self, position, height, context):
        axisGaussians = []
        for i, v in enumerate(self.variables):
            if self._mixed and v.periodic:
                # Deposit on the PADDED axis, with minimum-image distance over the TRUE
                # period: the hill's periodic images then fill the padding exactly, which is
                # what makes the seam smooth once the table is declared non-periodic.
                pv = self._padded_variables[i]
                period = v.maxValue - v.minValue
                d = np.linspace(pv.minValue, pv.maxValue, pv.gridWidth) - position[i]
                d -= period * np.round(d / period)
                dist = np.abs(d) / period          # normalised to the ORIGINAL range
            else:
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
        a = self._bias_view(self._totalBias).ravel() / (kt * (self.biasFactor - 1.0))
        return kt * (logsumexp(self.biasFactor * a) - logsumexp(a))
