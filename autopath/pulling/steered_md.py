import os
import math
import time
import logging
from glob import glob
from copy import deepcopy
from collections import deque
import numpy as np
from datetime import datetime

from autopath.utils import *
from autopath.customForces import (add_harmonic_restraints, remove_openmm_force)

from openmm.app import *
from openmm import XmlSerializer
import openmm.unit as openmmunit

import cvpack
import logging
logger = logging.getLogger("autopath")

try:
    from openmmtools.integrators import LangevinSplittingGirsanov
    from reweightingreporter import ReweightingReporter
except ImportError:
    girsanov = False
    logger.warning("Please install openmmtools to use Girsanov reweighting.")
    
class SteeredMD:
    """Steered molecular dynamics: pull a ligand along a COM-COM reaction
    coordinate using a moving harmonic restraint.

    Parameter design
    ----------------
    Three orthogonal knobs control a sweep:

    - ``max_displacement`` (nm): total RC range to traverse.
    - ``pulling_speed`` (nm/ps): physical pulling rate — varied across
      replicas in the sweep (passed to :meth:`run`).
    - ``dx_per_move`` (nm): RC grid spacing — how far the spring center
      r0 advances per Python-side update. Held fixed across the sweep.

    From these, the runtime derives::

        steps_per_move = round(dx_per_move / (pulling_speed * dt))
        sMD_moves      = max_displacement / dx_per_move
        total MD steps = max_displacement / (pulling_speed * dt)

    Note the total MD step count does **not** depend on ``dx_per_move``.
    ``dx_per_move`` is purely an *analysis-grid / log-resolution* knob: it
    controls how many rows are written to ``sMD_{run_id}.dat`` and the
    DCD frame cadence, but does not change the physics. Holding it
    constant across the sweep gives every speed the same protocol grid,
    which is what :class:`SMDAnalysis` expects when comparing speeds.

    Choosing ``dx_per_move``
    ------------------------
    Upper bound — keep it small vs. the thermal sigma of the spring,
    ``sigma_thermal = sqrt(kT/k)``. A rule of thumb is
    ``dx_per_move <= sigma_thermal / 10`` so the system sees an
    effectively continuous bias rather than discrete jumps. The
    constructor logs the ratio and warns if ``dx_per_move > 0.5 * sigma``.

    Lower bound — at the fastest sweep speed you don't want
    ``steps_per_move`` to clip; keep
    ``dx_per_move >= ~2 * v_max * dt`` (typically ~1e-4 nm with default
    timestep). Going smaller buys nothing physical and only inflates the
    .dat row count and per-move Python overhead. :meth:`run` warns when
    ``steps_per_move`` rounds below 1.5 or when the realized speed
    differs from the requested speed by more than 5%.

    Choosing ``sMD_spring_cte``
    ---------------------------
    Stiffer springs give tighter tracking but smaller ``sigma_thermal``,
    which in turn tightens the upper bound on ``dx_per_move`` (and
    eventually the lower bound on ``steps_per_move``). For a typical
    ligand the per-atom convention ``k = N_ha * k_per_atom`` (used in
    :mod:`autopath_core`) keeps the thermal sigma roughly invariant to
    ligand size.
    """

    def __init__(
        self,
        system: str = None,
        topology: str = None,
        groupA_atoms: list[int] = None,
        groupB_atoms: list[int] = None,
        restrained_atoms: list[int] = None,
        restart_velocities: bool = False,
        timestep: float = 0.004, #  # 4 fs timestep
        temperature: float = 300,
        autostop_nc: float | None = 0.1,  # fraction of NC_initial to stop at; None disables
        autostop_nc_window: int = 2,      # consecutive NC samples below threshold to confirm detachment
        autostop_min_displacement: float = 0.5,  # nm — don't fire before this displacement from r0
        sMD_max_r_offset: float = 3.0,   # max displacement offset (nm): cap pull at r0 + offset nm (also capped by PBC half-box minus pbc_safety_nm)
        pbc_safety_nm: float = 0.5,      # nm — safety margin subtracted from the per-axis PBC half-box cap (raise for large proteins)
        use_NVT: bool = False,  # Use NVT ensemble
        use_GReweighting: bool = False,
        out_dir: str = None,
        platform: str = "fastest",
        dx_per_move: float = 0.001,        # nm — RC grid spacing (held constant across speeds)
        max_displacement: float = 3.5,     # nm — total RC range
        sMD_spring_cte: float = 10000,     # kJ/mol/nm^2
        save_freq: int = 5,                # writes DCD every save_freq*steps_per_move
        force_n_samples: int = 10,         # cap on restraint-force samples averaged per move (>=1)
        force_sample_stride: int = 5,      # MD steps between force samples; n_samples adapts to move length
        verbose: int = 0,
        log_geom_features: bool | list[str] = False,  # log per-frame geom scalars to a sidecar
    ):

        self.system = system
        self.topology = topology
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.timestep = timestep * openmmunit.picoseconds
        self.temperature = temperature * openmmunit.kelvin
        self.groupA_atoms = groupA_atoms # ligand atoms
        self.groupB_atoms = groupB_atoms # pocket atoms
        self.restrained_atoms = restrained_atoms
        self.restart_velocities = restart_velocities

        self.verbose = verbose
        self.log_geom_features = log_geom_features
        self._shape_calc = None    # ShapeDescriptorCalculator, built once in run()
        self._lig_heavy_idx = None # ligand heavy-atom indices for shape descriptors
        self.autostop_nc        = float(autostop_nc) if autostop_nc is not None else None
        self.autostop_nc_window = int(autostop_nc_window)
        self.autostop_min_displacement = float(autostop_min_displacement)
        self.sMD_max_r_offset    = float(sMD_max_r_offset)
        self.pbc_safety_nm       = float(pbc_safety_nm)
        self.use_NVT = use_NVT

        self.integrator_friction = 1.0 / openmmunit.picoseconds  # Friction coefficient for Langevin integrator
        self.use_GReweighting = use_GReweighting
        if self.use_GReweighting:
            if not girsanov:
                logger.error("Disabled Girsanov reweighting because openmmtools is not installed.")
                self.use_GReweighting = False
            else:
                logger.info("Using Girsanov reweighting for steered MD.")

        self.platform = select_platform(platform)

        # Sweep-fixed parameters: identical for every replica/speed in a sweep,
        # so they live on the instance rather than on .run().
        self.max_displacement = float(max_displacement)                                                 # nm
        self.dx_per_move = float(dx_per_move) * openmmunit.nanometers                                   # Quantity
        self.sMD_spring_cte = float(sMD_spring_cte) * openmmunit.kilojoules_per_mole / openmmunit.nanometer**2
        self.save_freq = int(save_freq)
        self.force_n_samples = max(1, int(force_n_samples))
        self.force_sample_stride = max(1, int(force_sample_stride))
        self.sMD_moves = int(math.ceil(self.max_displacement / float(dx_per_move)))

        kB_kJ_per_mol_K = 0.0083144621
        T_K = self.temperature.value_in_unit(openmmunit.kelvin)
        k_spring = self.sMD_spring_cte.value_in_unit(
            openmmunit.kilojoules_per_mole / openmmunit.nanometer**2
        )
        sigma_thermal = math.sqrt(kB_kJ_per_mol_K * T_K / k_spring)  # nm
        sigma_over_dx = sigma_thermal / float(dx_per_move)
        logger.info(
            f"sMD spring: k={k_spring:.1f} kJ/mol/nm^2, sigma_thermal={sigma_thermal:.4f} nm, "
            f"dx_per_move={float(dx_per_move):.4g} nm (sigma/dx={sigma_over_dx:.1f}; rule of thumb >= 10)."
        )
        if float(dx_per_move) > 0.5 * sigma_thermal:
            logger.warning(
                f"dx_per_move ({dx_per_move:.4g} nm) exceeds 0.5*sigma_thermal "
                f"({0.5*sigma_thermal:.4g} nm) for k={k_spring:.1f} kJ/mol/nm^2, T={T_K:.1f} K. "
                f"Consider a stiffer spring or smaller dx_per_move."
            )

        self._params_logged_speeds: set = set()  # track which speeds already had the banner logged
        return None

    def pull_single_direction(self, 
                              simulation, 
                              run_id: int, 
                              direction: str = "forward",
                             ):
        """Main pulling loop for a single replica in one direction.

        Advances the spring centre r0 by ``dx_per_move`` each iteration and
        integrates ``steps_per_move`` MD steps. At each move the method records
        the within-move time-averaged restraint force ``force`` (with its SEM),
        the incremental protocol work ``dW_protocol`` (energy change due to the
        r0 shift *before* MD relaxation — distinct from the cumulative work),
        and, every ``save_freq`` moves, the soft contact count NC.

        NC autostop: the loop halts early when NC drops below
        ``autostop_nc × NC_initial`` for ``autostop_nc_window`` consecutive
        samples and the cumulative displacement exceeds
        ``autostop_min_displacement``.

        Output
        ------
        Writes ``{out_dir}/sMD_{run_id}.dat`` with a ``# key=value`` metadata
        header (spring_constant, requested/realized speed, steps_per_move,
        force_n_samples) followed by columns:
        step, time, r_target, r_before, r_after, force, force_sem,
        U_cvpack, dW_protocol

        ``force`` is the coherent quantity for the estimators: the within-move
        mean restraint force sampled from the dynamics under the current
        r_target (with ``force_sem`` its standard error). This is what feeds
        Feq / friction / the mean-force-TI PMF. The pre-step "kick"
        ``-k*(r_before - r_target)`` is intentionally NOT logged (it is the
        wrong ensemble for the mean force, and is trivially reconstructable
        from r_before, r_target and the header spring_constant if needed).

        The instantaneous lag ``r_target - r_after`` is likewise not logged: it
        is a finite-speed, single-endpoint quantity (equilibrium lag + friction
        drag γv/k + thermal noise), derivable from the two position columns. The
        coordinate deconvolution uses the equilibrium lag ``Feq/k`` instead.
        """
                    
        add_reporters(simulation, self.out_dir, f"sMD_{run_id}",
            total_steps=self.sMD_moves*self.steps_per_move, # total steps 
            logperiod=self.save_freq*self.steps_per_move, # steps
            verbose=0 #verbose level
        )

        if self.use_GReweighting:
            # NOTE: 'unperturebed' and 'firtsPertubation' are the canonical parameter names in
            # ReweightingReporter.__init__ (autopath/reweightingreporter.py) — the
            # misspellings are in the class definition itself and must be matched here.
            simulation.reporters.append(ReweightingReporter(f"{self.out_dir}/sMD_{run_id}.dat",
                                                            self.steps_per_move,
                                                            simulation.integrator,
                                                            unperturebed=True,
                                                            firtsPertubation=True,
                                                            ))

        # Set the initial r0 parameter
        initial_r0 = self.com_dist.getValue(simulation.context, allowReinitialization=False)
        logger.info(f"Initial COM distance: {initial_r0}")
        simulation.context.setParameter("r0_smd", initial_r0)

        # ── Move count cap: box safety + r0 factor ─────────────────────────────
        # PBC cap: cvpack uses pbc=True, so the COM-COM distance is the
        # minimum image. The scalar becomes ill-defined once any component of
        # the COM-COM vector exceeds L_axis/2. Per-axis limit on r:
        #     r_max_i = (L_i / 2) * |v| / |v_i|
        # The binding axis dominates; for a box elongated along the pull
        # direction this gives ~L_long/2, not the more conservative min(L)/2.
        # WARNING: minimum-image wrap (line below) is orthorhombic only.
        initial_r0_nm = initial_r0.value_in_unit(openmmunit.nanometers)
        dx_nm = float(self.dx_per_move.value_in_unit(openmmunit.nanometers))
        n_moves = self.sMD_moves
        if direction == "forward" and self.sMD_max_r_offset > 0:
            state    = simulation.context.getState(getPositions=True)
            box_vecs = state.getPeriodicBoxVectors()
            box_arr = np.array([
                [box_vecs[i][j].value_in_unit(openmmunit.nanometers) for j in range(3)]
                for i in range(3)
            ])
            off_diag_max = float(np.abs(box_arr - np.diag(np.diag(box_arr))).max())
            if off_diag_max > 1e-3:
                logger.warning(
                    f"PBC move-capping: non-orthorhombic box detected "
                    f"(max off-diagonal element = {off_diag_max:.4f} nm). "
                    f"Minimum-image wrapping assumes orthorhombic geometry and may be incorrect "
                    f"for triclinic boxes."
                )
            L = np.diag(box_arr)

            positions = state.getPositions(asNumpy=True).value_in_unit(openmmunit.nanometers)
            weigh_by_mass = not (len(self.groupA_atoms) == 1 or len(self.groupB_atoms) == 1)
            if weigh_by_mass:
                masses_A = np.array([simulation.system.getParticleMass(i).value_in_unit(openmmunit.dalton)
                                     for i in self.groupA_atoms])
                masses_B = np.array([simulation.system.getParticleMass(i).value_in_unit(openmmunit.dalton)
                                     for i in self.groupB_atoms])
            else:
                masses_A = np.ones(len(self.groupA_atoms))
                masses_B = np.ones(len(self.groupB_atoms))
            com_A = np.average(positions[self.groupA_atoms], axis=0, weights=masses_A)
            com_B = np.average(positions[self.groupB_atoms], axis=0, weights=masses_B)
            v = com_A - com_B
            v -= np.round(v / L) * L  # minimum-image (orthorhombic only; triclinic triggers a warning above)
            v_norm = float(np.linalg.norm(v))
            if abs(v_norm - initial_r0_nm) > 0.05:
                logger.warning(
                    f"PBC cap: |v|={v_norm:.3f} nm differs from cvpack initial_r0="
                    f"{initial_r0_nm:.3f} nm. Check that protein/ligand are not split "
                    f"across a periodic boundary at t=0; cap may be unreliable."
                )
            abs_v = np.abs(v)
            with np.errstate(divide='ignore', invalid='ignore'):
                r_max_per_axis = np.where(abs_v > 1e-6, (L / 2.0) * v_norm / abs_v, np.inf)
            r_max_pbc      = float(np.min(r_max_per_axis)) - self.pbc_safety_nm
            r_max_offset   = initial_r0_nm + self.sMD_max_r_offset
            r_max          = min(r_max_pbc, r_max_offset)
            moves_safe     = max(1, int((r_max - initial_r0_nm) / dx_nm))
            if n_moves > moves_safe:
                if r_max == r_max_pbc:
                    dom_axis = int(np.argmin(r_max_per_axis))
                    reason = (f"PBC axis-{('xyz')[dom_axis]} half-box "
                              f"(L={L[dom_axis]:.2f} nm, |v|/|v_{('xyz')[dom_axis]}|={v_norm/max(abs_v[dom_axis],1e-12):.2f}, "
                              f"safety={self.pbc_safety_nm:.2f} nm)")
                else:
                    reason = f"r0+{self.sMD_max_r_offset:.1f}nm offset"
                logger.warning(
                    f"Capping sMD_moves {n_moves} → {moves_safe}: planned r_end="
                    f"{initial_r0_nm + n_moves * dx_nm:.3f} nm exceeds {reason} limit "
                    f"({r_max:.3f} nm)."
                )
                n_moves = moves_safe

        # print ~10 progress lines per replica regardless of total move count
        print_interval = max(50, n_moves // 10)

        # ── Autostop state ─────────────────────────────────────────────────────
        # NC is the soft contact count (switching function 1/(1+(d/r0)^6))
        # sampled every save_freq moves. Autostop fires when NC falls below
        # autostop_nc × NC_initial for autostop_nc_window consecutive samples
        # after the min-displacement guard is satisfied.
        _nc_buf: deque[float] = deque(maxlen=self.autostop_nc_window)
        _nc_initial = None
        if self.autostop_nc is not None and self.subset_protein_CA is not None:
            pos_nm_init = simulation.context.getState(getPositions=True).getPositions(asNumpy=True) / openmmunit.nanometers
            _nc_initial = self._compute_nc(pos_nm_init)
            logger.info(
                f"NC initial: {_nc_initial:.2f}; "
                f"stop threshold: {self.autostop_nc * _nc_initial:.2f} "
                f"({self.autostop_nc:.0%} × NC_initial, window={self.autostop_nc_window})"
            )

        _buf: list[str] = []

        # ── Inline geom-feature sidecar (opt-in) ───────────────────────────────
        # Sampled at the same save_freq cadence as NC / DCD frames. Pocket-
        # dependent features (nc/mindist) are skipped if pocket atoms are absent.
        geom_feats = self._resolve_geom_features()
        geom_cols = [f for f in geom_feats
                     if f not in self._GEOM_POCKET or self.subset_protein_CA is not None]
        geom_dropped = set(geom_feats) - set(geom_cols)
        if geom_dropped:
            logger.warning(
                f"[geom] pocket atoms unavailable; skipping {sorted(geom_dropped)}."
            )
        geom_buf: list[str] = []
        geom_fh = None
        if geom_cols:
            geom_fh = open(f"{self.out_dir}/sMD_{run_id}_geom.dat", "w")
            geom_fh.write("step,time," + ",".join(f"geom_{c}" for c in geom_cols) + "\n")

        # Number of force samples averaged per move: adapt to the move length so
        # short (fast-speed) moves take few samples (low overhead) and long
        # (slow-speed) moves take up to the cap. Constant across a replica.
        n_sub = min(self.force_n_samples,
                    max(1, self.steps_per_move // self.force_sample_stride))

        # Per-replica constants recorded as `# key=value` comment lines. pandas
        # read_csv(comment='#') skips them; SMDData.parse_log_metadata reads them
        # so downstream estimators can use the logged k and realized speed
        # instead of re-deriving them.
        _k_spring_val = self.sMD_spring_cte.value_in_unit(
            openmmunit.kilojoules_per_mole / openmmunit.nanometer**2)
        with open(f"{self.out_dir}/sMD_{run_id}.dat", "w") as f:
            f.write(f"# spring_constant_kJ_mol_nm2={_k_spring_val:.10g}\n")
            f.write(f"# requested_speed_nm_per_ps={getattr(self, '_requested_speed', float('nan')):.10g}\n")
            f.write(f"# realized_speed_nm_per_ps={getattr(self, '_realized_speed', float('nan')):.10g}\n")
            f.write(f"# steps_per_move={self.steps_per_move}\n")
            f.write(f"# force_n_samples={n_sub}\n")
            f.write("step,time,r_target,r_before,r_after,force,force_sem,U_cvpack,dW_protocol\n")

            # Loop over the number of moves
            for i in range(n_moves):

                r_before = self.com_dist.getValue(simulation.context, allowReinitialization=False)
                time_before = simulation.context.getState().getTime().value_in_unit(openmmunit.picoseconds)
                U_pre_old = self.com_force.getValue(simulation.context, allowReinitialization=False)

                if direction == "backward":
                    r_target = initial_r0 - (i + 1) * self.dx_per_move
                    if r_target.value_in_unit(openmmunit.nanometers) <= 0.0:
                        logger.warning(f"Stopping backward pulling: r_target reached 0 at move {i}.")
                        break
                else:
                    r_target = initial_r0 + (i + 1) * self.dx_per_move

                simulation.context.setParameter("r0_smd", r_target)

                U_cvpack = self.com_force.getValue(simulation.context, allowReinitialization=False)
                dW_protocol = U_cvpack - U_pre_old

                # Integrate the move in n_sub chunks, sampling the restraint force over the
                # move so the logged `force` is the time-averaged pulling force (much lower
                # variance than a single pre-step sample -> cleaner Feq / friction / mean-force TI).
                # Splitting step(N) into chunks summing to N is trajectory-identical in OpenMM.
                _kunit = openmmunit.kilojoules_per_mole / openmmunit.nanometer
                base, rem = divmod(self.steps_per_move, n_sub)
                _fs = []
                _r_last = None
                for _j in range(n_sub):
                    simulation.step(base + (1 if _j < rem else 0))
                    _r_last = self.com_dist.getValue(simulation.context, allowReinitialization=False)
                    _fs.append((-self.sMD_spring_cte * (_r_last - r_target)).value_in_unit(_kunit))
                _fs = np.asarray(_fs, dtype=float)
                force_kjmnm = float(_fs.mean())
                force_sem   = float(_fs.std(ddof=1) / math.sqrt(_fs.size)) if _fs.size > 1 else 0.0

                r_after_nm  = _r_last.value_in_unit(openmmunit.nanometers)
                r_target_nm = r_target.value_in_unit(openmmunit.nanometers)
                r_before_nm = r_before.value_in_unit(openmmunit.nanometers)
                U_cvpack_kjm     = U_cvpack.value_in_unit(openmmunit.kilojoules_per_mole)
                dW_protocol_kjm  = dW_protocol.value_in_unit(openmmunit.kilojoules_per_mole)
                _row   = (f"{i},{time_before},{r_target_nm},{r_before_nm},"
                          f"{r_after_nm},{force_kjmnm},{force_sem},"
                          f"{U_cvpack_kjm},{dW_protocol_kjm}\n")

                # ── Sampled-frame diagnostics (every save_freq moves) ─────
                # One positions fetch shared by NC autostop and the geom sidecar.
                nc = None
                grow = None
                if i % self.save_freq == 0 and (self.subset_protein_CA is not None
                                                or geom_fh is not None):
                    pos_nm = simulation.context.getState(getPositions=True).getPositions(asNumpy=True) / openmmunit.nanometers
                    if self.subset_protein_CA is not None:
                        nc = self._compute_nc(pos_nm)
                    if geom_fh is not None:
                        grow = self._compute_geom_row(pos_nm)
                        geom_buf.append(f"{i},{time_before}," +
                                        ",".join(f"{grow[c]:.5f}" for c in geom_cols) + "\n")
                        if len(geom_buf) >= self.save_freq:
                            geom_fh.write(''.join(geom_buf)); geom_buf.clear()

                if self.autostop_nc is not None and nc is not None and _nc_initial is not None:
                    _nc_buf.append(nc)
                    current_displacement_nm = (i + 1) * dx_nm
                    nc_ready = (self.autostop_min_displacement <= 0.0 or
                                current_displacement_nm >= self.autostop_min_displacement)
                    if (nc_ready and len(_nc_buf) == self.autostop_nc_window and
                            all(v < self.autostop_nc * _nc_initial for v in _nc_buf)):
                        _buf.append(_row)
                        f.write(''.join(_buf))
                        _buf.clear()
                        logger.warning(
                            f"[Autostop/NC] Stopping at move {i}: NC={nc:.2f} "
                            f"< {self.autostop_nc:.0%} × NC_initial={_nc_initial:.2f} "
                            f"for {self.autostop_nc_window} consecutive samples "
                            f"(displacement={current_displacement_nm:.3f} nm)."
                        )
                        break

                # ── Progress reporting ─────────────────────────────────────
                if i % print_interval == 0:
                    lag_nm = r_target_nm - r_after_nm  # diagnostic only; not logged
                    if nc is not None and self.verbose > 0:
                        print(f"Move {i+1}/{n_moves}: r={r_after_nm:.3f} nm  lag={lag_nm:.4f} nm  NC={nc:.2f}")
                    else:
                        print(f"Move {i+1}/{n_moves}: r={r_after_nm:.3f} nm  lag={lag_nm:.4f} nm")

                _buf.append(_row)
                if len(_buf) >= self.save_freq:
                    f.write(''.join(_buf))
                    _buf.clear()

            if _buf:
                f.write(''.join(_buf))

        if geom_fh is not None:
            if geom_buf:
                geom_fh.write(''.join(geom_buf))
            geom_fh.close()

        # Save final positions
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors()) #saves correct box vectors to the pdb
        save_pdb(self.topology, final_positions, f"{self.out_dir}/sMD_{run_id}.pdb")
        return
    

    def run(self,
        pulling_speed: float = 0.001,           # nm/ps — varies per replica/call
        pulling_direction: str = "forward",
        run_id: str = None,
        checkpoint_file: str = None,
        pdb_file: str = None,
        state_xml_file: str = None,
    ):
        """Run one steered MD replica at the requested speed.

        Sweep-fixed parameters (dx_per_move, max_displacement, sMD_spring_cte,
        save_freq) are set in __init__. Only the speed-dependent bookkeeping
        (steps_per_move, realized_speed) is computed here.
        """

        if pulling_direction not in ["forward", "backward"]:
            raise ValueError("pulling_direction must be either 'forward' or 'backward'.")

        # ensure proper formating of the file name
        if run_id is None:
            timestamp = datetime.now().strftime("%H%M%S")
            run_id = f"replica-{timestamp}_v{pulling_speed}_{pulling_direction}"
        else:
            run_id = f"replica-{run_id}_v{pulling_speed}_{pulling_direction}"

        simulation_start_time = time.monotonic()

        # Speed-dependent derivation. dx_per_move and sMD_moves are set in __init__
        # (RC grid is held constant across speeds so the analysis-side protocol
        # grid is identical between replicas of different speeds).
        dt_ps = self.timestep.value_in_unit(openmmunit.picoseconds)
        dx_nm = self.dx_per_move.value_in_unit(openmmunit.nanometers)
        pulling_speed = float(pulling_speed)
        steps_per_move_float = dx_nm / pulling_speed / dt_ps
        self.steps_per_move = max(1, int(round(steps_per_move_float)))
        realized_speed = dx_nm / (self.steps_per_move * dt_ps)
        # Recorded in the .dat header so downstream estimators regress on the
        # realized pulling speed (steps_per_move rounding makes it differ from
        # the requested speed), not the nominal request.
        self._requested_speed = pulling_speed
        self._realized_speed = realized_speed

        if steps_per_move_float < 1.5:
            logger.warning(
                f"steps_per_move clipped to 1 (requested {steps_per_move_float:.3f}). "
                f"Speed {pulling_speed:g} nm/ps is too fast for dx_per_move={dx_nm:g} nm "
                f"and dt={dt_ps:g} ps; realized speed will be {realized_speed:g} nm/ps."
            )
        elif abs(realized_speed - pulling_speed) / pulling_speed > 0.05:
            logger.warning(
                f"Realized speed {realized_speed:.5g} nm/ps differs from "
                f"requested {pulling_speed:.5g} nm/ps by >5% due to rounding of "
                f"steps_per_move ({steps_per_move_float:.3f} -> {self.steps_per_move})."
            )

        ########################################################################################
        if pulling_speed not in self._params_logged_speeds:
            logger.info("#"*80)
            logger.info(f"Steered MD parameters (speed={pulling_speed} nm/ps):")
            logger.info(f"Pulling direction: {pulling_direction}")
            logger.info(f"Pulling speed: {pulling_speed} nm/ps (realized: {realized_speed:.5g} nm/ps)")
            logger.info(f"dx_per_move: {dx_nm:.4f} nm")
            logger.info(f"steps_per_move: {self.steps_per_move} steps")
            logger.info(f"Time per move: {self.steps_per_move * dt_ps:.3f} ps")
            logger.info(f"Max displacement: {self.max_displacement:.3f} nm, total sMD moves: {self.sMD_moves}")
            logger.info(f'Saving DCD every {self.save_freq*self.steps_per_move} steps')
            logger.info("#"*80)
            self._params_logged_speeds.add(pulling_speed)
        ########################################################################################
        
        if self.use_GReweighting:
            integrator = LangevinSplittingGirsanov(
                nstxout = self.steps_per_move,   
                temperature = self.temperature,
                collision_rate = self.integrator_friction,
                timestep = self.timestep,
                splitting = "R V O V R",        # ABOBA – reweightable
                constraint_tolerance = 1.0e-6,
            )
        else:
            # This is the default openMM but do not track work
            # dicussion https://github.com/openmm/openmm/issues/2520
            integrator = LangevinMiddleIntegrator(self.temperature, 
                                                self.integrator_friction, 
                                                self.timestep
                                                # constraint_tolerance = 1.0e-6
                                                )
                    
        system = deepcopy(self.system)  # Create a copy of the system to avoid modifying the original
        simulation = Simulation(self.topology, system, integrator, self.platform)

        #If the systems was equilibrated with a different integrator I get NaNs (even with same splitting)
        # so Im using the PDB instead of the checkpoint file
        if state_xml_file is not None:
            # Portable OpenMM State (positions+velocities+box, XML-serialized) —
            # unlike the binary .chk, this isn't tied to the Platform/GPU that
            # created it, so it loads safely on any node. Prefer this over
            # checkpoint_file when both are given.
            logger.info(f"Setting state from portable XML state file {state_xml_file}")
            with open(state_xml_file) as f:
                state = XmlSerializer.deserialize(f.read())
            # Restore positions/velocities/box only — not state.getParameters():
            # those include equilibration-only restraint globals (e.g. k_ligand)
            # that may no longer exist as Context parameters on the production
            # system, which setState() restores unconditionally and errors on.
            simulation.context.setPositions(state.getPositions())
            simulation.context.setVelocities(state.getVelocities())
            simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            simulation.integrator = integrator  # Replace the integrator with the new one
        elif checkpoint_file is None and pdb_file is None:
            logger.error("Either pdb_file, checkpoint_file, or state_xml_file must be provided to set initial positions.")
            exit(1)
        elif checkpoint_file is None and pdb_file is not None:
            logger.info(f"Setting positions from PDB file {pdb_file}")
            pdb = PDBFile(pdb_file)
            simulation.context.setPositions(pdb.getPositions())
            simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
            simulation.context.setVelocitiesToTemperature(self.temperature)

        elif checkpoint_file is not None and pdb_file is None:
            logger.info(f"Setting positions from checkpoint {checkpoint_file}")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator  # Replace the integrator with the new one
        else:
            # if both are provided, use the checkpoint file but warn the user
            logger.warning("Both checkpoint_file and pdb_file are provided. Using checkpoint_file.")
            simulation.loadCheckpoint(checkpoint_file)
            simulation.integrator = integrator

        # Add harmonic positional restraints to protein CA.
        input_positions = simulation.context.getState(getPositions=True).getPositions()
        if self.restrained_atoms is not None:
            add_harmonic_restraints(system, input_positions, 
                                    self.topology, self.restrained_atoms,
                                    restraint_force=100,
                                    force_name="k_CA",
                                    force_group=14,
                                )
            
        # system.setDefaultPeriodicBoxVectors(*PDBFile(pdb_file).topology.getPeriodicBoxVectors())
        # simulation.context.reinitialize(preserveState=True)
        
        # Pocket atoms needed for NC autostop, verbose monitoring, or geom
        # features that depend on the pocket (nc/mindist).
        geom_feats = self._resolve_geom_features()
        needs_pocket = (self.verbose > 0 or self.autostop_nc is not None
                        or bool(set(geom_feats) & self._GEOM_POCKET))
        if needs_pocket:
            self.subset_protein_CA, _ = self._get_pocket_atoms(simulation, cutoff=0.6)
        else:
            self.subset_protein_CA = None

        # Build the RDKit shape calculator once (elements-only, no SDF/bonds) for
        # any requested shape descriptors. Heavy atoms only, matching the
        # post-processing LigandFeatures convention. masses come from the elements.
        self._shape_calc = None
        self._lig_heavy_idx = None
        shape_feats = [f for f in geom_feats if f not in self._GEOM_POCKET]
        if shape_feats:
            from autopath.pulling.geom_kernel import ShapeDescriptorCalculator
            atoms = list(self.topology.atoms())
            self._lig_heavy_idx = [i for i in self.groupA_atoms
                                   if atoms[i].element is not None
                                   and atoms[i].element.symbol != "H"]
            elements = [atoms[i].element.symbol for i in self._lig_heavy_idx]
            self._shape_calc = ShapeDescriptorCalculator(elements, shape_feats)

        # Reset velocities to temperature. Check https://github.com/openmm/openmm/pull/259
        if self.restart_velocities:
            simulation.context.setVelocitiesToTemperature(self.temperature)
        
        if self.use_NVT:
            # Remove existing MonteCarloBarostat to run NVT
            system = remove_openmm_force(system, "MonteCarloBarostat")

        # Reinitialize the simulation context with the updated system
        simulation.context.reinitialize(preserveState=True)

        #run a super short simulation to ensure the system is stable after temp reset
        simulation.step(50/self.timestep.value_in_unit(openmmunit.picoseconds))  # 50 ps
        
        # try:
        #     simulation.integrator.reset()  # Reset the integrator. Only openmmtools integrators have this method    
        # except AttributeError:
        #     logger.warning("Integrator does not have reset method. This is expected for standard OpenMM integrators.")

        simulation.context.setTime(0)  # reset simulation time
        simulation.context.setStepCount(0)  # reset step count
        
        weighByMass = True
        if len(self.groupA_atoms) == 1 or len(self.groupB_atoms) == 1:
            weighByMass = False  # avoid problems with single DUM massless atom
        
        # Add COM force to the ligand and pocket groups with a harmonic potential shape
        groups = [self.groupA_atoms] + [self.groupB_atoms]
        self.com_force = cvpack.CentroidFunction(
            "0.5 * fc_pull * (distance(g1,g2)-r0_smd)^2",
            openmmunit.kilojoules_per_mole,  # energy not force
            groups,
            weighByMass=weighByMass,
            pbc=True,
        )
        
        self.com_force.addGlobalParameter("r0_smd", 0)
        self.com_force.addGlobalParameter('fc_pull', self.sMD_spring_cte)
        self.com_force.setForceGroup(1)  # Use a separate force group for the CV GROUP 1
        system.addForce(self.com_force)

        # Add COM distance function to measure the distance between the two groups
        self.com_dist = cvpack.CentroidFunction(
            "distance(g1,g2)",
            openmmunit.nanometers,  # distance not energy
            groups,
            weighByMass=weighByMass,
            pbc=True,
        )
        
        self.com_dist.setForceGroup(3)  # Use a separate force group for the CV GROUP 3
        system.addForce(self.com_dist)

        simulation.context.reinitialize(preserveState=True)

        # Run the actual pulling
        self.pull_single_direction(simulation, run_id, direction=pulling_direction)

        # Logging the total time for all replicas
        simulation_time = time.monotonic() - simulation_start_time
        logger.info(f"Finished {pulling_direction} sMD simulation in {simulation_time/60:.2f} min.")

        return run_id

    @staticmethod
    def adapt_speed_from_meff(m_eff_dalton, dx_target_nm=0.01, gammaL_ps=2.0,
                          v_min=0.0005, v_max=0.01):
        """Heuristically adapt the pulling speed from the ligand effective mass.

        Uses v = dx_target * gammaL / m_eff (Einstein–Smoluchowski scaling)
        then clamps the result to [v_min, v_max].  Useful for choosing a
        speed that keeps the ligand near equilibrium during the pull.
        """
        v = dx_target_nm * gammaL_ps / m_eff_dalton  # in nm/ps
        return min(max(v, v_min), v_max)
    
    # Bare feature names supported by the geom-logging sidecar.
    # Exit-direction (exit_1/2/3) = ligand->pocket COM unit vector projected onto
    # the pocket's principal axes (rotation-invariant); captures the unbinding
    # route, the strongest discriminator for rigid ligands where shape is flat.
    _GEOM_EXIT = ("exit_1", "exit_2", "exit_3")
    _GEOM_DEFAULT = ["nc", "mindist", "rog", "npr1", "npr2", "spherocity", "pbf",
                     "exit_1", "exit_2", "exit_3"]
    _GEOM_POCKET = {"nc", "mindist", "exit_1", "exit_2", "exit_3"}

    def _resolve_geom_features(self) -> list[str]:
        """Ordered list of geom feature names to log (empty when disabled)."""
        from autopath.pulling.geom_kernel import SHAPE_FEATURES
        if not self.log_geom_features:
            return []
        if self.log_geom_features is True:
            feats = list(self._GEOM_DEFAULT)
        else:
            feats = list(self.log_geom_features)
        allowed = set(SHAPE_FEATURES) | self._GEOM_POCKET
        bad = sorted(set(feats) - allowed)
        if bad:
            raise ValueError(
                f"Unsupported geom features {bad}. "
                f"Supported: {sorted(allowed)}"
            )
        return feats

    def _compute_geom_row(self, positions_nm) -> dict[str, float]:
        """Compute the requested geom scalars for one frame.

        positions_nm : (Natoms, 3) in nm. Shape descriptors need Angstrom, so
        ligand heavy-atom coords are scaled by 10 before the RDKit calculator;
        mindist stays in nm.
        """
        feats = self._resolve_geom_features()
        row: dict[str, float] = {}
        pocket_ok = self.subset_protein_CA is not None
        if "nc" in feats and pocket_ok:
            row["nc"] = self._compute_nc(positions_nm)
        if "mindist" in feats and pocket_ok:
            lig = positions_nm[self.groupA_atoms]
            poc = positions_nm[self.subset_protein_CA]
            diff = lig[:, np.newaxis, :] - poc[np.newaxis, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            row["mindist"] = float(np.sqrt(d2.min()))
        if any(f in feats for f in self._GEOM_EXIT) and pocket_ok:
            e = self._exit_direction(positions_nm)
            if e is not None:
                row["exit_1"], row["exit_2"], row["exit_3"] = e
        if self._shape_calc is not None:
            lig_ang = positions_nm[self._lig_heavy_idx] * 10.0
            row.update(self._shape_calc.compute(lig_ang))
        return row

    def _exit_direction(self, positions_nm):
        """Ligand->pocket COM unit vector in the pocket's principal-axis frame.

        Returns (e1, e2, e3), the components of the unit exit vector projected
        onto the pocket heavy-atom principal axes. Rotation/translation
        invariant (the axes co-rotate with the pocket), so it is comparable
        across replicas even without global alignment. Direction only — the
        magnitude (~r) is already captured by r_before. Returns None if the
        ligand and pocket COMs coincide.
        """
        lig = positions_nm[self.groupA_atoms]
        poc = positions_nm[self.subset_protein_CA]
        cA = lig.mean(0)
        cB = poc.mean(0)
        v = cA - cB
        nrm = float(np.sqrt(v @ v))
        if nrm < 1e-9:
            return None
        u = v / nrm
        Xp = poc - cB
        _, V = np.linalg.eigh(Xp.T @ Xp)      # columns: pocket principal axes
        # Deterministic eigenvector signs (svd_flip convention: largest-|loading|
        # component positive) so projections are comparable frame-to-frame.
        for j in range(3):
            k = int(np.argmax(np.abs(V[:, j])))
            if V[k, j] < 0:
                V[:, j] = -V[:, j]
        e = u @ V
        return float(e[0]), float(e[1]), float(e[2])

    def _compute_nc(self, positions_nm, threshold_nm=0.5):
        """Compute number of contacts via switching function 1/(1+(d/threshold)^6) using positions only.
        Similar to what CVPack does.

        Uses squared distances: 1/(1+(d/r0)^6) = 1/(1+(d^2/r0^2)^3), so the
        sqrt inside the norm is skipped (the result is immediately raised to
        the 6th power). Numerically identical to the norm form.
        """
        lig_pos = positions_nm[self.groupA_atoms]
        pocket_pos = positions_nm[self.subset_protein_CA]
        diff = lig_pos[:, np.newaxis, :] - pocket_pos[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        x2 = d2 / (threshold_nm * threshold_nm)
        return float(np.sum(1.0 / (1.0 + x2**3)))

    def _get_pocket_atoms(self, simulation, cutoff=0.6):
        """Return protein heavy atoms within *cutoff* nm of any ligand atom.

        Solvent, ions, and unknown residues are excluded. Used both for
        verbose NC monitoring and for the NC-based autostop criterion.

        Returns
        -------
        subset_protein_CA : np.ndarray of int
            Atom indices of the pocket alpha carbons within cutoff of the ligand.
        subset_protein_residues : list of Residue
            Corresponding residue objects.
        """

        protein_atoms = [atom for atom in self.topology.atoms() if atom.residue.name not in ["HOH", "WAT", "SOL", "NA", "CL", "K","MG", 'UNK']]
        # protein_HA = [atom.index for atom in protein_atoms if atom.element.symbol != "H"]  # Exclude hydrogens

        # Only alpha carbons. The element check rejects Ca2+ ions, which are also named "CA".
        protein_CA = np.array([atom.index for atom in self.topology.atoms()
                               if atom.name == "CA" and atom.element is not None
                               and atom.element.symbol == "C"], dtype=int)

        # Find a subset of protein_CA that are cutoff nm away from groupA_atoms
        state = simulation.context.getState(getPositions=True, getVelocities=False)
        positions = state.getPositions(asNumpy=True) / openmmunit.nanometers
        ligand_pos = positions[self.groupA_atoms]
        pocket_pos = positions[protein_CA]

        diff = ligand_pos[:, np.newaxis, :] - pocket_pos[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        close_indices = np.any(d2 < cutoff * cutoff, axis=0)
        subset_protein_CA = protein_CA[close_indices]
        _subset = set(subset_protein_CA.tolist())
        subset_protein_residues = [atom.residue for atom in protein_atoms if atom.index in _subset]

        return subset_protein_CA, subset_protein_residues

