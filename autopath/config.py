import os
import json
import logging
from inspect import signature


class Config(object):
    """Configuration container for an AutoPath simulation.

    Reads and validates settings from a JSON file (via :meth:`from_config`) or
    accepts keyword arguments directly.  Every parameter name and default here
    is kept in sync with :class:`autopath.autopath_core.AutoPath` so that a
    ``Config`` instance can be used to construct an ``AutoPath`` run.
    """

    def __init__(
        self,
        VS_mode: bool = False,
        pdb_path: str = None,
        do_fix_pdb: bool = True,
        pocket_selection: str | list = "same residue as protein and (around 4 resname UNK) and (not name H*)",
        temperature: float = 300,
        random_state: int = 42,
        platform: str = "fastest",
        run_preparation: bool = True,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3pfb_HFE_multivalent.xml",
        ],
        hydrogenMass: float = 1.5,  # amu
        timestep: float = 0.004,  # ps
        lig_ff: str = "openff-2.3.0",
        boxShape: str = "dodecahedron",
        padding: float = 1.2,
        ionicStrength: float = 0.15,
        ions: tuple = ("Na+", "Cl-"),
        variants: dict = None,
        is_membrane: bool = False,
        lipid_type: str = None,
        run_equilibration: bool = True,
        protocol_fname: str = None,
        run_sMDpulling: bool = True,
        sMD_outdir: str = "sMD",
        sMD_pulling_dir: str = "forward",  # "forward" or "backward"
        sMD_pulling_speeds: dict = {0.005: 5, 0.0025: 5, 0.001: 5},  # nm/ps; value = min reps (floor) if sMD_converge_speeds, else total reps
        sMD_max_pulling_dist: float = 3.5,  # nm
        sMD_max_r_offset: float = 3.0,  # max displacement offset (nm): cap pull at r0 + offset nm (also capped at half-box - 0.5 nm)
        sMD_autostop_nc: float = 0.01,  # fraction of NC_initial; None disables
        sMD_autostop_nc_window: int = 5,
        sMD_autostop_min_displacement: float = 0.5,
        sMD_converge_speeds: bool = True,
        sMD_conv_window: int = 5,
        sMD_conv_streak: int = 3,
        sMD_autostop_estimator: str = "cumulant",
        sMD_alternate_speeds: bool = False,
        sMD_time: int = None,  # ns
        sMD_steps_per_move: int = None,
        sMD_dx_per_move: float = 0.001,  # nm
        sMD_spring_cte: float = None,  # KJ/mol/nm2
        sMD_force_n_samples: int = 10,  # cap on restraint-force samples time-averaged per move
        sMD_force_sample_stride: int = 5,  # MD steps between force samples
        sMD_ligand_anchor_mode: str = "murcko",
        sMD_max_replicas: int = 50,
        sMD_run_analysis: bool = True,
        sMD_clust_selection: str = None,
        sMD_path_model: str = "dtw",  # "dtw" | "null" ("null" = no clustering: one path)
        sMD_n_paths: int = None,  # fixed number of paths; None = silhouette-selected
        sMD_max_frac_neg_dG_first_half: float = 0.25,  # pass-3 binding-well filter; <=0 disables
        sMD_features: list = None,
        sMD_log_geom_features: bool = True,  # log geom features during pulling + merge into clustering
        sMD_plateau_frac: float = 0.4,  # force-plateau TS boundary: fraction of peak |force|
        sMD_cluster_to_boundary: bool = True,  # cluster + RMSD-converge only up to the force-plateau boundary
        sMD_boundary_buffer_frac: float = 0.1,  # extend the boundary cap by this fraction of r_ts
        cluster_across_speeds: bool = False,
        sMD_min_replicas_per_path: int = 5,
        extract_milestones: bool = True,
        milestone_mode: str = "all_medoids",  # "per_path" or "all_medoids"
        milestone_min_frame_separation: int = 0,
        n_milestones: int = 5,
        relax_steps: int = 25000,
        run_metadynamics: bool = True,
        mMD_use_funnel_potential: bool = False,
        mMD_bias_factor: int = 15,
        mMD_bias_frequency: int = 2,  # ps
        mMD_hill_height: float = 1.2,  # kJ/mol
        mMD_hill_width: float = 0.05,
        mMD_time: int = 5,  # ns
        mMD_milestone_seeding: bool = True,
        mMD_multiple_walkers: bool = False,
        mMD_preseed_bias: bool = False,
        mMD_preseed_speed: float = None,
        mMD_funnel_host_selection: str = None,
    ):
        """Initialise an AutoPath configuration.

        Parameters
        ----------
        General
        ~~~~~~~
        VS_mode : bool, optional
            Enable virtual-screening mode.  Activates equilibration and pulling
            checkpoints to terminate early if the ligand drifts from the binding
            site.  Default ``False``.
        pdb_path : str, optional
            Path to the input protein-ligand PDB file.
        do_fix_pdb : bool, optional
            Run PDB preprocessing (capping termini, adding missing atoms, setting
            pH 7.4 protonation states) before building the system.  Default
            ``True``.
        pocket_selection : str or list of int, optional
            Defines the binding-site atoms used for COM distances and pocket
            features.  Two forms are accepted:

            * **list of int** — residue IDs in the *original PDB* numbering.
              AutoPath builds ``{sys_name}/residue_mapping.json`` and translates
              to ``system.pdb`` sequential IDs automatically::

                pocket_selection=[133, 156, 189, 226]   # original PDB resids

            * **str** — MDAnalysis selection applied directly to ``system.pdb``
              (sequential numbering after PDB preprocessing)::

                pocket_selection='resid 114 137 170 and name CA'

            Default selects non-hydrogen protein residues within 4 Å of
            resname UNK.
        temperature : float, optional
            Simulation temperature in Kelvin.  Default ``300``.
        random_state : int, optional
            Global random seed for reproducible clustering and MD restarts.
            Default ``42``.
        platform : str, optional
            OpenMM platform name: ``"fastest"``, ``"CUDA"``, ``"OpenCL"``, or
            ``"CPU"``.  Default ``"fastest"``.

        System preparation
        ~~~~~~~~~~~~~~~~~~
        run_preparation : bool, optional
            Build and solvate the system.  Set ``False`` to reuse an existing
            ``system.xml`` / ``system.pdb``.  Default ``True``.
        forcefield : list of str, optional
            OpenMM XML force-field files.  Default AMBER14 + TIP3P-FB.
        hydrogenMass : float, optional
            Hydrogen mass repartitioning target in amu; enables 4 fs timestep.
            Default ``1.5``.
        timestep : float, optional
            Integration timestep in ps.  Default ``0.004``.
        lig_ff : str, optional
            Ligand force field, given as ``"<family>-<version>"``: ``"openff-*"``,
            ``"espaloma-*"``, or ``"gaff-*"`` (e.g. ``"espaloma-0.3.2"``,
            ``"gaff-2.11"``).  Default ``"openff-2.3.0"``.
        boxShape : str, optional
            Simulation box shape: ``"dodecahedron"`` or ``"cube"``.
            Default ``"dodecahedron"``.
        padding : float, optional
            Minimum distance (nm) between solute and box face.  Default ``1.2``.
        ionicStrength : float, optional
            Target ionic strength in mol/L.  Default ``0.15``.
        ions : tuple of str, optional
            Ion species (positive, negative) for salting.  Default
            ``("Na+", "Cl-")``.
        variants : dict, optional
            Protonation-state variants passed to ``SystemPreparation``.
        is_membrane : bool, optional
            Enable membrane-protein mode (changes box builder and equilibration
            protocol).  Default ``False``.
        lipid_type : str, optional
            Lipid residue name when ``is_membrane=True`` (e.g. ``"POPC"``), or a
            path to a custom lipid patch PDB.

        Equilibration
        ~~~~~~~~~~~~~
        run_equilibration : bool, optional
            Run the equilibration protocol.  Default ``True``.
        protocol_fname : str, optional
            Path to a custom equilibration protocol JSON file.  ``None`` uses the
            built-in default protocol.

        Steered MD (sMD)
        ~~~~~~~~~~~~~~~~
        run_sMDpulling : bool, optional
            Run the steered-MD pulling simulations.  Default ``True``.
        sMD_outdir : str, optional
            Subdirectory name (relative to the system directory) for sMD output.
            Default ``"sMD"``.
        sMD_pulling_dir : str, optional
            Pulling direction: ``"forward"`` (ligand pulled out) or
            ``"backward"``.  Default ``"forward"``.
        sMD_pulling_speeds : dict, optional
            Mapping of pulling speed (nm/ps) to replica count. The count is a
            minimum-replicas floor when ``sMD_converge_speeds=True``, or the
            total replica count otherwise.  Default
            ``{0.005: 5, 0.0025: 5, 0.001: 5}``.
        sMD_max_pulling_dist : float, optional
            Maximum COM displacement from the binding site in nm.  Default
            ``3.5``.
        sMD_max_r_offset : float, optional
            Maximum pull offset from the starting COM distance (nm); also capped
            at half-box minus 0.5 nm.  Default ``3.0``.
        sMD_autostop_nc : float, optional
            Stop a replica once the ligand-pocket native-contact count drops
            below this fraction of its initial value.  ``None`` disables the
            check.  Default ``0.01``.
        sMD_autostop_nc_window : int, optional
            Number of trailing samples averaged before evaluating the native-
            contact autostop condition.  Default ``5``.
        sMD_autostop_min_displacement : float, optional
            Minimum COM displacement (nm) required before autostop is evaluated.
            Default ``0.5``.
        sMD_converge_speeds : bool, optional
            Keep adding replicas per speed (up to ``sMD_max_replicas``) until the
            convergence criterion is met, instead of running a fixed replica
            count.  Default ``True``.
        sMD_conv_window : int, optional
            Number of trailing replicas used to evaluate convergence.  Default
            ``5``.
        sMD_conv_streak : int, optional
            Number of consecutive converged windows required before stopping.
            Default ``3``.
        sMD_autostop_estimator : str, optional
            Free-energy estimator used to judge convergence: ``"cumulant"``,
            ``"jarzynski"``, or ``"force"``.  Default ``"cumulant"``.
        sMD_alternate_speeds : bool, optional
            Round-robin across pulling speeds while accumulating replicas,
            instead of finishing one speed before starting the next.  Default
            ``False``.
        sMD_time : int, optional
            Maximum pulling time in ns (overrides step-count calculation when
            set).  Default ``None``.
        sMD_steps_per_move : int, optional
            Number of MD steps per pull increment (overrides ``sMD_dx_per_move``
            when set).  Default ``None``.
        sMD_dx_per_move : float, optional
            COM displacement increment per move in nm.  Default ``0.001``.
        sMD_spring_cte : float, optional
            Pulling spring constant in kJ/mol/nm^2.  ``None`` auto-scales by
            ligand heavy-atom count.  Default ``None``.
        sMD_force_n_samples : int, optional
            Cap on the number of restraint-force samples time-averaged per pull
            move (``1`` reproduces the legacy single pre-step sample).  Default
            ``10``.
        sMD_force_sample_stride : int, optional
            MD steps between force samples within a move; the number of samples
            actually taken adapts to the move length, capped by
            ``sMD_force_n_samples``.  Default ``5``.
        sMD_ligand_anchor_mode : str, optional
            Strategy for selecting ligand anchor atoms for the COM pulling
            coordinate: ``"lig_ha"`` (all heavy atoms) or ``"murcko"`` (Murcko
            scaffold).  Default ``"murcko"``.
        sMD_max_replicas : int, optional
            Hard cap on replicas per speed to prevent infinite convergence loops.
            Default ``50``.
        sMD_run_analysis : bool, optional
            Run path clustering and PMF estimation after pulling.  Default
            ``True``.
        sMD_clust_selection : str, optional
            MDAnalysis selection for clustering features (in addition to ligand
            COM distance).  ``None`` uses only ligand COM.  Default ``None``.
        sMD_path_model : str, optional
            Path-clustering model: ``"dtw"`` (DTW + k-medoids) or ``"null"``
            (no clustering, a single path).  Default ``"dtw"``.
        sMD_n_paths : int, optional
            Fixed number of unbinding paths to cluster into.  ``None`` selects
            the number via silhouette score.  Default ``None``.
        sMD_max_frac_neg_dG_first_half : float, optional
            Pass-3 binding-well filter: maximum fraction of the first half of a
            path's ΔG trace allowed to be negative before the path is discarded.
            Values ``<= 0`` disable the filter.  Default ``0.25``.
        sMD_features : list, optional
            Extra per-frame features to include in path clustering, beyond
            ligand COM distance.  ``None`` uses the built-in feature set.
        sMD_log_geom_features : bool, optional
            Log ligand geometric features during pulling and merge them into the
            clustering feature set.  Default ``True``.
        sMD_plateau_frac : float, optional
            Fraction of the peak pulling force used to define the force-plateau
            transition-state boundary; higher values place the boundary nearer
            the rupture point.  Default ``0.4``.
        sMD_cluster_to_boundary : bool, optional
            Restrict clustering and RMSD convergence checks to frames up to the
            force-plateau boundary, excluding the bulk-solvent tail.  Default
            ``True``.
        sMD_boundary_buffer_frac : float, optional
            Extend the force-plateau boundary cap by this fraction of the
            transition-state distance, as a buffer past rupture.  Default
            ``0.1``.
        cluster_across_speeds : bool, optional
            Pool replicas from all pulling speeds into a single clustering pass,
            instead of clustering each speed independently.  Default ``False``.
        sMD_min_replicas_per_path : int, optional
            Minimum number of replicas required in a cluster for it to be kept
            as a distinct path.  Default ``5``.

        Milestone extraction
        ~~~~~~~~~~~~~~~~~~~~
        extract_milestones : bool, optional
            Extract representative frames from sMD trajectories to use as
            metadynamics starting points.  Default ``True``.
        milestone_mode : str, optional
            Extraction mode: ``"all_medoids"`` (pool all path medoids) or
            ``"per_path"`` (cluster each path's medoid independently).
            Default ``"all_medoids"``.
        milestone_min_frame_separation : int, optional
            Minimum number of frames between selected milestones to avoid
            redundancy.  Default ``0``.
        n_milestones : int, optional
            Number of milestones to extract per path (or in total for
            ``"all_medoids"`` mode).  Default ``5``.
        relax_steps : int, optional
            Number of MD steps for milestone relaxation before metadynamics.
            Default ``25000``.

        Metadynamics (mMD)
        ~~~~~~~~~~~~~~~~~~
        run_metadynamics : bool, optional
            Run the multi-walker well-tempered metadynamics stage.  Default
            ``True``.
        mMD_use_funnel_potential : bool, optional
            Add a funnel restraint around the sMD-derived exit path to suppress
            unproductive sampling.  Default ``False``.
        mMD_bias_factor : int, optional
            Well-tempered bias factor (gamma).  Default ``15``.
        mMD_bias_frequency : int, optional
            Interval (ps) at which Gaussian hills are deposited.  Default ``2``.
        mMD_hill_height : float, optional
            Initial Gaussian hill height in kJ/mol.  Default ``1.2`` (~0.5 kBT
            at 300 K), which is the recommended starting point; lower values
            slow convergence, higher values risk overfilling barriers.
        mMD_hill_width : float, optional
            Gaussian hill width (sigma) along the path CV.  Default ``0.05``.
        mMD_time : int, optional
            Metadynamics simulation time per walker in ns.  Default ``5``.
        mMD_milestone_seeding : bool, optional
            Start each walker from the relaxed checkpoint of its own milestone.
            When ``False``, all walkers start from the first (most-bound)
            milestone's checkpoint.  Default ``True``.
        mMD_multiple_walkers : bool, optional
            Have all walkers read/write the same bias directory for true
            multi-walker metadynamics.  When ``False``, each walker keeps an
            independent bias subdirectory.  Default ``False``.
        mMD_preseed_bias : bool, optional
            Pre-seed the metadynamics bias from the sMD PMF before launching
            walkers.  Default ``False``.
        mMD_preseed_speed : float, optional
            Pulling speed (nm/ps) whose sMD PMF is used for bias pre-seeding.
            ``None`` lets the preseed routine pick automatically.  Default
            ``None``.
        mMD_funnel_host_selection : str, optional
            MDAnalysis selection defining the funnel-axis host atoms.  ``None``
            auto-derives protein CA atoms within 6 Å of the ligand in the
            equilibrated structure.  Default ``None``.
        """

        # General
        self.VS_mode = VS_mode
        self.pdb_path = pdb_path
        self.do_fix_pdb = do_fix_pdb
        self.pocket_selection = pocket_selection
        self.temperature = temperature
        self.random_state = random_state
        self.platform = platform

        # Preparation
        self.run_preparation = run_preparation
        self.forcefield = forcefield
        self.hydrogenMass = hydrogenMass
        self.timestep = timestep
        self.lig_ff = lig_ff
        self.boxShape = boxShape
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.variants = variants
        self.is_membrane = is_membrane
        self.lipid_type = lipid_type
        self.ions = ions

        # Equilibration
        self.run_equilibration = run_equilibration
        self.protocol_fname = protocol_fname

        # Steered MD
        self.run_sMDpulling = run_sMDpulling
        self.sMD_outdir = sMD_outdir
        self.sMD_pulling_dir = sMD_pulling_dir
        self.sMD_pulling_speeds = sMD_pulling_speeds
        self.sMD_max_pulling_dist = sMD_max_pulling_dist
        self.sMD_max_r_offset = sMD_max_r_offset
        self.sMD_autostop_nc = sMD_autostop_nc
        self.sMD_autostop_nc_window = sMD_autostop_nc_window
        self.sMD_autostop_min_displacement = sMD_autostop_min_displacement
        self.sMD_converge_speeds = sMD_converge_speeds
        self.sMD_conv_window = sMD_conv_window
        self.sMD_conv_streak = sMD_conv_streak
        self.sMD_autostop_estimator = sMD_autostop_estimator
        self.sMD_alternate_speeds = sMD_alternate_speeds
        self.sMD_time = sMD_time  # ns
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_dx_per_move = sMD_dx_per_move
        self.sMD_spring_cte = sMD_spring_cte  # KJ/mol/nm2
        self.sMD_force_n_samples = sMD_force_n_samples
        self.sMD_force_sample_stride = sMD_force_sample_stride
        self.sMD_ligand_anchor_mode = sMD_ligand_anchor_mode
        self.sMD_max_replicas = sMD_max_replicas
        self.sMD_run_analysis = sMD_run_analysis
        self.sMD_clust_selection = sMD_clust_selection
        self.sMD_path_model = sMD_path_model
        self.sMD_n_paths = sMD_n_paths
        self.sMD_max_frac_neg_dG_first_half = sMD_max_frac_neg_dG_first_half
        self.sMD_features = sMD_features
        self.sMD_log_geom_features = sMD_log_geom_features
        self.sMD_plateau_frac = sMD_plateau_frac
        self.sMD_cluster_to_boundary = sMD_cluster_to_boundary
        self.sMD_boundary_buffer_frac = sMD_boundary_buffer_frac
        self.cluster_across_speeds = cluster_across_speeds
        self.sMD_min_replicas_per_path = sMD_min_replicas_per_path

        # Milestones
        self.extract_milestones = extract_milestones
        self.milestone_mode = milestone_mode
        self.milestone_min_frame_separation = milestone_min_frame_separation
        self.n_milestones = n_milestones
        self.relax_steps = relax_steps

        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_use_funnel_potential = mMD_use_funnel_potential
        self.mMD_bias_factor = mMD_bias_factor
        self.mMD_bias_frequency = mMD_bias_frequency
        self.mMD_hill_height = mMD_hill_height
        self.mMD_hill_width = mMD_hill_width
        self.mMD_time = mMD_time
        self.mMD_milestone_seeding = mMD_milestone_seeding
        self.mMD_multiple_walkers = mMD_multiple_walkers
        self.mMD_preseed_bias = mMD_preseed_bias
        self.mMD_preseed_speed = mMD_preseed_speed
        self.mMD_funnel_host_selection = mMD_funnel_host_selection

    @classmethod
    def get_defaults_dict(cls):
        """Return a mapping of constructor parameter names to their default values.

        Returns
        -------
        dict
            ``{parameter_name: default_value}`` for every parameter in
            :meth:`__init__`.
        """
        defaults = {}
        sig = signature(cls.__init__)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default
        return defaults

    @classmethod
    def from_config(cls, config):
        """Set up an AutoPath :class:`Config` from a JSON configuration file.

        The JSON file may use a nested structure (sections as top-level keys
        whose values are dicts of parameter names) or a flat key-value mapping.
        Legacy key names are translated automatically with a deprecation warning.

        Parameters
        ----------
        config : str
            Path to the ``config.json`` file.

        Returns
        -------
        Config
            Fully initialised :class:`Config` instance.

        Raises
        ------
        FileNotFoundError
            If *config* does not exist on disk.
        json.JSONDecodeError
            If *config* is not valid JSON.
        ValueError
            If the file contains keys that are not recognised constructor
            parameters.
        """
        logging.info("Loading settings from JSON file.")

        try:
            with open(config) as f:
                config_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file {config} not found.")
            raise
        except json.JSONDecodeError:
            logging.error(f"Config file {config} is not valid JSON.")
            raise

        # Flatten the nested dictionary structure
        flattened_config = {}
        for section, params in config_data.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    flattened_config[key] = value

        # Backward compatibility with old config keys
        legacy_map = {
            "equilibration_scheme": "protocol_fname",
        }
        for old_key, new_key in legacy_map.items():
            if old_key in flattened_config and new_key not in flattened_config:
                flattened_config[new_key] = flattened_config.pop(old_key)
                logging.warning(
                    f"Deprecated config key '{old_key}' detected. Please use '{new_key}'."
                )

        if "sMD_replicas" in flattened_config and "sMD_pulling_speeds" not in flattened_config:
            reps = flattened_config.pop("sMD_replicas")
            flattened_config["sMD_pulling_speeds"] = {0.001: reps}
            logging.warning(
                "Deprecated config key 'sMD_replicas' detected. "
                "Mapped to 'sMD_pulling_speeds' as {0.001: sMD_replicas}."
            )

        # Normalize pulling speed keys loaded from JSON (string keys) to float keys
        if "sMD_pulling_speeds" in flattened_config and isinstance(flattened_config["sMD_pulling_speeds"], dict):
            speed_map = {}
            for speed, reps in flattened_config["sMD_pulling_speeds"].items():
                try:
                    speed_map[float(speed)] = reps
                except (TypeError, ValueError):
                    speed_map[speed] = reps
            flattened_config["sMD_pulling_speeds"] = speed_map

        expected_keys = cls.get_defaults_dict().keys()
        bad_keys = [k for k in flattened_config if k not in expected_keys]
        if bad_keys:
            err_msg = "Unexpected keys in Config.from_config():" + os.linesep
            for key in bad_keys:
                err_msg += f"  - {key}" + os.linesep
            logging.error("Failed to load settings from JSON file.")
            raise ValueError(err_msg)

        p = cls(**flattened_config)
        return p
