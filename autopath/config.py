import os
import json
import logging
from inspect import signature


class Config(object):
    """Configuration container for an AutoPath simulation.

    Reads and validates settings from a JSON file (via :meth:`from_config`) or
    accepts keyword arguments directly.  All parameters map 1-to-1 to
    constructor arguments and are stored as instance attributes so that
    :class:`autopath.autopath_core.AutoPath` can consume them directly.
    """

    def __init__(
        self,
        VS_mode: bool = False,
        pdb_path: str = None,
        do_fix_pdb: bool = True,
        pocket_selection: str = "same residue as protein and (around 4 resname UNK) and (not name H*)",
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
        sMD_pulling_speeds: dict = {0.001: None, 0.002: None, 0.003: None},
        sMD_max_pulling_dist: float = 2.0,  # nm
        sMD_max_r_offset: float = 3.0,
        sMD_autostop_nc: bool = False,
        sMD_autostop_nc_threshold: float = 1.0,
        sMD_autostop_lag_sigma: float = 5.0,
        sMD_autostop_lag_window: int = 20,
        sMD_autostop_min_displacement: float = 0.5,
        sMD_time: int = None,  # ns
        sMD_steps_per_move: int = None,
        sMD_dx_per_move: float = 0.001,  # nm
        sMD_spring_cte: float = None,  # KJ/mol/nm2
        sMD_ligand_anchor_mode: str = "lig_ha",
        sMD_max_replicas: int = 50,
        sMD_run_analysis: bool = True,
        sMD_clust_selection: str = None,
        extract_milestones: bool = True,
        milestone_mode: str = "per_path",
        milestone_min_frame_separation: int = 0,
        n_milestones: int = 5,
        relax_steps: int = 25000,
        run_metadynamics: bool = True,
        mMD_use_funnel_potential: bool = True,
        mMD_bias_factor: int = 10,
        mMD_bias_frequency: int = 2,  # ps
        mMD_hill_height: float = 1.2,  # kJ/mol
        mMD_hill_width: float = 0.05,
        mMD_time: int = 10,  # ns
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
            OpenFF force field version for small-molecule parametrisation.
            Default ``"openff-2.3.0"``.
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
            Lipid residue name when ``is_membrane=True``.

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
            Mapping of pulling speed (nm/ps) to replica count (or ``None``).
            Default ``{0.001: None, 0.002: None, 0.003: None}``.
        sMD_max_pulling_dist : float, optional
            Maximum COM displacement from the binding site in nm.  Default
            ``2.0``.
        sMD_max_r_offset : float, optional
            Maximum pull offset from the starting COM distance (nm); also capped
            at half-box minus 0.5 nm.  Default ``3.0``.
        sMD_autostop_nc : bool, optional
            Stop a replica when the fraction of native contacts drops below
            ``sMD_autostop_nc_threshold``.  Default ``False``.
        sMD_autostop_nc_threshold : float, optional
            Native-contact fraction threshold for autostop.  Default ``1.0``.
        sMD_autostop_lag_sigma : float, optional
            Lag-window sigma for the autostop smoothing filter.  Default ``5.0``.
        sMD_autostop_lag_window : int, optional
            Lag window length for autostop.  Default ``20``.
        sMD_autostop_min_displacement : float, optional
            Minimum COM displacement (nm) required before autostop is evaluated.
            Default ``0.5``.
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
        sMD_ligand_anchor_mode : str, optional
            Strategy for selecting ligand anchor atoms: ``"lig_ha"`` (all heavy
            atoms) or ``"murcko"`` (Murcko scaffold).  Default ``"lig_ha"``.
        sMD_max_replicas : int, optional
            Hard cap on replicas per speed to prevent infinite convergence loops.
            Default ``50``.
        sMD_run_analysis : bool, optional
            Run path clustering and PMF estimation after pulling.  Default
            ``True``.
        sMD_clust_selection : str, optional
            MDAnalysis selection for clustering features (in addition to ligand
            COM distance).  ``None`` uses only ligand COM.  Default ``None``.

        Milestone extraction
        ~~~~~~~~~~~~~~~~~~~~
        extract_milestones : bool, optional
            Extract representative frames from sMD trajectories to use as
            metadynamics starting points.  Default ``True``.
        milestone_mode : str, optional
            Extraction mode: ``"per_path"`` (cluster each path's medoid
            independently) or ``"all_medoids"`` (pool all medoids).
            Default ``"per_path"``.
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
            unproductive sampling.  Default ``True``.
        mMD_bias_factor : int, optional
            Well-tempered bias factor (gamma).  Default ``10``.
        mMD_bias_frequency : int, optional
            Interval (ps) at which Gaussian hills are deposited.  Default ``2``.
        mMD_hill_height : float, optional
            Initial Gaussian hill height in kJ/mol.  Default ``1.2`` (~0.5 kBT
            at 300 K), which is the recommended starting point; lower values
            slow convergence, higher values risk overfilling barriers.
        mMD_hill_width : float, optional
            Gaussian hill width (sigma) along the path CV.  Default ``0.05``.
        mMD_time : int, optional
            Metadynamics simulation time per walker in ns.  Default ``10``.
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
        self.sMD_autostop_nc_threshold = sMD_autostop_nc_threshold
        self.sMD_autostop_lag_sigma = sMD_autostop_lag_sigma
        self.sMD_autostop_lag_window = sMD_autostop_lag_window
        self.sMD_autostop_min_displacement = sMD_autostop_min_displacement
        self.sMD_time = sMD_time  # ns
        self.sMD_steps_per_move = sMD_steps_per_move
        self.sMD_dx_per_move = sMD_dx_per_move
        self.sMD_spring_cte = sMD_spring_cte  # KJ/mol/nm2
        self.sMD_ligand_anchor_mode = sMD_ligand_anchor_mode
        self.sMD_max_replicas = sMD_max_replicas
        self.sMD_run_analysis = sMD_run_analysis
        self.sMD_clust_selection = sMD_clust_selection

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

        self.equilibration_checkpoint = False
        if VS_mode:
            self.equilibration_checkpoint = True
            self.eq_checkpoint_cutoff = 0.3  # nm
            self.pulling_checkpoint = True

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
