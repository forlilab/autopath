import os
import json
import logging
from inspect import signature


class Config(object):
    """This class handles the config.json options to set up an AutoPath simulation.

    :param object: inherits from the base class
    :type object: object
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
        """Returns a dictionary with all the default values for the class attributes.

        :return: dictionary of class attributes and their default values
        :rtype: dict
        """
        defaults = {}
        sig = signature(cls.__init__)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default
        return defaults

    @classmethod
    def from_config(cls, config):
        """Sets up the parameters to run cosolvent from the config.json file supplied.

        :param config: path to the config.json file
        :type config: str
        :raises ValueError: if some attributes are not recognized
        :return: instance of the Config class
        :rtype: Config
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
