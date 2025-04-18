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
        pocket_selection: str = "protein and (around 3 resname UNK) and (not name H*)",
        temperature: float = 300,
        random_state: int = 42,
        run_preparation: bool = True,
        forcefield: list = [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
            "amber/tip3p_HFE_multivalent.xml",
        ],
        lig_ff: str = "espaloma",
        boxShape: str = "dodecahedron",
        padding: float = 1.0,
        ionicStrength: float = 0.0,
        is_membrane: bool = False,
        lipid_type: str = None,
        variants: dict = None,
        run_equilibration: bool = True,
        equilibration_scheme: str = "autopath/data/equilibration.json",
        run_sMDpulling: bool = True,
        sMD_pulling_dist: float = 0.5,  # nm
        sMD_time: int = 1,  # ns
        sMD_steps_per_move: int = 250,  # 1 ps
        sMD_pulling_force: float = 20000,  # KJ/mol/nm2
        sMD_replicas: int = 5,
        sMD_autostop: bool = False,
        extract_milestones: bool = True,
        n_milestones: int = 10,
        run_relax: bool = True,
        relax_steps: int = 25000,
        run_metadynamics: bool = True,
        mMD_walkers: int = 10,
        mMD_time: int = 1,  # ns
        mMD_bias_factor: int = 3,
        mMD_hill_height: float = 0.3,  # Kcal/mol
        mMD_hill_width: float = 0.05,
    ):

        # General
        self.VS_mode = VS_mode
        self.pdb_path = pdb_path
        self.do_fix_pdb = do_fix_pdb
        self.pocket_selection = pocket_selection
        self.temperature = temperature
        self.random_state = random_state

        # Preparation
        self.run_preparation = run_preparation
        self.forcefield = forcefield
        self.lig_ff = lig_ff
        self.boxShape = boxShape
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.variants = variants
        self.is_membrane = is_membrane
        self.lipid_type = lipid_type

        # Equilibration
        self.run_equilibration = run_equilibration
        self.equilibration_scheme = equilibration_scheme

        # Steered MD
        self.run_sMDpulling = run_sMDpulling
        self.sMD_time = sMD_time  # ns
        self.sMD_replicas = sMD_replicas
        self.sMD_pulling_dist = sMD_pulling_dist
        self.sMD_steps_per_move = sMD_steps_per_move  # 1 ps
        self.sMD_pulling_force = sMD_pulling_force  # KJ/mol/nm2
        self.sMD_autostop = sMD_autostop

        # Milestones
        self.extract_milestones = extract_milestones
        self.n_milestones = n_milestones
        self.run_relax = run_relax
        self.relax_steps = relax_steps

        # Metadynamics
        self.run_metadynamics = run_metadynamics
        self.mMD_walkers = mMD_walkers
        self.mMD_time = mMD_time
        self.mMD_bias_factor = mMD_bias_factor
        self.mMD_hill_height = mMD_hill_height
        self.mMD_hill_width = mMD_hill_width

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
            for key, value in params.items():
                flattened_config[key] = value

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
