import os
import re
import numpy as np
import pandas as pd
from typing import List, Optional

import MDAnalysis as mda
import pytraj as pt
from prolif import Fingerprint

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
style.use("fivethirtyeight")

import logging
logger = logging.getLogger("autopath")

class ProteinLigandAnalyzer:
    """
    Simple analysis class for protein-ligand or protein-protein MD simulations.

    Features:
    - load multiple trajectories (replicas). Should have same topology
    - compute interactions with ProLif
    - compute LIE using pytraj. Optional selection of residues based on ProLif results.
    """

    def __init__(self,
                 top: str,
                 trajs: List[str],
                 ligand_mda_selection: str = "resname UNK",
                 protein_mda_selection: Optional[str] = None,
                 traj_start:Optional[int] = None,
                 traj_end:Optional[int] = None,
                 traj_step:Optional[int] = None,
                 outdir: str = "aplip_analysis"
                 
                 ):
        """
        Parameters
        ----------
        top : str
            Topology file (PDB, PRMTOP, PSF, etc.)
        trajs : list of str
            List of trajectory files (each considered a separate replica)
        ligand_resname : str
            Residue name for the ligand (default "UNK")
        selection : str or None
            Residue selection string (MDAnalysis-style). If None, LIE uses all residues.
        """
        self.top = top
        self.traj_paths = trajs
        # MDAnalysis selections, not Amber
        self.ligand_mda_selection = ligand_mda_selection
        self.protein_mda_selection = protein_mda_selection # used for LIE if provided

        self._load_trajectories(traj_start, traj_end, traj_step)
        
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        
        return
    # -----------------------------------------------------------
    #   INTERNAL HELPERS
    # -----------------------------------------------------------

    def _load_trajectories(self):
        """Load each replica separately using MDAnalysis."""
        #FIXME the slicing is nto working as intended
        self.replicas = {}
        for t in self.traj_paths:
            try:
                u = mda.Universe(self.top, t)
                repname = os.path.basename(t).split('.')[0]
                self.replicas[repname] = u
            except Exception as e:
                logger.warning(f"Failed to load trajectory {t} with topology {self.top}: {e}")
                pass
            
    def concat_trajectories(self):
        """Concatenate all replicas into a single MDAnalysis Universe."""
        all_trajs = []
        for rep in self.replicas.values():
            all_trajs.append(rep.trajectory)
        
        # Create a new Universe with concatenated trajectories
        concat_u = mda.Universe(self.top)
        concat_u.load_new(np.concatenate([t.timeseries() for t in all_trajs], axis=0))
        
        return concat_u

    def write_combined_trajectory(
            topology: str,
            trajectories: List[str],
            output_traj: str,
            start: Optional[int] = None,
            stop: Optional[int] = None,
            step: Optional[int] = None,
        ) -> str:
        """
        Combine multiple trajectories into a single trajectory file,
        applying start/stop/step independently to each input trajectory.

        Parameters
        ----------
        topology : str
            Topology file (PRMTOP, PSF, PDB, etc.)
        trajectories : list of str
            Input trajectory files (DCD, XTC, etc.)
        output_traj : str
            Output trajectory filename.
        start, stop, step : int or None
            Slicing applied independently to each trajectory.
        """

        # Load first trajectory to initialize writer
        u0 = mda.Universe(topology, trajectories[0])

        with mda.Writer(output_traj, n_atoms=u0.atoms.n_atoms) as W:

            for traj in trajectories:
                u = mda.Universe(topology, traj)

                for ts in u.trajectory[start:stop:step]:
                    W.write(u.atoms)

        return output_traj
            
    # -----------------------------------------------------------
    #   PROLIF STUFF
    # -----------------------------------------------------------

    def get_persistent_interactions(self,
                                   fp_interactions: Optional[List[str]] = None,
                                   stride: int = 1,
                                   frequency_cutoff: float = 0.5
                                   ):
        """
        Identify most persistent interactions across replicas using ProLif.

        Parameters
        ----------
        frequency_cutoff : float
            Only interactions present in at least this fraction of frames are kept.

        Returns
        -------
        important_resids : list of int
        """

        important_resids = set()
        if fp_interactions is not None:
            fp = Fingerprint(interactions=fp_interactions)
        else:
            fp = Fingerprint()  # default interaction set

        for rep_name, rep in self.replicas.items():
            fp_fname = os.path.join(self.outdir, f"FP_{rep_name}.pkl")
            if os.path.exists(fp_fname):
                fp = Fingerprint.from_pickle(fp_fname)
                logger.info(f"Loaded cached ProLif fingerprint for replica {rep_name}")
            else:
                logger.info(f"Computing ProLif fingerprint for replica {rep_name}")
                protein_sel = rep.select_atoms(self.protein_mda_selection) if self.protein_mda_selection else rep.select_atoms("protein")
                ligand_sel = rep.select_atoms(self.ligand_mda_selection)

                # Compute interaction fingerprint over trajectory, optionally strided
                fp = fp.run(rep.trajectory, #FIXME stride does not work here
                                protein_sel, 
                                ligand_sel
                                )
                # TODO: another function should read and analyze these pickles
                fp.to_pickle(fp_fname)
            
            # convert to DataFrame
            df = fp.to_dataframe()
            fp.plot_barcode()
            plt.savefig(os.path.join(self.outdir, f"prolif_barcode_{rep_name}.png"))
            plt.close()
            # percentage of the trajectory where each interaction is present
            persistence_byRes_byType = (df.mean().sort_values(ascending=False).to_frame(name="%").T * 1).T

            # same but we regroup all interaction types
            persistence_byRes = (
                df.T.groupby(level=["protein", 'ligand'])
                .sum()
                .T.astype(bool)
                .mean()
                .sort_values(ascending=False)
                .to_frame(name="%")
                .T
                * 1
            ).T
            
            # Filter residues by frequency
            selected_residues = persistence_byRes[persistence_byRes["%"] >= frequency_cutoff].index.tolist()
            selected_resnames = [res[1] for res in selected_residues]
            selected_resids = [int(res[3:]) for res in selected_resnames]
            important_resids.update(selected_resids)

        return sorted(list(important_resids))

    # -----------------------------------------------------------
    #   LIE CALCULATION VIA PYTRAJ
    # -----------------------------------------------------------

    def compute_LIE(self,
                    prmtop: Optional[str] = None,
                    use_residues: Optional[List[int]] = None,
                    ligand_amber_selection: str = ":UNK",
                    exclude_amber_selection: Optional[str] = ':Na+,Cl-,NA,CL,K,K+',
                    cutoff: float = 6.0,
                    stride: int = 1,
                    lie_options:str = 'nopbc cutvdw 10.0 cutelec 10.0' # dielec 2
                    ):
        """
        Compute LIE for each replica using pytraj.
        some literature:
        https://ambermd.org/tutorials/advanced/tutorial24/liew.php
        https://pubs.acs.org/doi/10.1021/acs.jcim.9b00609
        https://pmc.ncbi.nlm.nih.gov/articles/PMC7311763/
        
        Parameters
        ----------
        prmtop : str
            Amber PRMTOP topology for pytraj. If not provided, use topology from self.top.
        use_residues : list of ints or None
            If provided, only use these residues in LIE mask.
        cutoff : float
            Distance cutoff (Å) for selecting residues when use_residues=None.

        Returns
        -------
        lie_df : pd.DataFrame
            Combined LIE results for all replicas.
        """

        if prmtop is None:
            prmtop = self.top
            
        all_reps = []
        for traj in self.traj_paths:
            run_name = os.path.basename(traj).split('.')[0]

            ptraj = pt.iterload(traj, prmtop, stride=stride)

            # LIE mask logic
            if use_residues is not None:
                res_string = ",".join(str(r) for r in use_residues)
            else:
                # auto-select residues within cutoff of ligand
                ref = pt.iterload(traj, prmtop)[0]
                ptraj.top.set_reference(ref)

                idx = ptraj.top.select(f"({ligand_amber_selection}<:{cutoff}) & !({exclude_amber_selection}) & !({ligand_amber_selection})")
                resids = sorted({ptraj.top[i].resid for i in idx})
                res_string = ",".join(str(r) for r in resids)

            mask = f"LIE {ligand_amber_selection} :{res_string}"

            # Run LIE
            lie = pt.analysis.energy_analysis.lie(
                ptraj,
                mask=mask,
                options=lie_options,
                dtype='dict'
            )

            df = pd.concat([
                pd.DataFrame(lie["LIE[EELEC]"]),
                pd.DataFrame(lie["LIE[EVDW]"])
            ], axis=1)
            df.columns = ["EELEC", "VDW"]
            df["run"] = run_name
            df["Total"] = df["EELEC"] + df["VDW"]

            all_reps.append(df)

        df_all = pd.concat(all_reps, ignore_index=True)
        
        # plot LIE components
        self.plot_lie_components(df_all)
        df_all.to_csv(os.path.join(self.outdir, "LIE_results.csv"), index=False)
        
        return df_all

    def plot_lie_components(self, lie_df: pd.DataFrame):
        
        # create one subplot for each component
        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
            sharex=True, sharey=False)

        for component, ax in zip(["Total", "EELEC", "VDW"], axes.flatten()):
            sns.lineplot(data=lie_df, x=lie_df.index, y=component, ax=ax, label=component)
            ax.set_title(f"{component.upper()}")
            ax.set_xlabel("Frame") ;    ax.set_ylabel("LIE Energy (kJ/mol)")
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "LIE_components.png"))
        plt.close()
        
        return
    
    # -----------------------------------------------------------
    #   MMPB(GB)SA 
    # -----------------------------------------------------------
    @staticmethod
    def write_qfile_mmpbsa(
                    sysname:str=None,
                    out_dir:str=None,
                    mmpbsa_in:str=None,
                    system_prmtop:str=None,
                    trajectory:str=None,
                    lig_selection:str=None,
                    strip_selection:str=":POP:WAT:Na+:Cl-:Mg+:K+:HOH:NA:CL:K:MG",
                    gpu_resource="rtxa6000", 
                    gpu_num=1, 
                    time="3-0",
                    omp_threads=64,
                    partition="forli,alphafold,shared"
                    ):
        """Function to write a SLURM qfile for MMPBSA calculations."""    
        
        template='''#!/bin/bash
    #SBATCH -e ${out_dir}/${sysname}_mmpbsa.err
    #SBATCH -o ${out_dir}/${sysname}_mmpbsa.out
    ##SBATCH --gres=gpu#:${gpu_resource}:${gpu_num} # COMMENT OUT THE # IF YOU WANT TO USE A SPECIFIC GPU TYPE
    #SBATCH --time=${time}
    #SBATCH --partition=${partition}
    #SBATCH --exclude=nodea0111,nodea0110 # EXCLUDE KNOWN PROBLEMATIC NODES
    #SBATCH --ntasks=${omp_threads}  # Request 32 separate MPI processes/slots
    #SBATCH --cpus-per-task=1 # Each process uses 1 CPU. for MPI runs
    ## SBATCH --cpus-per-task=${omp_threads} # Each process uses multiple CPUs. for OpenMP runs
    #SBATCH --job-name="mmpbsa_${sysname}"

    # module purge
    module load openmpi/3.1.6
    # module load gcc

    source ~/.bashrc
    micromamba activate autopath3

    module load amber/24
    #export OMP_NUM_THREADS=${omp_threads}

    echo "Starting mmpbsa calculation for ${sysname} at $(date)"
    echo "Running on $(hostname)"
    echo "Entering output directory ${out_dir} ..."
    cd ${out_dir}

    echo "Running ante-mmpbsa to generate prmtop files..."
    ante-MMPBSA.py -p ${system_prmtop} -s ${strip_selection} -n ${lig_selection} --radii mbondi2 -c complex.prmtop -r receptor.prmtop -l ligand.prmtop

    echo "Finished ante-mmpbsa at $(date)"
    echo "Running mmpbsa.py for trajectory ${trajectory} ..."

    # MMPBSA.py -O -i ${mmpbsa_in} -o FINAL_RESULTS_mmpbsa.dat -do FINAL_DECOMP_mmpbsa.dat -sp ${system_prmtop} -y ${trajectory} -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop
    mpirun -np ${omp_threads} MMPBSA.py.MPI -O -i ${mmpbsa_in} -o FINAL_RESULTS_mmpbsa.dat -do FINAL_DECOMP_mmpbsa.dat -sp ${system_prmtop} -y ${trajectory} -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop

    '''

        with open(f"qfiles_mmpbsa/{sysname}_mmpbsa.q", "w") as f:
            template = template.replace("${sysname}", sysname)
            template = template.replace("${out_dir}", out_dir)
            template = template.replace("${mmpbsa_in}", mmpbsa_in)
            template = template.replace("${system_prmtop}", system_prmtop)
            template = template.replace("${trajectory}", trajectory)
            template = template.replace("${lig_selection}", lig_selection)
            template = template.replace("${strip_selection}", strip_selection)
            template = template.replace("${gpu_resource}", gpu_resource)
            template = template.replace("${gpu_num}", str(gpu_num))
            template = template.replace("${time}", time)
            template = template.replace("${omp_threads}", str(omp_threads))
            template = template.replace("${partition}", partition)
            f.write(template)

        return
    
    @staticmethod
    def parse_mmpbsa_deltas_all_components(filepath:str=None) -> pd.DataFrame:
        """
        Parse the DELTAS section of an AMBER MMPBSA per-residue decomposition file
        with all energy components (PB/GB).
        """

        with open(filepath) as f:
            lines = f.readlines()

        start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("DELTAS:"):
                start = i
                break
        if start is None:
            raise ValueError("No DELTAS section found in file")

        # Line layout around DELTAS:
        # start       : "DELTAS:"
        # start + 1   : "Total Energy Decomposition:"
        # start + 2   : "Residue,Location,Internal,,,van der Waals,,,Electrostatic,..."
        # start + 3   : ",,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,..."
        header_line = lines[start + 2].strip()
        header_parts = [h.strip() for h in header_line.split(",")]

        # Should be "Residue,Location,..."
        if not header_parts[0].startswith("Residue"):
            raise ValueError(f"Unexpected header format: {header_line}")

        has_location = (len(header_parts) > 1 and header_parts[1] == "Location")
        offset = 2 if has_location else 1  # number of leading non-numeric columns

        #figure out group names (Internal, van der Waals, etc.)
        group_names = []
        for idx, token in enumerate(header_parts[offset:]):
            token = token.strip()
            # every 3rd token is a group name (name, '', '')
            if token and idx % 3 == 0:
                group_names.append(token)

        # build column names: Group_Avg, Group_StdDev, Group_StdErr
        suffixes = ("Avg", "StdDev", "StdErr")
        group_cols = []
        for g in group_names:
            g_slug = re.sub(r"\s+", "_", g)
            g_slug = g_slug.replace(".", "").replace("-", "_")
            for s in suffixes:
                group_cols.append(f"{g_slug}_{s}")

        # data start: skip DELTAS, "Total Energy Decomposition:", group header, subheader
        data_start = start + 4

        records = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith(("Run on", "Complex:", "Receptor:", "Ligand:")):
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < offset + len(group_cols):
                # likely end of table or malformed line
                continue

            if has_location:
                res_field, loc_field = parts[0], parts[1]
            else:
                res_field, loc_field = parts[0], ""

            # Parse residue name and ID, e.g. "ARG   1"
            m = re.match(r"^([A-Z0-9]{3})\s+(-?\d+)$", res_field)
            if not m:
                continue
            resname, resid = m.groups()
            resid = int(resid)

            record = {
                "resname": resname,
                "resid": resid,
            }

            if has_location:
                # "R ARG   1" - "R"
                record["location"] = loc_field.split()[0] if loc_field else ""

            # numeric values: 6 groups x 3 fields = 18 numbers (for PB/GB)
            numeric_raw = parts[offset:]
            # drop empty trailing commas
            numeric_raw = [x for x in numeric_raw if x != ""]
            nums = []
            for x in numeric_raw:
                try:
                    nums.append(float(x))
                except ValueError:
                    # if anything weird appears, put NaN
                    nums.append(float("nan"))

            for col, val in zip(group_cols, nums[:len(group_cols)]):
                record[col] = val

            records.append(record)

        df = pd.DataFrame.from_records(records)
        df['label'] = df['resname'] + df['resid'].astype(str)
        return df
        