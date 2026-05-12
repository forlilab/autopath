import os
import re
import string
import numpy as np
import pandas as pd
from typing import List, Optional
from collections import Counter

import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.distances import distance_array
import pytraj as pt
from prolif import Fingerprint
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")
plt.rcParams["savefig.facecolor"] = 'white'
plt.rcParams["savefig.edgecolor"] = 'white'
plt.rcParams["axes.facecolor"] = 'white'
# plt.rcParams["axes.edgecolor"] = 'black'

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

        self._load_trajectories()
        
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

    @staticmethod
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

    def find_interfacial_waters(
            u: mda.Universe,
            ligand_sel: str = "resname UNK",
            protein_sel: str = "protein",
            traj_slice: tuple = None,
            cutoff: float = 3.5,
            fraction_persistence: float = 0.9,
            pdb_fname: str = None,
    ):
        """
        Identify persistent interfacial water molecules across a trajectory.

        A water is interfacial in a frame if:
            - any atom of that water is within `cutoff` Å of protein
            - AND within `cutoff` Å of ligand
        """

        WATER_SEL= "resname HOH or resname WAT or resname SOL"
        
        start, end, step = None, None, None
        if traj_slice is not None:
            start, end, step = traj_slice
        
        counts = Counter()
        
        n_frames = []
        for ts in u.trajectory[start:end:step]:

            # waters = u.select_atoms(WATER_SEL)

            # Waters close to protein
            near_protein = u.select_atoms(
                f"({WATER_SEL}) and around {cutoff} ({protein_sel})"
            )

            # Waters close to ligand
            near_ligand = u.select_atoms(
                f"({WATER_SEL}) and around {cutoff} ({ligand_sel})"
            )

            # Residue-level intersection
            interfacial_resids = np.intersect1d(
                near_protein.resids,
                near_ligand.resids
            )

            counts.update(interfacial_resids)
            n_frames.append(1)
            
        n_frames = sum(n_frames)
        counts = {resid: c/n_frames for resid, c in counts.items()}
        persistent = {resid: c for resid, c in counts.items() if c >= fraction_persistence}
        # print(counts)
        
        if pdb_fname is not None:
            keep_water_str = " ".join(str(r) for r in persistent.keys())
            persistent_water_sel = f"({WATER_SEL}) and resid {keep_water_str}"

            # Final cleaned selection
            final_sel_str = f"({protein_sel}) or ({ligand_sel}) or ({persistent_water_sel})"

            cleaned = u.select_atoms(final_sel_str)

            # Write the final frame
            u.trajectory[-1]
            cleaned.write(pdb_fname)

        return dict(sorted(persistent.items(), key=lambda item: item[1], reverse=True))
            
    # -----------------------------------------------------------
    #   PROLIF STUFF
    # -----------------------------------------------------------

    def get_persistent_interactions(self,
                                   fp_interactions: Optional[List[str]] = None,
                                   frequency_cutoff: float = 0.5,
                                   stride: int = 1,
                                   n_jobs: int = 1
                                   ):
        """
        Identify most persistent interactions across replicas using ProLif.
        Prolif gives me problems with parallel processing, so n_jobs=1 by default.

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

        for rep_name, u in self.replicas.items():
            fp_fname = os.path.join(self.outdir, f"FP_{rep_name}.pkl")
            if os.path.exists(fp_fname):
                fp = Fingerprint.from_pickle(fp_fname)
                logger.info(f"Loaded cached ProLif fingerprint for replica {rep_name}")
            else:
                logger.info(f"Computing ProLif fingerprint for replica {rep_name}")
                protein_sel = u.select_atoms(self.protein_mda_selection) if self.protein_mda_selection else u.select_atoms("protein")
                logger.info(f"Protein selection has {protein_sel.n_atoms} atoms.")
                ligand_sel = u.select_atoms(self.ligand_mda_selection)
                logger.info(f"Ligand selection has {ligand_sel.n_atoms} atoms.")
                
                # Compute interaction fingerprint over trajectory, optionally strided
                fp = fp.run(u.trajectory, #FIXME stride does not work here
                            protein_sel, 
                            ligand_sel,
                            n_jobs=n_jobs, # BROKEN IN PROLIF FOR DILL AND MULTIPROCESS version problems be careful
                            # parallel_strategy='chunk' #chunk, queue
                            )
                # TODO: another function should read and analyze these pickles
                fp.to_pickle(fp_fname)
            
            # convert to DataFrame
            df = fp.to_dataframe()
            fp.plot_barcode()
            plt.savefig(os.path.join(self.outdir, f"prolif_barcode_{rep_name}.png"))
            # plt.show()
            plt.close()
            # percentage of the trajectory where each interaction is present
            persistence_byRes_byType = (df.mean().sort_values(ascending=False).to_frame(name="%").T * 100).T

            # same but we regroup all interaction types
            persistence_byRes = (
                df.T.groupby(level=["protein", 'ligand'])
                .sum()
                .T.astype(bool)
                .mean()
                .sort_values(ascending=False)
                .to_frame(name="%")
                .T
                * 100
            ).T
            
            # Filter residues by frequency
            selected_residues = persistence_byRes[persistence_byRes["%"] >= frequency_cutoff/100].index.tolist()
            selected_resnames = [res[1] for res in selected_residues]
            selected_resids = [int(res[3:]) for res in selected_resnames]
            important_resids.update(selected_resids)

        return sorted(list(important_resids)), persistence_byRes, persistence_byRes_byType

    @staticmethod
    def plot_tanimoto_similarity(query_fp, reference_fp, use_frame:int=None, outdir:str=None):
        
        if outdir is None:
            outdir = "."
            
        if reference_fp is None:
            reference_fp = query_fp
            
        query_bit = query_fp.to_bitvectors()
        query_df = query_fp.to_dataframe()
        reference_bit = reference_fp.to_bitvectors()

        if use_frame is not None:
            refe_bit = reference_bit[use_frame]
            tanimoto_sims = DataStructs.BulkTanimotoSimilarity(refe_bit, query_bit)
            plt.figure(figsize=(6,4))
            sns.lineplot(x=range(len(tanimoto_sims)), y=tanimoto_sims)
            plt.xlabel("Frame index"); plt.ylabel("Tanimoto similarity")
            plt.title(f"Tanimoto similarity to frame {use_frame}")
            plt.savefig(f"{outdir}/tanimoto_to_frame_{use_frame}.png", dpi=300)
            plt.show()
            plt.close()
            return tanimoto_sims
        else:
            # Tanimoto similarity matrix
            similarity_matrix = []
            for bv in query_bit:
                similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(bv, query_bit))
            similarity_matrix = pd.DataFrame(similarity_matrix, index=query_df.index, columns=query_df.index)
            fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
            colormap = sns.color_palette('viridis', as_cmap=True)
            sns.heatmap(similarity_matrix, ax=ax,
            square=True, cmap=colormap, vmin=0, vmax=1,
            center=0.5, xticklabels=5,  yticklabels=5, )
            ax.invert_yaxis()
            plt.yticks(rotation="horizontal", fontsize=5); plt.xticks(fontsize=5)
            plt.ylabel("Frame", fontsize=7); plt.xlabel("Frame", fontsize=7)
            fig.patch.set_facecolor("white")
            plt.title("Tanimoto similarity matrix", fontsize=8)
            plt.savefig(f"{outdir}/tanimoto_similarity_matrix.png", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            return similarity_matrix
        
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
            mean_ = lie_df[component].mean()
            std_ = lie_df[component].std()
            sns.lineplot(data=lie_df, x=lie_df.index, y=component, ax=ax, label=f"Mean: {mean_:.2f} kJ/mol\nStd: {std_:.2f} kJ/mol")
            ax.set_title(f"{component.upper()}")
            ax.set_xlabel("Frame") ;    ax.set_ylabel("LIE Energy (kJ/mol)")
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "LIE_components.png"))
        # plt.show()
        plt.close()
        
        return
    
    # -----------------------------------------------------------
    #   MMPB(GB)SA 
    # -----------------------------------------------------------
    @staticmethod
    def _write_qfile_mmpbsa(
                    sysname: str = None,
                    out_dir: str = None,
                    mmpbsa_in: str = None,
                    system_prmtop: str = None,
                    trajectory: str = None,
                    lig_selection: str = None,
                    strip_selection: str = ":POP:WAT:Na+:Cl-:Mg+:K+:HOH:NA:CL:K:MG",
                    radii: str = 'mbondi2',
                    time: str = "0:10:00",
                    omp_threads: int = 222,
                    partition: str = "forli,forli-pro,shared,highmem,gpu",
                    slurm_template_fname: str = None,
                    ):
        """Write a SLURM qfile for MMPBSA calculations.

        Parameters
        ----------
        slurm_template_fname : str, optional
            Path to a custom SLURM bash template. If None, uses the bundled
            autopath/data/mmpbsa_slurm_template.q. The template must use
            ${variable} placeholders. Shell command substitutions like $(date)
            must be written as $$(date) in the template file.
        """
        if slurm_template_fname is None:
            slurm_template_fname = os.path.join(
                os.path.dirname(__file__), "data", "mmpbsa_slurm_template.q"
            )

        with open(slurm_template_fname, "r") as f:
            tmpl = string.Template(f.read())

        rendered = tmpl.substitute(
            sysname=sysname,
            out_dir=out_dir,
            mmpbsa_in=mmpbsa_in,
            system_prmtop=system_prmtop,
            trajectory=trajectory,
            lig_selection=lig_selection,
            strip_selection=strip_selection,
            radii=radii,
            time=time,
            omp_threads=str(omp_threads),
            partition=partition,
        )

        with open(f"qfiles_mmgbsa/{sysname}_mmgbsa.q", "w") as f:
            f.write(rendered)
    
    @staticmethod
    def prepare_mmgbsa_batch(
            sysname:str=None,
            prmtop:str=None,
            traj_fname:str=None,
            ligand_amber_selection:str=":UNK",
            ligand_mda_selection:str="resname UNK",
            strip_amber_selection:str=":POP:HOH:WAT:NA:CL:K:MG",
            traj_slice:tuple=None, #(start, end, step)
            persistent_waters_cutoff:float=None,
            mmpbsa_in:str="mmgbsa.in",
            radii:str='mbondi2',
            output_folder:str="mmgbsa_results",
            bash_fname:str="run_mmgbsa_batch.sh",
            mpi_threads:int=222,
            slurm_template_fname:str=None,
            ):
        """Prepare MMPBSA batch script and qfiles."""
        
        if sysname is None or prmtop is None or traj_fname is None:
            raise ValueError("sysname, prmtop, and traj_fname must be provided.")
        
        os.makedirs('qfiles_mmgbsa', exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        if ligand_mda_selection is None:
            # brittle conversion from Amber to MDA selection
            ligand_mda_selection = ligand_amber_selection.replace(":", "resname ")
        
        with open(mmpbsa_in, 'r') as file:
            mmpbsa_template = file.readlines()
        
        u = mda.Universe(prmtop, traj_fname, in_memory=True)
        start, end, step = None, None, None

        if traj_slice is not None:
            start, end, step = traj_slice
            n_frames = len(u.trajectory[start:end:step])
        else:
            n_frames = len(u.trajectory)
            
        mpi_threads = min(mpi_threads, n_frames) #avoid problems with too many threads
        logger.info(f"Writing MMPBSA qfile for {sysname} with {n_frames} frames and {mpi_threads} MPI threads.")

        mmpbsa_template = _replace_line(mmpbsa_template,line_to_match='#startframe', 
                                            new_line=f'startframe = {start if start is not None else 1},'
                                            )
        mmpbsa_template = _replace_line(mmpbsa_template,line_to_match='#endframe', 
                                            new_line=f'endframe = {end if end is not None else len(u.trajectory)},'
                                            )
        mmpbsa_template = _replace_line(mmpbsa_template,line_to_match='#interval',
                                            new_line=f'interval = {step if step is not None else 1},'
                                            )
        
        system_prmtop_abs = os.path.abspath(prmtop)
        trajectory_abs = os.path.abspath(traj_fname)
        
        # Identify persistent interfacial waters if requested                    
        if persistent_waters_cutoff is not None:
            persistent_waters = ProteinLigandAnalyzer.find_interfacial_waters(u,
                ligand_sel=ligand_mda_selection, protein_sel="protein", cutoff=3.5,
                fraction_persistence=persistent_waters_cutoff,
                traj_slice=traj_slice,
                pdb_fname=os.path.join(output_folder, f"{sysname}_persistentWaters.pdb")
            )
            if persistent_waters:
                # exclude these waters from stripping
                water_resids_str = ",".join(str(r) for r in persistent_waters.keys())
                logger.info(f"Found {len(persistent_waters)} persistent interfacial waters: {water_resids_str} for {sysname}")
                dried_amber_selection = strip_amber_selection.replace(':WAT', '').replace(':HOH', '')
                strip_amber_selection = f'((:WAT,HOH)&!(:{water_resids_str}))|{dried_amber_selection}'
                mmpbsa_template = _replace_line(mmpbsa_template, 
                                        line_to_match='strip_mask', 
                                        new_line=f'strip_mask= "{strip_amber_selection}",'
                                        )                    
            else:
                logger.info(f"No persistent interfacial waters found with the given cutoff {persistent_waters_cutoff}")
            
        # Write the modified content back to the file
        mmpbsa_out = os.path.join(output_folder, f"{sysname}_mmgbsa.in")
        mmpbsa_out_abs = os.path.abspath(mmpbsa_out)
        with open(mmpbsa_out_abs, 'w') as file:
            file.writelines(mmpbsa_template)

        ProteinLigandAnalyzer._write_qfile_mmpbsa(sysname=sysname,
                                                out_dir=output_folder,
                                                mmpbsa_in=mmpbsa_out_abs,
                                                system_prmtop=system_prmtop_abs,
                                                trajectory=trajectory_abs,
                                                lig_selection=ligand_amber_selection,
                                                strip_selection=strip_amber_selection,
                                                radii=radii,
                                                omp_threads=mpi_threads,
                                                slurm_template_fname=slurm_template_fname,
                                                )
        
        # update batch bash script with all qfiles in folder
        qfiles = [f for f in os.listdir('qfiles_mmgbsa') if f.endswith('_mmgbsa.q')]
        with open(bash_fname, "w") as f:
            f.write("#!/bin/bash\n\n")
            for qf in qfiles:
                f.write(f"sbatch {os.path.abspath(os.path.join('qfiles_mmgbsa', qf))}\n")

        os.chmod(bash_fname, 0o755)
        
        return bash_fname
        
    @staticmethod
    def parse_mmpbsa_differences_table(path:str) -> pd.DataFrame:
        """
        Parse the 'Differences (Complex - Receptor - Ligand):' table from FINAL_RESULTS_mmpbsa.dat.

        Returns DataFrame with:
        Component, Average, Std_Dev, Std_Err_Mean
        """
        with open(path, "r") as f:
            lines = f.readlines()

        #Locate the start of the Differences section
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Differences (Complex - Receptor - Ligand):"):
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("Could not find 'Differences (Complex - Receptor - Ligand):' section.")

        # ve to first data line (after dashed separator)
        i = start_idx + 1
        while i < len(lines):
            if re.match(r"^-{5,}\s*$", lines[i].strip()):  # line of dashes
                i += 1
                break
            i += 1

        #Parse rows: name (possibly with spaces) + 3 floats
        float_row = re.compile(
            r"^\s*(?P<name>.*?)\s+"
            r"(?P<avg>-?\d+(?:\.\d+)?)\s+"
            r"(?P<std>-?\d+(?:\.\d+)?)\s+"
            r"(?P<sem>-?\d+(?:\.\d+)?)\s*$"
        )

        rows = []
        while i < len(lines):
            line = lines[i].rstrip("\n")
            s = line.strip()

            # Skip blank lines (DELTA rows often come after blanks)
            if s == "":
                i += 1
                continue

            # Stop when the next section begins (usually a header ending with :)
            # e.g. "Energy Component ..." blocks elsewhere, or other section titles
            if s.endswith(":") and not s.startswith("DELTA"):
                break

            m = float_row.match(line)
            if m:
                rows.append({
                    "Component": m.group("name").strip(),
                    "Average": float(m.group("avg")),
                    "Std_Dev": float(m.group("std")),
                    "Std_Err_Mean": float(m.group("sem")),
                })

            i += 1

        if not rows:
            raise ValueError("Found the Differences section, but parsed zero rows.")

        return pd.DataFrame(rows)
        
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
    
    @staticmethod
    def parse_mmpbsa_differences_table(path):
        """
        Parse the 'Differences (Complex - Receptor - Ligand):' table from FINAL_RESULTS_mmpbsa.dat.

        Returns DataFrame with:
        Component, Average, Std_Dev, Std_Err_Mean
        """
        with open(path, "r") as f:
            lines = f.readlines()

        #Locate the start of the Differences section
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Differences (Complex - Receptor - Ligand):"):
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("Could not find 'Differences (Complex - Receptor - Ligand):' section.")

        # ve to first data line (after dashed separator)
        i = start_idx + 1
        while i < len(lines):
            if re.match(r"^-{5,}\s*$", lines[i].strip()):  # line of dashes
                i += 1
                break
            i += 1

        #Parse rows: name (possibly with spaces) + 3 floats
        float_row = re.compile(
            r"^\s*(?P<name>.*?)\s+"
            r"(?P<avg>-?\d+(?:\.\d+)?)\s+"
            r"(?P<std>-?\d+(?:\.\d+)?)\s+"
            r"(?P<sem>-?\d+(?:\.\d+)?)\s*$"
        )

        rows = []
        while i < len(lines):
            line = lines[i].rstrip("\n")
            s = line.strip()

            # Skip blank lines (DELTA rows often come after blanks)
            if s == "":
                i += 1
                continue

            # Stop when the next section begins (usually a header ending with :)
            # e.g. "Energy Component ..." blocks elsewhere, or other section titles
            if s.endswith(":") and not s.startswith("DELTA"):
                break

            m = float_row.match(line)
            if m:
                rows.append({
                    "Component": m.group("name").strip(),
                    "Average": float(m.group("avg")),
                    "Std_Dev": float(m.group("std")),
                    "Std_Err_Mean": float(m.group("sem")),
                })

            i += 1

        if not rows:
            raise ValueError("Found the Differences section, but parsed zero rows.")

        return pd.DataFrame(rows)
        
    @staticmethod
    def plot_mmpbsa_byresidue(df_decomp: pd.DataFrame,
                            top_residues: int=10,
                            out_dir: str=None
                            ):
        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        mmpbsa_components_list = ['TOTAL_Avg', 'Electrostatic_Avg', "van_der_Waals_Avg",
                    # 'Internal_Avg', #this one is usually not very informative
                    "Polar_Solvation_Avg", "Non_Polar_Solv_Avg"]
        
        #you may care about ligands if you are studying protein-protein interactions
        locations = {'R':'receptor', 'L':'ligand'}
        for loc, location in locations.items():
            df = df_decomp[df_decomp["location"] == loc].copy()
            
            for component in mmpbsa_components_list:
                out_fname = os.path.join(out_dir, f'mmpbsa_byres_{location}_{component}.png')
                
                # Plot per-residue MMGBSA decomposition for top/bottom residues
                top = df.sort_values(component).head(top_residues)
                bottom = df.sort_values(component).tail(top_residues)
                
                plt.figure(figsize=(int(1*top_residues), int(top_residues/2)))
                plt.bar(top["label"], top[component], color="skyblue", yerr=top[component.replace('Avg', 'StdErr')], capsize=4)
                if component != 'van_der_Waals_Avg':
                    plt.bar(bottom["label"], bottom[component], color="salmon", yerr=bottom[component.replace('Avg', 'StdErr')], capsize=4)
                plt.xticks(rotation=45)
                plt.ylabel("ΔG_res (kcal/mol)")
                plt.title(f"{component} energy, {location}")
                plt.tight_layout()
                plt.savefig(out_fname, dpi=300)
                # plt.show()
                plt.close()
        
        return
    
    @staticmethod
    def paint_mmpbsa_byresidue(df_decomp,
                            pdb_file: str=None,
                            prmtop_file: str=None,
                            mmpbsa_component: str='all',
                            normalize: bool=False,
                            outdir: str=None,
                            ):
        """paint_mmpbsa_byresidue This function colors a PDB structure 
        based on per-residue MMGBSA decomposition values.

        Args:
            df_decomp (pd.DataFrame): DataFrame containing MMGBSA decomposition data.
            pdb_file (str): Path to the PDB file.
            prmtop_file (str): Path to the topology file.
            mmpbsa_component (str): Component to use for coloring (e.g., 'TOTAL').
        """
        mmpbsa_components_list = ['TOTAL_Avg', 'Electrostatic_Avg', "van_der_Waals_Avg",
                            # 'Internal_Avg', #this one is usually not very informative
                            "Polar_Solvation_Avg", "Non_Polar_Solv_Avg"]
        if outdir is None:
            outdir = '.'
        
        os.makedirs(outdir, exist_ok=True)
            
        if mmpbsa_component.upper() == 'ALL':
            components_list = mmpbsa_components_list
        else:
            if mmpbsa_component not in mmpbsa_components_list:
                print(f"ERROR: mmpbsa_component must be one of {mmpbsa_components_list} or 'all'.")
            else:
                components_list = [mmpbsa_component]

        if pdb_file is None or prmtop_file is None:
            print("ERROR: Both pdb_file and prmtop_file must be provided.")
            exit(1)
        
        u = mda.Universe(prmtop_file, pdb_file)
        
        # Initialize all B-factors to 0
        u.add_TopologyAttr("tempfactors")

        for component in components_list:
            print(f"Painting component: {component}")
            out_fname = f'{outdir}/mmpbsa_painted_{component}.pdb'
            u.atoms.tempfactors = 0.0

            # Process protein residues
            for res in u.residues:
                # Determine if the residue is in the receptor (R) or ligand (L)
                location = "R" if res.resid in df_decomp[df_decomp["location"] == "R"]["resid"].values else "L"
                
                # Filter the decomposition data for the current residue
                _df = df_decomp[(df_decomp["resid"] == res.resid) & (df_decomp["location"] == location)]
                
                if not _df.empty:
                    residue = _df.iloc[0]
                    # Check if residue names match
                    if res.resname != residue['resname']:
                        print(f'WARNING: Residue name mismatch for resid {res.resid}: '
                            f'{res.resname} (PDB) vs {residue["resname"]} (decomp)')
                        continue
                    
                    # Assign the MMGBSA component value to the B-factor
                    res.atoms.tempfactors = residue[component]

            if normalize:
                # Normalize B-factors to 0-100 range for better visualization
                b_factors = u.atoms.tempfactors
                min_b = np.min(b_factors)
                max_b = np.max(b_factors)
                u.atoms.tempfactors = 100 * (b_factors - min_b) / (max_b - min_b)
                out_fname = f'{outdir}/mmpbsa_painted_{component}_NORM.pdb'
            # Write out the new PDB with B-factors set to the decomposition values
            # Strip water, ions, and some lipids for clarity. this can be improved

            u.select_atoms("not resname HOH and not resname NA and not resname CL and not resname POP"
                        ).write(out_fname)
            # print(f"Painted PDB saved to {out_fname}")
        return
    
def _replace_line(lines: list,
                line_to_match: str = None,
                new_line: str = None,
                ):
    """
    Replace the strip_mask line in the mmpbsa input file.
    """

    # Replace the specific line
    for i, line in enumerate(lines):
        if line.startswith(line_to_match):
            lines[i] = new_line + '\n'
            break
    return lines

def calculate_contact_frequency(u, ligand_resname: str, pocket_cutoff: float = 5.0, contact_cutoff: float = 3.5) -> np.ndarray:
    """Return per-atom contact frequency (fraction of frames) for ligand heavy atoms within contact_cutoff of pocket atoms."""
    lig_ha = u.select_atoms(f'resname {ligand_resname} and not name H*')
    u.trajectory[0]
    pocket = u.select_atoms(f'protein and not name H* and around {pocket_cutoff} resname {ligand_resname}')
    contact_counts = np.zeros(len(lig_ha))
    for ts in u.trajectory:
        dmat = distance_array(lig_ha.positions, pocket.positions)
        contact_counts += (dmat < contact_cutoff).any(axis=1)
    return contact_counts / len(u.trajectory)

def calculate_ligand_rmsf(u, lig_resname: str = 'UNK') -> np.ndarray:
    """Return RMSF values for ligand heavy atoms."""
    lig_ha = u.select_atoms(f'resname {lig_resname} and not name H*')
    return RMSF(atomgroup=lig_ha).run().rmsf

def plot_atomic_property(u, weights: np.ndarray, lig_resname: str = 'UNK',
                         outname: str = 'property.png', ref_mol=None,
                         color=None, colormap=None) -> None:
    """Plot a per-atom scalar property on the ligand 2D structure and save as image.

    color: any matplotlib color ('blue', '#1f77b4', (r,g,b,a)) — sets the fill
           gradient highlight for high positive values (white → color).
           Takes precedence over colormap.
    colormap: full 3-color colourMap as a matplotlib colormap object/string,
              or a list of three RGBA tuples for [negative, zero, positive].
    """
    from rdkit import Geometry
    from rdkit.Chem import Draw, rdDepictor

    if outname.endswith('.svg'):
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    else:
        outname = outname.replace('.svg', '.png')
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)

    # make the background transparent
    # drawer.drawOptions().setBackgroundColour((1.0, 1.0, 1.0, 0.0))

    if ref_mol is not None:
        probe_mol = Chem.RemoveHs(ref_mol)
        AllChem.Compute2DCoords(probe_mol)
    else:
        lig_full = u.select_atoms(f'resname {lig_resname}')
        probe_mol = lig_full.convert_to('RDKIT', NoImplicit=False)
        probe_mol.Compute2DCoords()
        probe_mol = Chem.RemoveHs(probe_mol)
    assert len(weights) == probe_mol.GetNumAtoms(), "Mismatch between weights length and atom count"

    # Replicate the draw2d path from GetSimilarityMapFromWeights so we can
    # set the colourMap directly — that field is not exposed through the public API.
    mol_prepared = rdMolDraw2D.PrepareMolForDrawing(probe_mol, addChiralHs=False)
    if not mol_prepared.GetNumConformers():
        rdDepictor.Compute2DCoords(mol_prepared)

    conf = mol_prepared.GetConformer()
    if mol_prepared.GetNumBonds() > 0:
        bond = mol_prepared.GetBondWithIdx(0)
        p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
    else:
        p1, p2 = conf.GetAtomPosition(0), conf.GetAtomPosition(1)
    sigma = round(0.3 * (p1 - p2).Length(), 2)

    sigmas = [sigma] * mol_prepared.GetNumAtoms()
    locs = [Geometry.Point2D(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
            for i in range(mol_prepared.GetNumAtoms())]

    drawer.ClearDrawing()
    ps = Draw.ContourParams()
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 0.5

    if color is not None:
        from matplotlib.colors import to_rgba
        r, g, b, a = to_rgba(color)
        # white for zero/negative, target color for high positive values
        clrs = [(1.0, 1.0, 1.0, 0.3), (1.0, 1.0, 1.0, 0.1), (r, g, b, a)]
        ps.setColourMap(clrs)
    elif colormap is not None:
        from matplotlib import cm as mpl_cm
        if isinstance(colormap, str):
            clrs = [tuple(x) for x in mpl_cm.get_cmap(colormap)([0, 0.5, 1])]
        elif hasattr(colormap, '__call__'):
            clrs = [tuple(x) for x in colormap([0, 0.5, 1])]
        else:
            clrs = [colormap[0], colormap[1], colormap[2]]
        ps.setColourMap(clrs)

    Draw.ContourAndDrawGaussians(drawer, locs, weights.tolist(), sigmas, nContours=5, params=ps)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol_prepared)
    drawer.FinishDrawing()

    if outname.endswith('.svg'):
        with open(outname, 'w+') as outf:
            outf.write(drawer.GetDrawingText())
    else:
        drawer.WriteDrawingText(outname)
    return

def plot_rmsd(rmsd_df:pd.DataFrame=None,
              sys_name:str=None,
              out_dir:str=None) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rmsd_df, y="rmsd", x=rmsd_df.index)
    plt.ylabel("RMSD (nm)");     plt.xlabel("Frame #")
    plt.title(f"RMSD {sys_name}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_rmsd.png")
    plt.close()
    return