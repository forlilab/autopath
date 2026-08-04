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
            List of trajectory files (each treated as an independent replica).
        ligand_mda_selection : str
            MDAnalysis selection string for the ligand (default ``"resname UNK"``).
        protein_mda_selection : str or None
            MDAnalysis selection string for the protein. Used as the ProLIF protein
            selection and to restrict LIE calculations. If None, ``"protein"`` is used.
        traj_start, traj_end, traj_step : int or None
            Frame slicing parameters (reserved for future use).
        outdir : str
            Directory for cached fingerprints and output figures.
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
        """Load each replica separately using MDAnalysis.

        Note
        ----
        Trajectory slicing (``traj_start``/``traj_end``/``traj_step``) is not
        applied here; slicing is currently handled at the point of analysis.
        """
        # A coordinate-only topology (PDB/GRO) gives MDAnalysis no atom types, so the
        # RDKit conversion ProLIF depends on perceives the ligand poorly: on these systems
        # system.pdb yields ~1.5 interactions/frame (VdWContact only) where system.prmtop
        # yields ~17.6 including Hydrophobic. Prefer a prmtop/psf.
        if str(self.top).lower().endswith((".pdb", ".gro", ".cif")):
            logger.warning(
                f"Topology {os.path.basename(str(self.top))} carries no atom types; ProLIF "
                "interaction detection will be incomplete. Use the .prmtop (or .psf) instead."
            )
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
        """Concatenate all replica trajectories into a single MDAnalysis Universe.

        Returns
        -------
        mda.Universe
            A new Universe whose trajectory is the in-memory concatenation of all replicas.
        """
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
        """Combine multiple trajectory files into a single output trajectory.

        Each input trajectory is sliced with ``start``/``stop``/``step``
        independently before being appended to the output.

        Parameters
        ----------
        topology : str
            Topology file (PRMTOP, PSF, PDB, etc.)
        trajectories : list of str
            Input trajectory files (DCD, XTC, etc.)
        output_traj : str
            Output trajectory filename.
        start, stop, step : int or None
            Slicing applied independently to each input trajectory.

        Returns
        -------
        str
            Path to the combined output trajectory file (``output_traj``).
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
        """Identify persistent interfacial water molecules across a trajectory.

        A water residue is considered interfacial in a frame when at least one
        of its atoms is within ``cutoff`` Å of the protein **and** within
        ``cutoff`` Å of the ligand simultaneously.  Only waters that are
        interfacial in at least ``fraction_persistence`` of all analyzed frames
        are returned.

        Parameters
        ----------
        u : mda.Universe
            MDAnalysis Universe with topology and trajectory loaded.
        ligand_sel : str
            MDAnalysis selection string for the ligand.
        protein_sel : str
            MDAnalysis selection string for the protein.
        traj_slice : tuple of (int, int, int) or None
            ``(start, end, step)`` frame slice. If None, all frames are analyzed.
        cutoff : float
            Distance threshold (Å) for the protein–water and ligand–water contacts.
        fraction_persistence : float
            Minimum fraction of frames in which a water must be interfacial to be
            included in the returned set (e.g. 0.9 means present in >= 90% of frames).
        pdb_fname : str or None
            If provided, write a PDB containing the protein, ligand, and persistent
            waters from the last trajectory frame to this path.

        Returns
        -------
        dict
            ``{resid: persistence_fraction}`` for waters that meet the threshold,
            sorted by descending persistence.
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

        Parameters
        ----------
        fp_interactions : list of str or None
            ProLIF interaction types to fingerprint. If None, ProLIF defaults are used.
        frequency_cutoff : float
            Only interactions present in at least this fraction of frames are kept.
        stride : int
            Frame stride passed to ProLIF. See Warning below.
        n_jobs : int
            Number of parallel workers for ProLIF. See Warning below.

        Returns
        -------
        important_resids : list of int
            Sorted residue IDs whose interaction frequency exceeds ``frequency_cutoff``.
            These are numbered as in the TOPOLOGY that was loaded. A prmtop renumbers
            sequentially from 1, so they are generally offset from the original PDB author
            numbering (e.g. 5 for a structure whose first modelled residue is 6) -- convert
            before comparing against crystal-structure residue numbers.
        persistence_byRes : pd.DataFrame
            Interaction persistence grouped by residue (all interaction types merged).
        persistence_byRes_byType : pd.DataFrame
            Interaction persistence broken down by residue and interaction type.

        Warning
        -------
        The ``stride`` parameter is currently not applied correctly by ProLIF; all
        frames in each replica trajectory are analyzed regardless of the value passed.

        The ``n_jobs`` parameter is currently non-functional due to serialization
        incompatibilities (dill/multiprocess version conflicts in ProLIF). All
        fingerprint computations run serially regardless of the value passed.
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
                
                # ProLIF's signature is run(traj, lig, prot). Passing the protein first put
                # protein residues under the "ligand" level and reduced detection to
                # VdWContact only (no Hydrophobic, no H-bonds).
                fp = fp.run(u.trajectory,
                            ligand_sel,
                            protein_sel,
                            n_jobs=n_jobs,
                            )
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
            # groupby(level=["protein", "ligand"]) puts the protein residue first; the old
            # code read res[1] (the ligand) and sliced res[3:], which breaks on ProLIF's
            # chain-suffixed labels such as "ALA50.A" -> int("50.A").
            selected_resnames = [res[0] for res in selected_residues]
            selected_resids = [int(re.search(r"(\d+)", r).group(1))
                               for r in selected_resnames if re.search(r"(\d+)", r)]
            important_resids.update(selected_resids)

        return sorted(list(important_resids)), persistence_byRes, persistence_byRes_byType

    @staticmethod
    def plot_tanimoto_similarity(query_fp, reference_fp, use_frame: int = None, outdir: str = None):
        """Plot Tanimoto similarity between ProLIF fingerprints and save the figure(s).

        Parameters
        ----------
        query_fp : prolif.Fingerprint
            Fingerprint to compare against (all frames).
        reference_fp : prolif.Fingerprint or None
            Reference fingerprint. If None, ``query_fp`` is used as its own reference
            (self-similarity matrix).
        use_frame : int or None
            If provided, compute similarity of every query frame to this reference frame
            and plot a line graph. If None, compute and plot the full similarity matrix.
        outdir : str or None
            Directory for output figures. Defaults to current directory.

        Returns
        -------
        list of float or pd.DataFrame
            If ``use_frame`` is set: list of per-frame Tanimoto similarities.
            Otherwise: symmetric similarity matrix as a DataFrame.
        """
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
        """Compute Linear Interaction Energy (LIE) for each replica using pytraj.

        LIE approximates binding free energy as a linear combination of
        electrostatic and van der Waals interaction energies between the ligand
        and the surrounding environment.  See:

        - https://ambermd.org/tutorials/advanced/tutorial24/liew.php
        - https://pubs.acs.org/doi/10.1021/acs.jcim.9b00609
        - https://pmc.ncbi.nlm.nih.gov/articles/PMC7311763/

        Parameters
        ----------
        prmtop : str or None
            Amber PRMTOP topology for pytraj. Defaults to ``self.top``.
        use_residues : list of int or None
            Explicit list of residue numbers to include in the LIE environment
            mask. If None, residues within ``cutoff`` Å of the ligand are
            selected automatically.
        ligand_amber_selection : str
            Amber mask for the ligand (e.g. ``":UNK"``).
        exclude_amber_selection : str or None
            Amber mask of atoms to exclude from the auto-selected environment
            (ions, counter-ions, etc.).
        cutoff : float
            Distance cutoff (Å) used to auto-select environment residues when
            ``use_residues`` is None.
        stride : int
            Frame stride passed to ``pt.iterload``.
        lie_options : str
            Additional options string passed to ``pt.analysis.energy_analysis.lie``
            (e.g. ``'nopbc cutvdw 10.0 cutelec 10.0'``).

        Returns
        -------
        pd.DataFrame
            Combined LIE results for all replicas with columns
            ``EELEC``, ``VDW``, ``Total``, and ``run``.  Also saved to
            ``{outdir}/LIE_results.csv``.
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

    def plot_lie_components(self, lie_df: pd.DataFrame) -> None:
        """Plot LIE energy components (Total, EELEC, VDW) over time and save the figure.

        Parameters
        ----------
        lie_df : pd.DataFrame
            Output of :meth:`compute_LIE`, with columns ``Total``, ``EELEC``, ``VDW``.
        """
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
            slurm_template_fname: str = None,
            ):
        """Prepare a per-system MMPBSA SLURM qfile and a master batch script.

        Writes a modified MMPBSA input file (with correct frame range and optional
        persistent-water strip mask) and a SLURM submission script, then updates
        a master bash script that submits all qfiles in the ``qfiles_mmgbsa/``
        directory with ``sbatch``.

        Parameters
        ----------
        sysname : str
            Identifier for this system (used in output filenames).
        prmtop : str
            Amber topology file (.prmtop).
        traj_fname : str
            Trajectory file (DCD, XTC, NC, …).
        ligand_amber_selection : str
            Amber mask for the ligand (e.g. ``":UNK"``).
        ligand_mda_selection : str or None
            MDAnalysis selection string for the ligand. Derived from
            ``ligand_amber_selection`` if not provided.
        strip_amber_selection : str
            Amber mask of residues to strip before MMPBSA (waters, ions, lipids).
        traj_slice : tuple of (int, int, int) or None
            ``(start, end, step)`` frame slice applied to the trajectory.
        persistent_waters_cutoff : float or None
            If provided, waters whose occupancy at the protein–ligand interface
            exceeds this fraction are kept in the MMPBSA calculation. Waters are
            identified with :meth:`find_interfacial_waters`.
        mmpbsa_in : str
            Path to the MMPBSA input template file.
        radii : str
            Implicit-solvent radii set (e.g. ``'mbondi2'``).
        output_folder : str
            Directory for MMPBSA output and the modified input file.
        bash_fname : str
            Filename of the master ``sbatch`` batch script.
        mpi_threads : int
            Maximum number of MPI threads. Capped at the number of frames.
        slurm_template_fname : str or None
            Path to a custom SLURM template. Defaults to the bundled template.

        Returns
        -------
        str
            Path to the written master batch script (``bash_fname``).
        """
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
    def parse_mmpbsa_differences_table(path: str) -> pd.DataFrame:
        """Parse the 'Differences' table from an AMBER MMPBSA results file.

        Reads the ``Differences (Complex - Receptor - Ligand):`` section from
        ``FINAL_RESULTS_mmpbsa.dat`` and returns one row per energy component.

        Parameters
        ----------
        path : str
            Path to ``FINAL_RESULTS_mmpbsa.dat``.

        Returns
        -------
        pd.DataFrame
            Columns: ``Component``, ``Average``, ``Std_Dev``, ``Std_Err_Mean``.

        Raises
        ------
        ValueError
            If the Differences section is not found, or if no data rows are parsed.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Differences (Complex - Receptor - Ligand):"):
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("Could not find 'Differences (Complex - Receptor - Ligand):' section.")

        # Advance to the first data line (after the dashed separator)
        i = start_idx + 1
        while i < len(lines):
            if re.match(r"^-{5,}\s*$", lines[i].strip()):  # line of dashes
                i += 1
                break
            i += 1

        # Parse rows: component name (possibly with spaces) followed by 3 floats
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
    def parse_mmpbsa_deltas_all_components(filepath: str = None) -> pd.DataFrame:
        """Parse the DELTAS section of an AMBER MMPBSA per-residue decomposition file.

        Reads the CSV-formatted ``DELTAS:`` block produced by AMBER's MMPBSA
        per-residue decomposition (PB or GB), and returns one row per residue.

        Parameters
        ----------
        filepath : str
            Path to the MMPBSA per-residue decomposition output file (typically
            ``FINAL_DECOMP_MMPBSA.dat``).

        Returns
        -------
        pd.DataFrame
            Columns: ``resname``, ``resid``, ``location``, and one ``{Component}_Avg``,
            ``{Component}_StdDev``, ``{Component}_StdErr`` column per energy group
            (e.g. ``Internal``, ``van_der_Waals``, ``Electrostatic``, etc.).
            An additional ``label`` column is appended (``resname + resid``).

        Raises
        ------
        ValueError
            If the ``DELTAS:`` section is not found or the header format is unexpected.
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
    def plot_mmpbsa_byresidue(df_decomp: pd.DataFrame,
                            top_residues: int = 10,
                            out_dir: str = None
                            ):
        """Plot per-residue MMGBSA energy decomposition as bar charts.

        For each energy component and each molecular location (receptor, ligand),
        saves one bar chart showing the ``top_residues`` most favorable and most
        unfavorable residues.

        Parameters
        ----------
        df_decomp : pd.DataFrame
            Output of :meth:`parse_mmpbsa_deltas_all_components`.
        top_residues : int
            Number of top and bottom residues to display per component.
        out_dir : str or None
            Output directory for PNG files. Defaults to the current directory.

        Note
        ----
        The ``'L'`` (ligand) location is included for protein–protein interaction
        studies where both partners are decomposed.
        """
        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        mmpbsa_components_list = ['TOTAL_Avg', 'Electrostatic_Avg', "van_der_Waals_Avg",
                    # 'Internal_Avg' is omitted — not informative for binding
                    "Polar_Solvation_Avg", "Non_Polar_Solv_Avg"]

        locations = {'R': 'receptor', 'L': 'ligand'}
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
        """Color a PDB structure by per-residue MMGBSA decomposition values via B-factor.

        Parameters
        ----------
        df_decomp : pd.DataFrame
            Output of :meth:`parse_mmpbsa_deltas_all_components`, containing per-residue
            MMGBSA energy components.
        pdb_file : str
            Path to the PDB file to paint.
        prmtop_file : str
            Path to the Amber topology file (.prmtop).
        mmpbsa_component : str
            Energy component column to write into the B-factor field. Use ``'all'`` to
            iterate over all standard components. One of
            ``{'TOTAL_Avg', 'Electrostatic_Avg', 'van_der_Waals_Avg',
            'Polar_Solvation_Avg', 'Non_Polar_Solv_Avg', 'all'}``.
        normalize : bool
            If True, rescale B-factors to [0, 100] for easier VMD/PyMOL visualization.
        outdir : str or None
            Directory for output PDB files. Defaults to current directory.

        Raises
        ------
        ValueError
            If ``pdb_file`` or ``prmtop_file`` is None, or if ``mmpbsa_component``
            is not a recognized component name.
        """
        mmpbsa_components_list = ['TOTAL_Avg', 'Electrostatic_Avg', "van_der_Waals_Avg",
                            # 'Internal_Avg' is omitted — not informative for binding
                            "Polar_Solvation_Avg", "Non_Polar_Solv_Avg"]
        if outdir is None:
            outdir = '.'

        os.makedirs(outdir, exist_ok=True)

        if mmpbsa_component.upper() == 'ALL':
            components_list = mmpbsa_components_list
        else:
            if mmpbsa_component not in mmpbsa_components_list:
                raise ValueError(
                    f"mmpbsa_component must be one of {mmpbsa_components_list} or 'all', "
                    f"got '{mmpbsa_component}'."
                )
            else:
                components_list = [mmpbsa_component]

        if pdb_file is None or prmtop_file is None:
            raise ValueError("Both pdb_file and prmtop_file must be provided.")

        u = mda.Universe(prmtop_file, pdb_file)

        # Initialize all B-factors to 0
        u.add_TopologyAttr("tempfactors")

        for component in components_list:
            logger.info(f"Painting component: {component}")
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
                    if res.resname != residue['resname']:
                        logger.warning(
                            f'Residue name mismatch for resid {res.resid}: '
                            f'{res.resname} (PDB) vs {residue["resname"]} (decomp)'
                        )
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
    """Replace the first line that starts with ``line_to_match`` in an MMPBSA input template.

    Parameters
    ----------
    lines : list of str
        Lines of the MMPBSA input file (as returned by ``file.readlines()``).
    line_to_match : str
        Prefix of the line to replace (e.g. ``'startframe'``).
    new_line : str
        Replacement text (without trailing newline).

    Returns
    -------
    list of str
        Modified line list with the first matching line replaced.
    """
    for i, line in enumerate(lines):
        if line.startswith(line_to_match):
            lines[i] = new_line + '\n'
            break
    return lines

def calculate_contact_frequency(u, ligand_selection: str, pocket_cutoff: float = 5.0, contact_cutoff: float = 3.5) -> np.ndarray:
    """Return per-atom contact frequency for ligand heavy atoms.

    A ligand atom is counted as "in contact" in a frame if any pocket atom
    lies within ``contact_cutoff`` Å of it.  Pocket atoms are protein heavy
    atoms within ``pocket_cutoff`` Å of the ligand in the first frame.

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe with topology and trajectory loaded.
    ligand_selection : str
        MDAnalysis selection string for the ligand.
    pocket_cutoff : float
        Distance (Å) used to define the pocket in the first frame.
    contact_cutoff : float
        Distance threshold (Å) for declaring a contact in each frame.

    Returns
    -------
    np.ndarray
        Fraction of trajectory frames in which each ligand heavy atom is in
        contact with any pocket atom. Shape ``(n_ligand_heavy_atoms,)``.
    """
    lig_ha = u.select_atoms(f'({ligand_selection}) and not name H*')
    u.trajectory[0]
    pocket = u.select_atoms(f'protein and not name H* and around {pocket_cutoff} ({ligand_selection})')
    contact_counts = np.zeros(len(lig_ha))
    for ts in u.trajectory:
        dmat = distance_array(lig_ha.positions, pocket.positions)
        contact_counts += (dmat < contact_cutoff).any(axis=1)
    return contact_counts / len(u.trajectory)

def calculate_ligand_rmsf(u, ligand_selection: str = 'resname UNK') -> np.ndarray:
    """Return per-atom RMSF values for ligand heavy atoms.

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe with topology and trajectory loaded.
    ligand_selection : str
        MDAnalysis selection string for the ligand.

    Returns
    -------
    np.ndarray
        RMSF (Å) for each ligand heavy atom over the full trajectory.
        Shape ``(n_ligand_heavy_atoms,)``.
    """
    lig_ha = u.select_atoms(f'({ligand_selection}) and not name H*')
    return RMSF(atomgroup=lig_ha).run().rmsf

def plot_atomic_property(u, weights: np.ndarray, lig_resname: str = 'resname UNK',
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
        lig_full = u.select_atoms(lig_resname)
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

def plot_rmsd(rmsd_df: pd.DataFrame = None,
              sys_name: str = None,
              out_dir: str = None) -> None:
    """Plot RMSD over time and save the figure.

    Parameters
    ----------
    rmsd_df : pd.DataFrame
        DataFrame with at least an ``rmsd`` column; the index is used as the x-axis.
    sys_name : str
        System name used in the plot title and output filename.
    out_dir : str
        Directory where the PNG is saved.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rmsd_df, y="rmsd", x=rmsd_df.index)
    plt.ylabel("RMSD (nm)");     plt.xlabel("Frame #")
    plt.title(f"RMSD {sys_name}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sys_name}_rmsd.png")
    plt.close()
    return