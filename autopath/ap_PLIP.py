import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
import pytraj as pt
from prolif import Fingerprint
from typing import List, Optional
import logging
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

    def _load_trajectories(self,
                           traj_start: Optional[int] = None,
                           traj_end: Optional[int] = None,
                           traj_step: Optional[int] = None
                           ):
        """Load each replica separately using MDAnalysis."""
        #FIXME the slicing is nto working as intended
        self.replicas = {}
        for t in self.traj_paths:
            try:
                u = mda.Universe(self.top, t)
                # u.trajectory = u.trajectory[traj_start:traj_end:traj_step]
                repname = os.path.basename(t).split('.')[0]
                self.replicas[repname] = u
            except Exception as e:
                logger.warning(f"Failed to load trajectory {t} with topology {self.top}: {e}")
                pass
            
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
        
        #plot LIE components
        self.plot_lie_components(df_all)
        
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