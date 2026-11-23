"""Write one SLURM job per equilibrated system running ``run_LIE.py``.

Usage
-----
    python orchestrator_LIE.py [--eq_root DIR] [--resname UNK] [--partition P] [--time D-H]
    bash run_LIE_batch.sh
"""
import argparse
import os
from glob import glob

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_LIE.py")


def write_script(ligname, eq_root, resname, time, partition, workdir):
    template = '''#!/bin/bash
#SBATCH -e ${workdir}/${ligname}/lie/${ligname}.err
#SBATCH -o ${workdir}/${ligname}/lie/${ligname}.out
#SBATCH --time=${time}
#SBATCH --partition=${partition}
#SBATCH --job-name="LIE_${ligname}"

# EDIT FOR YOUR CLUSTER
source ~/.bashrc
micromamba activate autopath

cd ${workdir}
python ${script} -s ${ligname} --eq_root ${eq_root} --resname ${resname}
'''
    for key, val in dict(ligname=ligname, time=time, partition=partition, workdir=workdir,
                         script=SCRIPT, eq_root=eq_root, resname=resname).items():
        template = template.replace("${%s}" % key, val)
    # SLURM does not create the -e/-o directory
    os.makedirs(f"{ligname}/lie", exist_ok=True)
    with open(f"qfiles_LIE/{ligname}_LIE.q", "w") as f:
        f.write(template)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eq_root", default="../01_Build_and_Equilibrate",
                    help="directory holding one <sys>/ folder per equilibrated system")
    ap.add_argument("--resname", default="UNK", help="ligand residue name (as in 01_Build_and_Equilibrate)")
    ap.add_argument("--partition", default="forli,forli-pro,alphafold,shared", help="SLURM partition(s)")
    ap.add_argument("--time", default="1-0", help="SLURM time limit")
    args = ap.parse_args()

    eq_root = os.path.abspath(args.eq_root)
    workdir = os.getcwd()
    trajectories = sorted(glob(f"{eq_root}/*/equilibration/equilibration_*_aligned.dcd"))
    print(f"Found {len(trajectories)} trajectories to process.")

    out_fname = "run_LIE_batch.sh"
    os.makedirs('qfiles_LIE', exist_ok=True)

    with open(out_fname, "w") as f:
        f.write("#!/bin/bash\n\n")
        for trajectory in trajectories:
            sysname = os.path.basename(os.path.dirname(os.path.dirname(trajectory)))
            write_script(sysname, eq_root, args.resname, args.time, args.partition, workdir)
            f.write(f"sbatch {workdir}/qfiles_LIE/{sysname}_LIE.q\n")

    os.chmod(out_fname, 0o755)


if __name__ == "__main__":
    main()
