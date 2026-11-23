#!/usr/bin/env python
"""Submit the membrane-permeation build and start generation on Slurm.

    python submit.py ethanol CCO
    python submit.py domperidone /path/to/domperidone.sdf --replicas 2
    python submit.py ligands.csv                  # one ligand per row: name,smiles_or_sdf
    python submit.py ethanol CCO --dry-run        # print the jobs, submit nothing

Per replica, each stage waits for the previous one (Slurm afterok):
  1. build      prepare_permeation_system.py --both-sides     -> md_input/
  2. starts     make_starts.py, + and - side as parallel jobs  -> equilibration_+-2.7/, smd_output_+-1.0/,
                                                                  equilibration_+-1.0/
Every replica is independent (own build, equilibration and SMD).
The metadynamics runs on these starts live in HADES ``workflows/metad/permeability``.

    NAME_replica_1/
      md_input/  equilibration_2.7/  equilibration_-2.7/  smd_output_1.0/  smd_output_-1.0/
      equilibration_1.0/  equilibration_-1.0/
      build.out  equilibration_plus.out ...     (logs, with .err)

Stages already done are skipped, so rerunning the same command resumes.
"""
import argparse
import csv
import os
import shlex
import subprocess

EX = "/gpfs/home/abarkdull/software/autopath-membrane/examples/08_Membrane-Permeation"
ENV = "autopath-membrane"
MEMBRANE = "50_lip_per_leaf_100_percent_POPC_20A_wat"
SLURM = ["--partition=forli-pro,forli,alphafold", "--gres=gpu:1", "--cpus-per-task=4", "--mem=16G",
         "--exclude=nodea0110,nodea0111,nodea0410,"       # GTX 1080 Ti, Titan V, Quadro RTX 6000: slow
         "nodeb[201,203,205,207,209,211,213,301,303,305,307,309,311,601,603,605,607,609,611,613,619]"]  # RTX A5000: slow
SIDES = {"plus": ["2.7", "1.0"], "minus": ["-1.0", "-2.7"]}


def sbatch(name, log, cmd, time, after=(), dry=False):
    """Submit cmd as a GPU job; returns its job id."""
    args = ["sbatch", "--parsable", f"--job-name={name}", f"--time={time}", *SLURM,
            f"--output={log}.out", f"--error={log}.err"]
    if after:
        args.append("--dependency=afterok:" + ":".join(after))
    args.append(f"--wrap=source ~/.bashrc; micromamba activate {ENV}; set -eu; hostname; {cmd}")
    if dry:
        print("  " + " ".join(shlex.quote(x) for x in args))
        return name
    job = subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    print(f"  {job:>10}  {name}")
    return job


def submit(name, ligand, r, a):
    root = os.path.abspath(f"{name}_replica_{r}")
    tag = f"{name}_r{r}"
    os.makedirs(root, exist_ok=True)
    print(os.path.basename(root))
    done = lambda *p: os.path.exists(os.path.join(root, *p))

    # 1. build
    build = []
    if not (done("md_input", "system.xml") and done("md_input", "system_minus.pdb")):
        lig = f"--sdf {os.path.abspath(ligand)}" if ligand.endswith(".sdf") else f"--smiles {shlex.quote(ligand)}"
        cmd = (f"python -u {EX}/prepare_permeation_system.py {lig} --membrane {a.membrane} "
               f"--z 2.7 --both-sides --out {root}/md_input")
        build = [sbatch(f"{tag}_build", f"{root}/build", cmd, "2:00:00", dry=a.dry_run)]
    if a.stop_after == "build":
        return

    # 2. starts, one job per side
    starts = []
    for side, zs in SIDES.items():
        if not all(done(f"equilibration_{z}", "state.xml") for z in zs):
            cmd = f"python -u {EX}/make_starts.py {root}/md_input --side {side}"
            if a.protocol:
                cmd += f" --protocol {os.path.abspath(a.protocol)}"
            starts.append(sbatch(f"{tag}_{side}", f"{root}/equilibration_{side}", cmd, "6:00:00",
                                 after=build, dry=a.dry_run))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="ligand name, or a CSV of name,smiles_or_sdf rows")
    ap.add_argument("ligand", nargs="?", help="SMILES or path to an .sdf (not with a CSV)")
    ap.add_argument("--membrane", default=MEMBRANE)
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--protocol", help="equilibration JSON for make_starts.py (default: its own)")
    ap.add_argument("--stop-after", choices=["build"], help="submit only up to this stage")
    ap.add_argument("--dry-run", action="store_true", help="print the sbatch commands, submit nothing")
    a = ap.parse_args()

    if a.name.endswith(".csv"):
        with open(a.name) as f:
            rows = [r[:2] for r in csv.reader(f)                 # name, smiles_or_sdf; extra columns ignored
                    if r and not r[0].startswith("#") and r[0] != "name"]
    elif a.ligand:
        rows = [[a.name, a.ligand]]
    else:
        ap.error("give NAME LIGAND, or a CSV")
    for name, ligand in rows:
        for r in range(1, a.replicas + 1):
            submit(name.strip(), ligand.strip(), r, a)


if __name__ == "__main__":
    main()
