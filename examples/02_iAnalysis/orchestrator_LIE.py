from builtins import str

import os
from glob import glob

def write_script(ligname:str=None, 
                time:str="1-0", 
                partition:str="forli,forli-pro,alphafold,shared"
                ):
    
    template='''#!/bin/bash
#SBATCH -e ${ligname}/lie/${ligname}.err
#SBATCH -o ${ligname}/lie/${ligname}.out
#SBATCH --time=${time}
#SBATCH --partition=${partition}
#SBATCH --job-name="LIE_${ligname}"

source ~/.bashrc
micromamba activate autopath
python run_LIE.py -s ${ligname}
'''

    with open(f"qfiles_LIE/{ligname}_LIE.q", "w") as f:
        template = template.replace("${ligname}", ligname)
        template = template.replace("${time}", time)
        template = template.replace("${partition}", partition)
        f.write(template)

    return

trajectories = glob(f"../01_Build_and_Equilibrate/*/equilibration/equilibration_*_aligned.dcd")
print(f"Found {len(trajectories)} trajectories to process.")

out_fname = f"run_LIE_batch.sh"
os.makedirs('qfiles_LIE', exist_ok=True)

with open(out_fname, "w") as f:
    f.write("#!/bin/bash\n\n")

    for trajectory in trajectories:
        sysname = os.path.basename(os.path.dirname(os.path.dirname(trajectory)))
        write_script(sysname)
        f.write(f"sbatch qfiles_LIE/{sysname}_LIE.q\n")

os.chmod(out_fname, 0o755)
