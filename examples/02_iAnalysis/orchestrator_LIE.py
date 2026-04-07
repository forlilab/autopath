import os
from glob import glob

def write_script(
                ligname:str=None, 
                gpu_resource="rtxa6000", 
                gpu_num=1, 
                time="3-0", 
                partition="forli,forli-pro,alphafold,shared"
                ):
    
    template='''#!/bin/bash
#SBATCH -e ${ligname}/equilibration/lie/${ligname}.err
#SBATCH -o ${ligname}/equilibration/lie/${ligname}.out
#SBATCH --time=${time}
#SBATCH --partition=${partition}
#SBATCH --job-name="LIE_${ligname}"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath
python run_LIE.py -s ${ligname}
'''

    with open(f"qfiles_LIE/{ligname}.q", "w") as f:
        template = template.replace("${ligname}", ligname)
        template = template.replace("${gpu_resource}", gpu_resource)
        template = template.replace("${gpu_num}", str(gpu_num))
        template = template.replace("${time}", time)
        template = template.replace("${partition}", partition)
        f.write(template)

    return

basename = 'HSP90_OFF-LIE'
ligands = glob(f'input/*.sdf')
# receptor = f'input/HAstV2_prep.pdb'

out_fname = f"run_ap_{basename}.sh"
os.makedirs('qfiles_LIE', exist_ok=True)

# folders = os.listdir('./')
# ligands = [lig for lig in ligands if os.path.basename(lig).split('.')[0] not in folders]
print(f"Found {len(ligands)} ligands to process.")

with open(out_fname, "w") as f:
    f.write("#!/bin/bash\n\n")

    for ligand in ligands:
        ligname = os.path.basename(ligand).split('.')[0]       
        write_script(ligname)
        f.write(f"sbatch qfiles_LIE/{ligname}.q\n")

os.chmod(out_fname, 0o755)
