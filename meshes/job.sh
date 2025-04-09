#!/bin/bash
#SBATCH --job-name=sym_points      # nom du job
### SBATCH -N 2
###SBATCH --ntasks-per-node=128        
#SBATCH --ntasks=1
#SBATCH --time=5:00:00            # max execution time (HH:MM:SS)
#SBATCH --output=JobLogs/%x.o%j  # job returns
#SBATCH --error=JobLogs/%x.err%j   # errors
#SBATCH --partition=milan       # intel/intel_32/defq/mem768/rome/milan/genoa
#SBATCH --exclusive
#SBATCH -A fwd


cd ${SLURM_SUBMIT_DIR}


date
module purge
# module load openmpi/4.1.6

module load python/3.12.2
# source /gpfs/users/botezv/.venv/pod/bin/activate

set -x


echo 'finding sym points'
srun python /home/botez18/APPLICATIONS_POD/pod_computations/meshes/find_sym_points.py

wait
date