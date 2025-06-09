#!/bin/bash
#SBATCH --job-name=pod      # nom du job
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

source ~/environments/code_env
pod_env

set -x

data_directory="/gpfs/users/botezv/APPLICATIONS_POD/pod_computations/data_example.txt"

echo 'running pod'
srun python /home/botez18/APPLICATIONS_POD/pod_computations/codes/main.py "$data_directory"

wait
date