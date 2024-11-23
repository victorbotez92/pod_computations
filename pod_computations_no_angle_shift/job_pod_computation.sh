#!/bin/bash 
#SBATCH --job-name=test_pod
#SBATCH -o ./JobLogs/%x.o
#SBATCH -e ./JobLogs/%x.err
#SBATCH --exclusive
#SBATCH --ntasks=16
#SBATCH --partition=mem_short
#SBATCH --time=1:00:00


cd ${SLURM_SUBMIT_DIR}


date
module purge
module load anaconda3/2022.10/gcc-11.2.0
module load openmpi/3.1.6/gcc-11.2.0

source /gpfs/users/botezv/.venv/pod/bin/activate

#python POD_fourier_parallel.py 


set -x
# date
# srun python compute_L2_norm.py
# wait
# echo 'done calculating L2 norms'
date

data_directory="my_data_directory"

srun python -m memory_profiler /gpfs/users/botezv/APPLICATION_POD/POD_on_fourier.py "$data_directory"
# srun python /gpfs/users/botezv/pod_entirely_auto/extract_modes.py "$data_directory"
echo 'running pod'

wait

date