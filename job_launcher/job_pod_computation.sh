#!/bin/bash 
#SBATCH --job-name=test
#SBATCH -o ./JobLogs/%x.o
#SBATCH -e ./JobLogs/%x.err
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --partition=cpu_short
#SBATCH --time=1:00:00


cd ${SLURM_SUBMIT_DIR}


date
module purge
module load anaconda3/2022.10/gcc-11.2.0
module load openmpi/3.1.6/gcc-11.2.0

source /gpfs/users/botezv/.venv/pod/bin/activate

set -x

data_directory="/gpfs/users/botezv/APPLICATIONS_POD/pod_computations/job_launcher/data_hydro.txt"

echo 'running pod'
# srun python /gpfs/users/botezv/APPLICATIONS_POD/pod_computations_test_angle_shift_v3/initialization.py "$data_directory"
srun python /gpfs/users/botezv/APPLICATIONS_POD/pod_computations/codes/initialization.py "$data_directory"


wait

date