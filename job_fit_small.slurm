#!/bin/bash
#SBATCH --job-name=fit_coupled_LDS_small     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G is default)
#SBATCH --time=09:00:00         # total run time limit (HH:MM:SS)
#SBATCH --array=0-269             # job array with index values 0, 1, ..., 10
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=lc3919@cumc.columbia.edu
#SBATCH --output=/home/lc3919/out_files/%x-%j.out

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)


ml load anaconda3-2019.03
conda activate lds_env

python fit_multiple_init_small.py