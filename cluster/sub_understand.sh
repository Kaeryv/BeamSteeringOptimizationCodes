#!/bin/bash
#SBATCH --job-name=under
#SBATCH --output=logs/%x.%A_%a.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --hint=multithread
#SBATCH --array=0-11

set -e

. config


export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1

python user/understand/twisted.py sweep $(( 50 * $SLURM_ARRAY_TASK_ID))