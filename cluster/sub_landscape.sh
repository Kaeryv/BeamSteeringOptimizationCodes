#!/bin/bash
#SBATCH --job-name=landscape
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-4

set -e

. config

export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1

export PW=7
python fitness_landscape.py c ${PW} fl_ell_${PW}_${SLURM_ARRAY_TASK_ID}_8.npz $SLURM_ARRAY_TASK_ID
python fitness_landscape.py p ${PW} fl_ell_${PW}_${SLURM_ARRAY_TASK_ID}_8.npz $SLURM_ARRAY_TASK_ID