#!/bin/bash
#SBATCH --job-name=keever
#SBATCH --output=logs/%x.%A_%a.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --hint=multithread
#SBATCH --array=0-100
 # 32
 # 48h
set -e

. config


export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=4
export OPTIMIZER=pso
export POLARIZATIONS="['X','Y','RCP','LCP']"
export SEED=$SLURM_ARRAY_TASK_ID
export WORKDIR="./data/gratings_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"
export BUDGET=30000

mkdir -p $WORKDIR
envsubst < projects/multigrating.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml
