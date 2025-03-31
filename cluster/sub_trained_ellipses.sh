#!/bin/bash

#SBATCH --job-name=cma_0
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-30

set -e

. config

export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=16
export OPTIMIZER=cma
export SEED=$SLURM_ARRAY_TASK_ID
export BUDGET=1000
export NUM_ITEMS=14
export POLARIZATIONS="['X','Y','RCP','LCP']"
export WORKDIR="${GLOBALSCRATCH}/bs/ellipsis_${SLURM_JOB_NAME}_${NUM_ITEMS}_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"
export CONTROLLER=${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.json

#python train_optimizer.py $SLURM_ARRAY_TASK_ID $BUDGET $CONTROLLER 1000001

mkdir -p $WORKDIR
envsubst < projects/ellipses.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml ${files[$SLURM_ARRAY_TASK_ID]}
exit 0