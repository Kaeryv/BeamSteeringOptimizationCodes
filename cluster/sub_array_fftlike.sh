#!/bin/bash
#SBATCH --job-name=mini_5_y
#SBATCH --output=logs/%x.%A_%a.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --hint=multithread
#SBATCH --array=0-30
 # 32
 # 48h
set -e

. config


export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=16
export OPTIMIZER=pso
export POLARIZATIONS="['X','Y','RCP','LCP']"
export SEED=$SLURM_ARRAY_TASK_ID
export WORKDIR="${GLOBALSCRATCH}/bs/gratings_${SLURM_JOB_NAME}_${LAYERS}_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"
export CONTROLLER=none #wd/controllers/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_120.json

export BUDGET=30000

#[ ! -f "$CONTROLLER" ] && python train_optimizer.py $SLURM_ARRAY_TASK_ID $BUDGET $CONTROLLER 1000001

mkdir -p $WORKDIR
envsubst < projects/multigrating.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml
