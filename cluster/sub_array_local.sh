#!/bin/bash
set -e
source config
export TQDM_DISABLE=1
export LAYERS=16
export OPTIMIZER=pso
export SEED=10 # $SLURM_ARRAY_TASK_ID
export BUDGET=5000
export POLARIZATIONS="['X']"  
export WORKDIR="./data/local_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"

mkdir -p $WORKDIR
envsubst < projects/local_opt.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml
