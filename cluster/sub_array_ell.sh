#!/bin/bash
#SBATCH --job-name=keever
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=0-50
    
set -e

. config


export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=16
export OPTIMIZER=pso
export SEED=$SLURM_ARRAY_TASK_ID
export BUDGET=5000
export NUM_ITEMS=1
export POLARIZATIONS="['Y']"  
export WORKDIR="./data/el1_y_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"

mkdir -p $WORKDIR
envsubst < projects/ellipses.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml
