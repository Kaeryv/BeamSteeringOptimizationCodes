#!/bin/bash
#SBATCH --job-name=keever
#SBATCH --output=logs/%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --array=0-10
    
set -e

. config


export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=2
export OPTIMIZER=pso
export SEED=$SLURM_ARRAY_TASK_ID
export BUDGET=4000
export NUM_ITEMS=2
export POLARIZATIONS="['Y']"
export WORKDIR="./data/2Dx_${NUM_ITEMS}_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"

mkdir -p $WORKDIR
envsubst < projects/2D.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml
