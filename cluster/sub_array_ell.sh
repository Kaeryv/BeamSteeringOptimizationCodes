#!/bin/bash
#SBATCH --job-name=keever
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-100

set -e

. config
#2h

export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=16
export OPTIMIZER=pso
export SEED=$SLURM_ARRAY_TASK_ID
export BUDGET=30000
export NUM_ITEMS=14
export POLARIZATIONS="['X','Y','RCP','LCP']"
export WORKDIR="./data/ellipsis_largepop_${NUM_ITEMS}_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"


mkdir -p $WORKDIR
envsubst < projects/ellipses.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml ${files[$SLURM_ARRAY_TASK_ID]}