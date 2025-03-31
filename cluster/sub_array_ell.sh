#!/bin/bash
#SBATCH --job-name=keever
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-80

set -e

. config
#2h

export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=16
export OPTIMIZER=fpso
export SEED=$SLURM_ARRAY_TASK_ID
export BUDGET=20000
export NUM_ITEMS=14
export POLARIZATIONS="['X','Y','RCP','LCP']"
export WORKDIR="${GLOBALSCRATCH}/data/ellipsis_${SLURM_JOBNAME}_${NUM_ITEMS}_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"
export CONTROLLER=explore_${SLURM_ARRAY_TASK_ID}.json

python train_optimizer.py $SLURM_ARRAY_TASK_ID $BUDGET

mkdir -p $WORKDIR
envsubst < projects/ellipses.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml ${files[$SLURM_ARRAY_TASK_ID]}