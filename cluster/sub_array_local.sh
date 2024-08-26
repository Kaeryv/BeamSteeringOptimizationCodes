#!/bin/bash
#SBATCH --job-name=keever
#SBATCH --output=logs/debug_%x.%A_%a.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=0-20

set -e
source config
export NCPUS=$(( 2 * $SLURM_CPUS_PER_TASK))
export TQDM_DISABLE=1
export LAYERS=32
export OPTIMIZER=pso
export SEED=$SLURM_ARRAY_TASK_ID
export BUDGET=$(( 40 * 40))
export POLARIZATIONS="['X']"
export WORKDIR="./data/both4_${LAYERS}_${OPTIMIZER}/free_pixmap_$SEED/"
export PCAMODEL="./data/both4_${LAYERS}_${OPTIMIZER}/pca_${SEED}.pkl"
rm -f $PCAMODEL
mkdir -p $WORKDIR
envsubst < projects/local_opt.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml
