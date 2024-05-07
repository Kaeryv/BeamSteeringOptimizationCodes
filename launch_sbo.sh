#!/usr/bin/env bash
set -e
export NCPUS=10
# Disable multithreading
# We use EP workload instead
export LAYERS=12
export OPTIMIZER=pso
export SEED=81
export WORKDIR="./data/free.pixmap.h3.$OPTIMIZER.$LAYERS.$SEED/"
mkdir -p $WORKDIR
. config
envsubst < projects/multigrating.yml > $WORKDIR/project.yml
python -m keever.play --project $WORKDIR/project.yml