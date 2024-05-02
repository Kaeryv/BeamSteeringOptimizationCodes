#!/usr/bin/env bash
set -e
export NCPUS=10
# Disable multithreading
# We use EP workload instead
. config
python -m keever.play --project projects/multigrating.yml
