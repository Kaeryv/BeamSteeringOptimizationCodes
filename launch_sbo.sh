#!/usr/bin/env bash
set -e
source env/bin/activate
python -m keever.play --project projects/multigrating.yml
