#!/bin/bash
# football: train the decentralized (fully observable) policy, collect the belief dataset with it, and train the
# belief model on that dataset. configs and output paths are listed in env_configs in common.sh.
#
# usage: scripts/train_football.sh
#   STEPS="data belief" MODEL_DIR=<checkpoint> scripts/train_football.sh   reuse an existing policy
#   see common.sh for RUNS / N_WORKERS / RAY_ARGS / FORCE
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
train_pipeline football
