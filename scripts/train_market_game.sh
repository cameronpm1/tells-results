#!/bin/bash
# market_game: train the decentralized (fully observable) policy, collect the belief dataset with it, and train the
# belief model on that dataset. configs and output paths are listed in env_configs in common.sh.
#
# usage: scripts/train_market_game.sh
#   STEPS="data belief" MODEL_DIR=<checkpoint> scripts/train_market_game.sh   reuse an existing policy
#   see common.sh for RUNS / N_WORKERS / RAY_ARGS / FORCE
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
train_pipeline market_game
