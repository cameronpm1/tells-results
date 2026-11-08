#!/bin/bash
# shared helpers for the scripts in this directory; source it, don't run it.
#
# every path (policy logdir, belief data name, belief checkpoint dir) is read from the configs, so these
# scripts follow the configs instead of duplicating them.
#
# environment variables (all optional):
#   PY         python interpreter            (default: the tells conda env)
#   RUNS       episodes per collection/eval  (default: 1000)
#   N_WORKERS  parallel workers              (default: 10)
#   RAY_ARGS   extra run.py runtime flags for the RL commands, e.g. "--ray_gpus 1 --policy_gpus 1"
#   FORCE=1    overwrite an existing belief dataset / belief model instead of stopping
#   CODE_DIR   the training code (run.py, confs/, evals/, plotting/, data/, logs/)  (default: <tells-results>/tells-train)
#   RESULTS_DIR  where every env's evaluation results live, results/<env>/        (default: <tells-results>/results)
#   MODELS_DIR   trained belief models, belief/_<run>_NN_<i>/epoch<N>/<ckpt>     (default: <tells-results>/models)
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # tells-results
CODE_DIR=$(realpath -m "${CODE_DIR:-$ROOT/tells-train}")
RESULTS_DIR=$(realpath -m "${RESULTS_DIR:-$ROOT/results}")
MODELS_DIR=$(realpath -m "${MODELS_DIR:-$ROOT/models}")
[ -f "$CODE_DIR/run.py" ] || { echo "ERROR: no run.py in $CODE_DIR (set CODE_DIR to the training code)" >&2; exit 1; }
cd "$CODE_DIR"
PY=${PY:-/home/cameron/miniforge3/envs/tells/bin/python}
RUNS=${RUNS:-1000}
N_WORKERS=${N_WORKERS:-10}
RAY_ARGS=${RAY_ARGS:-}

# env -> configs. ALLO_CFG is the config the allocentric (belief) eval runs with; PF_CFG is empty where the
# particle filter (egocentric) baseline is not implemented. RESULTS_ROOT is where the env's eval results live:
# $RESULTS_DIR/<env> (results_fully_obs/, results_allocentric/, results_pf/ and the run's config.yaml).
env_configs() {
    RESULTS_ROOT=
    case "$1" in
        predator_prey)       BASE_CFG=confs/predator_prey/3a_game.yaml
                             ALLO_CFG=confs/predator_prey/3a_game_allocentric.yaml
                             PF_CFG=confs/predator_prey/3a_game_pf.yaml ;;
        drones)              BASE_CFG=confs/drones/3a_game.yaml
                             ALLO_CFG=confs/drones/3a_game_allocentric.yaml
                             PF_CFG=confs/drones/3a_game_pf.yaml ;;
        football)            BASE_CFG=confs/football/5a_game.yaml
                             ALLO_CFG=confs/football/5a_game_allocentric.yaml
                             PF_CFG=confs/football/5a_game_pf.yaml ;;
        fire)                BASE_CFG=confs/fire/4a_game.yaml
                             ALLO_CFG=confs/fire/4a_game_allocentric.yaml
                             PF_CFG=confs/fire/4a_game_pf.yaml ;;
        market_game)         BASE_CFG=confs/market_game/7a_game.yaml
                             ALLO_CFG=confs/market_game/7a_game.yaml  # the belief wrapper turns the team obs off
                             PF_CFG=confs/market_game/7a_game_pf.yaml ;;
        market_game_large)   BASE_CFG=confs/market_game_large/60a_game.yaml
                             ALLO_CFG=confs/market_game_large/60a_game.yaml
                             PF_CFG= ;;
        predator_prey_swarm) BASE_CFG=confs/predator_prey_swarm/40a_game.yaml
                             ALLO_CFG=confs/predator_prey_swarm/40a_game_allocentric.yaml
                             PF_CFG= ;;
        *) echo "unknown env '$1' (predator_prey drones football fire market_game market_game_large predator_prey_swarm)" >&2
           return 1 ;;
    esac
    BELIEF_CFG=confs/belief/$1.yaml
    # config the belief dataset is collected with: the base config, except football, whose policy runs noise-free
    # but whose belief data is collected with observation noise 0.03 (5a_game_collect.yaml)
    COLLECT_CFG=$BASE_CFG
    [ "$1" = football ] && COLLECT_CFG=confs/football/5a_game_collect.yaml
    RESULTS_ROOT=$RESULTS_DIR/$1
}

log() { echo "[$(date '+%F %T')] $*"; }

# yaml_get <config> <dotted.key>
yaml_get() {
    "$PY" -c "import sys, yaml
v = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): v = v[k]
print(v)" "$1" "$2"
}

# policy logdir from the env config (training writes <logdir>/checkpoint<N>, checkpoint_latest and final)
policy_logdir() { yaml_get "$1" logdir; }

# data/<data_name>: where marl_collect_data writes the belief dataset and where belief_train reads it
belief_data_dir() { echo "$(yaml_get "$1" data_dir)$(yaml_get "$1" data_name)"; }

# policy run name: the last part of the base config's logdir (e.g. market_game_fully_obs); names the belief models
run_name() { basename "$(policy_logdir "$BASE_CFG")"; }

# belief models of the current env (env_configs first): $MODELS_DIR/belief/_<run>_NN_<i>/epoch<N>/<ckpt>, one per
# training, numbered from 1
belief_model_root() { echo "$MODELS_DIR/belief/_$(run_name)_NN"; }
next_belief_index() {
    local i=1 root; root=$(belief_model_root)
    while [ -e "${root}_$i" ]; do i=$((i + 1)); done
    echo $i
}

# checkpoint dir of the newest belief model (highest index) - the default for allocentric evaluations
belief_ckpt_dir() {
    local root latest; root=$(belief_model_root)
    latest=$(ls -d "${root}"_[0-9]* 2>/dev/null | sort -V | tail -1)
    [ -n "$latest" ] || { echo "ERROR: no belief model under ${root}_<i> (train one, or pass the checkpoint dir)" >&2; return 1; }
    ls -d "$latest"/epoch* | tail -1
}

# where learn/belief/train.py puts a training's checkpoints (its log_dir), for a given belief config
belief_train_dir() {
    "$PY" -c "import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print('_'.join([c['log_dir'], c['data_name'], c['model_name'], str(c['label'])]) + '/lightning_logs/checkpoints/{epoch}_{val_loss}')" "$1"
}

# copy a training's best checkpoint to $MODELS_DIR/belief/_<run>_NN_<i>/epoch<N>/
publish_belief() {  # <training checkpoint dir> <index>
    local ck ep dest
    ck=$(ls -t "$1"/*.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] || { echo "ERROR: no checkpoint in $1" >&2; return 1; }
    ep=$(basename "$ck" | sed -E 's/^epoch=([0-9]+)-.*/\1/')
    dest=$(belief_model_root)_$2/epoch$ep
    mkdir -p "$dest" && cp "$ck" "$dest/"
    echo "$dest"
}

# stop if a dir already has content, unless FORCE=1
guard_nonempty() {
    if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "ERROR: $1 already exists and is not empty ($2). Move it away or rerun with FORCE=1." >&2
        exit 1
    fi
}

# train_pipeline <env>: decentralized policy -> belief dataset -> belief model
#   STEPS     subset of "policy data belief" to run (default: all three)
#   MODEL_DIR policy checkpoint to collect the dataset with (default: <logdir>/final from the training step)
train_pipeline() {
    local env=$1
    env_configs "$env"
    local steps=${STEPS:-policy data belief}
    local logdir; logdir=$(policy_logdir "$BASE_CFG")
    local model_dir=${MODEL_DIR:-$logdir/final}
    local data_dir; data_dir=$(belief_data_dir "$BELIEF_CFG")
    local log_dir=logs/scripts/$env
    mkdir -p "$log_dir"

    if [[ " $steps " == *" policy "* ]]; then
        # marl_train resumes from the newest checkpoint in the logdir, so rerunning continues an interrupted run
        log "$env: training policy ($BASE_CFG -> $logdir), log $log_dir/policy.log"
        "$PY" -u run.py --command marl_train --config "$BASE_CFG" $RAY_ARGS > "$log_dir/policy.log" 2>&1
        log "$env: policy done"
    fi

    if [[ " $steps " == *" data "* ]]; then
        [ -d "$model_dir" ] || { echo "ERROR: no policy checkpoint at $model_dir (set MODEL_DIR)" >&2; exit 1; }
        guard_nonempty "$data_dir" "belief dataset"
        log "$env: collecting belief data ($model_dir -> $data_dir), log $log_dir/data.log"
        "$PY" -u run.py --command marl_collect_data --config "$COLLECT_CFG" --model_dir "$model_dir" \
            --save_dir "$(yaml_get "$BELIEF_CFG" data_name)" --runs "$RUNS" --n_workers "$N_WORKERS" \
            > "$log_dir/data.log" 2>&1
        log "$env: data done"
    fi

    if [[ " $steps " == *" belief "* ]]; then
        # each training gets the next index i: it trains from a copy of the belief config labelled i (its own
        # training folder in logs/, so nothing resumes or overwrites), then its best checkpoint is copied to
        # $MODELS_DIR/belief/_<run>_NN_<i>/epoch<N>/
        local i cfg train_dir dest
        i=$(next_belief_index)
        cfg=$log_dir/belief_$i.yaml
        "$PY" -c "import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c['label'] = 'run' + sys.argv[3]
yaml.safe_dump(c, open(sys.argv[2], 'w'), sort_keys=False)" "$BELIEF_CFG" "$cfg" "$i"
        train_dir=$(belief_train_dir "$cfg")
        guard_nonempty "$train_dir" "belief training folder"
        log "$env: training belief model $i ($BELIEF_CFG), log $log_dir/belief_$i.log"
        "$PY" -u run.py --command belief_train --config "$cfg" > "$log_dir/belief_$i.log" 2>&1
        dest=$(publish_belief "$train_dir" "$i") || exit 1
        log "$env: belief model $i -> $dest"
    fi
    log "$env: pipeline finished"
}
