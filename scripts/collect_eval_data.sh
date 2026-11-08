#!/bin/bash
# evaluation data for one environment, all with the same policy checkpoint:
#   fully observable  marl_eval         base config          -> <model_dir>/results
#   egocentric (PF)   ego_eval          *_pf config          -> <model_dir>/results_pf
#   allocentric       marl_eval_belief  allocentric config   -> <model_dir>/results_allocentric
# each run is then moved to $RESULTS_DIR/<env>/results_fully_obs, results_pf, results_allocentric (replacing the
# previous run) and the training run's config.yaml is copied next to them (RESULTS_DIR: see common.sh).
# these results dirs are what plotting/, plotting/evaluate/ and evals/marl/cca_analysis.py read.
#
# usage: scripts/collect_eval_data.sh <env> <model_dir> [belief_ckpt_dir]
#   env              predator_prey drones football fire market_game market_game_large predator_prey_swarm
#   model_dir        policy checkpoint, e.g. logs/marl/football_fully_obs/checkpoint3000
#   belief_ckpt_dir  belief checkpoint dir (default: the one confs/belief/<env>.yaml trains to)
#
# environment variables: MODES (subset of "fully_obs egocentric allocentric", default all), PARALLEL=1 to run
# the modes at the same time, plus RUNS / N_WORKERS / RAY_ARGS / PY from common.sh. Existing results dirs are
# overwritten by the eval.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || { sed -n '2,15p' "$0"; exit 1; }
ENV=$1
MODEL_DIR=$2
env_configs "$ENV"
BELIEF_DIR=${3:-$(belief_ckpt_dir "$BELIEF_CFG")}
MODES=${MODES:-fully_obs egocentric allocentric}
LOG_DIR=logs/scripts/$ENV
mkdir -p "$LOG_DIR"
[ -d "$MODEL_DIR" ] || { echo "ERROR: no policy checkpoint at $MODEL_DIR" >&2; exit 1; }

run_mode() {
    local mode=$1
    case "$mode" in
        fully_obs)   set -- --command marl_eval --config "$BASE_CFG" ;;
        egocentric)  if [ -z "$PF_CFG" ]; then log "$ENV: no egocentric (PF) baseline, skipping"; return 0; fi
                     set -- --command ego_eval --config "$PF_CFG" ;;
        allocentric) [ -n "$(ls "$BELIEF_DIR"/*.ckpt 2>/dev/null)" ] || {
                         echo "ERROR: no belief checkpoint in $BELIEF_DIR" >&2; return 1; }
                     set -- --command marl_eval_belief --config "$ALLO_CFG" \
                            --belief_config "$BELIEF_CFG" --belief_dir "$BELIEF_DIR" ;;
        *) echo "unknown mode '$mode'" >&2; return 1 ;;
    esac
    log "$ENV: $mode eval started, log $LOG_DIR/eval_$mode.log"
    if "$PY" -u run.py "$@" --model_dir "$MODEL_DIR" --runs "$RUNS" --n_workers "$N_WORKERS" \
            --no_eval_videos --collect_results $RAY_ARGS > "$LOG_DIR/eval_$mode.log" 2>&1; then
        log "$ENV: $mode done"
        [ -z "$RESULTS_ROOT" ] || move_results "$mode"
    else
        log "$ENV: $mode FAILED (see $LOG_DIR/eval_$mode.log)"; return 1
    fi
}

# move a finished run from the checkpoint into RESULTS_ROOT
move_results() {
    local src dst
    case "$1" in
        fully_obs)   src=results;             dst=results_fully_obs ;;
        egocentric)  src=results_pf;          dst=results_pf ;;
        allocentric) src=results_allocentric; dst=results_allocentric ;;
    esac
    mkdir -p "$RESULTS_ROOT"
    rm -rf "${RESULTS_ROOT:?}/$dst"
    mv "$MODEL_DIR/$src" "$RESULTS_ROOT/$dst"
    cp "$(dirname "$MODEL_DIR")/config.yaml" "$RESULTS_ROOT/config.yaml"
    log "$ENV: moved $MODEL_DIR/$src -> $RESULTS_ROOT/$dst"
}

status=0
if [ "${PARALLEL:-0}" = 1 ]; then
    pids=()
    for m in $MODES; do run_mode "$m" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || status=1; done
else
    for m in $MODES; do run_mode "$m" || status=1; done
fi
for r in "$MODEL_DIR"/results "$MODEL_DIR"/results_pf "$MODEL_DIR"/results_allocentric \
         ${RESULTS_ROOT:+"$RESULTS_ROOT"/results_fully_obs "$RESULTS_ROOT"/results_pf "$RESULTS_ROOT"/results_allocentric}; do
    [ -d "$r" ] && log "$r: $(ls "$r" | wc -l) entries"
done
exit $status
