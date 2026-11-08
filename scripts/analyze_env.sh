#!/bin/bash
# full analysis of one policy checkpoint: collect its evaluation data, then make every paper figure / metric
# for the env from it.
#
# usage: scripts/analyze_env.sh <env> <model_dir> [belief_ckpt_dir]
#   env              predator_prey drones football fire market_game market_game_large predator_prey_swarm
#   model_dir        policy checkpoint; its eval data lives in $RESULTS_DIR/<env>/results_{fully_obs,pf,allocentric}
#   belief_ckpt_dir  belief checkpoint dir for the allocentric eval (default: from confs/belief/<env>.yaml)
#
# steps (STEPS, default "collect cca belief success"):
#   collect  scripts/collect_eval_data.sh: fully observable, egocentric (PF) and allocentric eval data
#   cca      evals/marl/cca_analysis.py + plotting/plot_cca_analysis.py -> centroid_map.png, task_ridgeline.png
#   belief   plotting/combined_std.py -> combined_std.png (belief error / belief std / decomposed reward),
#            combined_std_belief.png
#   success  prints success rate and episode length per method (from the CCA summary), plus the
#            plotting/analyze_all_data.py report (alloc vs PF incl. belief metrics) where supported
# market_game_large and predator_prey_swarm have no PF baseline or CCA spec; for them the belief and success
# steps run their own figure scripts (plotting/{market_game_large,predator_prey_swarm}_figures.py).
#
# everything goes to OUT_DIR (default $RESULTS_DIR/<env>/analysis/<checkpoint name>), logs to OUT_DIR/logs.
# drop "collect" from STEPS to analyze data that is already there. CCA_DEVICE (default cuda) is the device
# for fire's policy encoder; CCA_ARGS adds cca_analysis.py flags (e.g. "--limit-episodes 50 --n-boot 5" for a
# quick check); RUNS / N_WORKERS / RAY_ARGS / PY as in common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || { sed -n '2,24p' "$0"; exit 1; }
ENV=$1
MODEL_DIR=${2%/}
BELIEF_ARG=${3:-}
env_configs "$ENV"
STEPS=${STEPS:-collect cca belief success}
OUT_DIR=${OUT_DIR:-$RESULTS_DIR/$ENV/analysis/$(basename "$MODEL_DIR")}
LOGS=$OUT_DIR/logs
mkdir -p "$LOGS"
[ -d "$MODEL_DIR" ] || { echo "ERROR: no policy checkpoint at $MODEL_DIR" >&2; exit 1; }
DATA_ROOT=${RESULTS_ROOT:-$MODEL_DIR}  # folder holding this env's results folders
has() { [[ " $STEPS " == *" $1 "* ]]; }

case "$ENV" in
    predator_prey|drones|football|fire|market_game) CORE=1 ;;
    *) CORE=0 ;;
esac
status=0
step() {  # step <name> <log> <cmd...>
    local name=$1 logf=$2; shift 2
    log "$ENV: $name started, log $logf"
    # always returns 0 (set -e would otherwise end the script on one failed step); STEP_OK says how it went
    if "$@" > "$logf" 2>&1; then log "$ENV: $name done"; STEP_OK=1
    else log "$ENV: $name FAILED (see $logf)"; status=1; STEP_OK=0; fi
}

if has collect; then
    # its own log lines go to the terminal, each eval mode logs to logs/scripts/<env>/
    "$ROOT/scripts/collect_eval_data.sh" "$ENV" "$MODEL_DIR" $BELIEF_ARG \
        || { log "$ENV: collection had failures, stopping before the analysis"; exit 1; }
fi

if [ $CORE = 1 ]; then
    if has cca; then
        step cca "$LOGS/cca.log" "$PY" -u evals/marl/cca_analysis.py --env "$ENV" --data-dir "$DATA_ROOT" --checkpoint "$MODEL_DIR" \
                --out-dir "$OUT_DIR/cca" --device "${CCA_DEVICE:-cuda}" ${CCA_ARGS:-}
        [ $STEP_OK = 1 ] && step cca_plots "$LOGS/cca_plots.log" "$PY" plotting/plot_cca_analysis.py --in-dir "$OUT_DIR/cca"
    fi
    if has belief; then
        step belief "$LOGS/belief.log" "$PY" plotting/combined_std.py --envs "$ENV" --results-root "$DATA_ROOT" \
            --out-dir "$OUT_DIR"
    fi
    if has success; then
        if [ -f "$OUT_DIR/cca/summary.json" ]; then
            log "$ENV: success rates (from the CCA summary)"
            "$PY" - "$OUT_DIR/cca/summary.json" <<'EOF'
import json, sys
A = json.load(open(sys.argv[1]))['actions']
print(f"{'method':<12}{'success rate':>14}{'mean ep. length':>18}")
for m, v in A.items():
    print(f"{m:<12}{v['success_rate']:>14.3f}{v['mean_episode_length']:>18.1f}")
EOF
        else
            log "$ENV: no $OUT_DIR/cca/summary.json, run the cca step for the per-method success table"
        fi
        if [ "$ENV" != market_game ]; then  # analyze_all_data has no market success rule
            log "$ENV: success / belief report (analyze_all_data.py)"
            "$PY" -u plotting/analyze_all_data.py --envs "$ENV" --results-root "$DATA_ROOT" \
                || { log "$ENV: success report FAILED"; status=1; }
        fi
    fi
else
    if has cca; then log "$ENV: no CCA spec for this env, skipping cca"; fi
    if has belief || has success; then
        if [ "$ENV" = market_game_large ]; then
            step figures "$LOGS/figures.log" "$PY" plotting/market_game_large_figures.py --results-root "$DATA_ROOT" \
                --out-dir "$OUT_DIR" --table "$OUT_DIR/results_table.tex"
        else
            step figures "$LOGS/figures.log" "$PY" plotting/predator_prey_swarm_figures.py \
                --results "$DATA_ROOT/results_allocentric" --out-dir "$OUT_DIR"
        fi
    fi
fi

log "$ENV: analysis finished, outputs in $OUT_DIR"
exit $status
