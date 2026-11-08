#!/bin/bash
# plot belief-noise sweeps from their summary .npz files alone (written by scripts/noise_sweep_run.sh); no rollouts or
# models are needed.
#
# usage: scripts/noise_sweep_plot.sh [env ...] [--no-combined]
#   env            envs to plot (default: every env with $RESULTS_DIR/<env>/noise_sweep/noise_sweep_results)
#                  -> $RESULTS_DIR/<env>/noise_sweep/noise_sweep.png  (weighted task R^2 vs success, per noise level)
#   --no-combined  skip the combined figure
#                  -> $RESULTS_DIR/noise_sweep_combined.png  (every env with summaries, logistic fit per env)
# RESULTS_DIR / CODE_DIR / PY: see common.sh. The plotting code is $CODE_DIR/plotting/plot_cca_analysis.py.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ENVS=(); COMBINED=1
for a in "$@"; do
    case "$a" in
        --no-combined) COMBINED=0 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) ENVS+=("$a") ;;
    esac
done
if [ ${#ENVS[@]} = 0 ]; then
    for d in "$RESULTS_DIR"/*/noise_sweep/noise_sweep_results; do
        [ -d "$d" ] && ENVS+=("$(basename "$(dirname "$(dirname "$d")")")")
    done
fi
PLOT=$CODE_DIR/plotting/plot_cca_analysis.py

status=0
for env in "${ENVS[@]}"; do
    d=$RESULTS_DIR/$env/noise_sweep/noise_sweep_results
    if [ -z "$(ls "$d"/*.npz 2>/dev/null)" ]; then
        log "$env: no summaries in $d, skipped"; status=1; continue
    fi
    "$PY" "$PLOT" --sweep "$d" > /dev/null || { log "$env: plot FAILED"; status=1; continue; }
    log "$env: $(ls "$d"/*.npz | wc -l) runs -> $(dirname "$d")/noise_sweep.png"
done

if [ $COMBINED = 1 ]; then
    # the combined figure reads <TELLS_RESULTS>/results/<env>/noise_sweep/noise_sweep_results (util.util.RESULTS_ROOT)
    [ "$(basename "$RESULTS_DIR")" = results ] || { log "combined figure needs RESULTS_DIR to be a folder named results"; exit 1; }
    TELLS_RESULTS="$(dirname "$RESULTS_DIR")" "$PY" "$PLOT" --sweep-combined "$RESULTS_DIR/noise_sweep_combined.png" \
        | grep -E "saved|skipping" | sed "s|^|[combined] |" || { log "combined figure FAILED"; status=1; }
fi
exit $status
