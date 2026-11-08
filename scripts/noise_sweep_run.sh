#!/bin/bash
# belief-noise sweep experiments for one decentralized policy: for every noise level, train a belief model on noisy
# data, evaluate the policy with it under the same noise, and summarise every level with the CCA sweep. Only the
# summary .npz files (and the log) are kept: the trained belief models, datasets and rollouts are deleted. Plot the
# summaries with scripts/noise_sweep_plot.sh.
#
# usage: scripts/noise_sweep_run.sh <model_dir> <env_config> [--sigmas "s1 s2 ..."] [--slots N] [--dry-run]
#   model_dir    policy checkpoint, e.g. logs/marl/market_game_fully_obs/checkpoint829
#   env_config   its base env config, e.g. confs/market_game/7a_game.yaml; the env (confs/<env>/) picks the noise:
#       predator_prey  Gaussian sigma on the belief input and the observed prey
#       drones         Gaussian sigma on x and y only: belief input and the observed intruder
#       fire           each fire-window cell flipped to a random state with probability sigma (belief history + image)
#       market_game    Gaussian sigma on the belief input and the observed release, 0.25 x sigma on observed demand
#       football       Gaussian sigma on the belief input, the observed opponent and the ball position the policy receives;
#                      clean belief data collected with observation noise 0.03 (5a_game_collect.yaml)
#   --sigmas     noise levels (default: the env's paper levels)
#   --slots      noise levels trained / evaluated at once (default 4)
#   --dry-run    print the plan and stop
#
# steps (each leaves a marker in the work folder, so rerunning resumes):
#   1. collect the clean belief dataset with the policy          -> $CODE_DIR/data/noise_sweep_<env>_clean
#   2. per sigma: noisy dataset + configs (evals/marl/noise_sweep_build.py) -> train the belief model -> evaluate
#      EVAL_RUNS episodes into the work folder, then delete that level's dataset, configs and belief model
#   3. CCA sweep over all levels (sigma = 0 and egocentric are the env's results_allocentric / results_pf)
#      -> OUT_DIR/noise_sweep_results/<run>.npz: weighted task R^2, success, per-episode success / length, ...
#   4. delete the clean dataset and the work folder (all rollouts and models); the log is kept as OUT_DIR/sweep.log
#
# environment variables: OUT_DIR (default $RESULTS_DIR/<env>/noise_sweep), WORK (default $CODE_DIR/logs/noise_sweep/<env>),
#   DATA_RUNS (clean dataset episodes, default: as many as the paper dataset), EVAL_RUNS (default 1000),
#   COLLECT_WORKERS (default 4), CCA_ARGS (extra cca_analysis.py flags), EPOCHS (override belief epochs, for testing),
#   KEEP_WORK=1 (skip step 4), FORCE=1 (replace existing summaries in OUT_DIR), plus PY / N_WORKERS from common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() { sed -n '2,31p' "$0"; exit 1; }
[ $# -ge 2 ] || usage
MODEL_DIR=${1%/}; ENV_CFG=$2; shift 2
SIGMAS=; SLOTS=4; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sigmas) SIGMAS=$2; shift 2 ;;
        --slots) SLOTS=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) usage ;;
    esac
done
[ -d "$MODEL_DIR" ] || { echo "ERROR: no policy checkpoint at $MODEL_DIR" >&2; exit 1; }
[ -f "$ENV_CFG" ] || { echo "ERROR: no config at $ENV_CFG" >&2; exit 1; }
ENV=$(basename "$(dirname "$ENV_CFG")")

# env -> sweep settings (the noise each env's final paper sweep used)
case "$ENV" in
    predator_prey) SWEEP_ENV=predator_prey; DEF_SIGMAS="0.25 0.375 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2"
        NOISE="Gaussian std sigma on the belief input (training + evaluation) and on the observed prey" ;;
    drones) SWEEP_ENV=drones_xy; DEF_SIGMAS="0.15 0.25 0.375 0.5 0.6 0.7 0.85 1 1.2 1.4 1.6"
        NOISE="Gaussian std sigma on x and y only: belief input (training + evaluation) and the observed intruder" ;;
    fire) SWEEP_ENV=fire; DEF_SIGMAS="0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.07 0.1"
        NOISE="each fire-window cell replaced by a random fire state with probability sigma, in the belief history (training + evaluation) and the policy image" ;;
    market_game) SWEEP_ENV=market_game; DEF_SIGMAS="0.075 0.15 0.225 0.3 0.375 0.45 0.525 0.6 0.7 0.8 0.9 1 1.2"
        NOISE="Gaussian std sigma on the belief input (training + evaluation) and the observed release; 0.25 x sigma on the observed demand" ;;
    football) SWEEP_ENV=football; DEF_SIGMAS="0.02 0.03 0.04 0.05 0.06 0.075 0.09 0.1 0.125 0.15 0.2"
        NOISE="belief data collected with observation noise 0.03; Gaussian std sigma on the belief input (training + evaluation), the observed opponent and the ball position the policy receives" ;;
    *) echo "ERROR: no noise sweep for env '$ENV' (predator_prey, drones, fire, market_game, football)" >&2; exit 1 ;;
esac
env_configs "$ENV"
[ "$(realpath "$ENV_CFG")" = "$(realpath "$BASE_CFG")" ] && DATA_CFG=$COLLECT_CFG || DATA_CFG=$ENV_CFG
SIGMAS=${SIGMAS:-$DEF_SIGMAS}
OUT_DIR=${OUT_DIR:-$RESULTS_DIR/$ENV/noise_sweep}
WORK=${WORK:-logs/noise_sweep/$ENV}
SOURCE=noise_sweep_${ENV}_clean
PAPER_DATA=$(belief_data_dir "$BELIEF_CFG")
DATA_RUNS=${DATA_RUNS:-$(ls "$PAPER_DATA" 2>/dev/null | wc -l || true)}
[ "$DATA_RUNS" -gt 0 ] || { echo "ERROR: set DATA_RUNS (no paper dataset at $PAPER_DATA to size it by)" >&2; exit 1; }
EVAL_RUNS=${EVAL_RUNS:-1000}
COLLECT_WORKERS=${COLLECT_WORKERS:-4}  # each collection worker starts its own Ray instance (~40 processes)
DATA_ROOT=${RESULTS_ROOT:-$MODEL_DIR}  # holds the policy's fully observed / allocentric / pf rollouts
SIGMA0=$DATA_ROOT/results_allocentric
EGO=$DATA_ROOT/results_pf

echo "noise sweep: $ENV (sweep settings '$SWEEP_ENV')"
echo "  policy      $MODEL_DIR   config $ENV_CFG"
echo "  noise       $NOISE"
echo "  sigmas      $SIGMAS   ($SLOTS at a time)"
echo "  clean data  data/$SOURCE ($DATA_RUNS episodes, collected with $DATA_CFG)   eval $EVAL_RUNS episodes per level"
echo "  sigma = 0   $SIGMA0   egocentric $EGO"
echo "  output      $OUT_DIR   work $WORK"
for d in "$SIGMA0" "$EGO"; do
    [ "$(ls "$d" 2>/dev/null | wc -l)" -gt 0 ] || { echo "ERROR: missing $d (collect it with scripts/collect_eval_data.sh)" >&2; exit 1; }
done
if [ -n "$(ls "$OUT_DIR/noise_sweep_results" 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "ERROR: $OUT_DIR/noise_sweep_results already has summaries; use another OUT_DIR or FORCE=1 to replace them" >&2
    exit 1
fi
[ $DRY = 1 ] && exit 0

mkdir -p "$WORK/rollouts" "$OUT_DIR"
LOG=$WORK/sweep.log
slog() { log "$*" | tee -a "$LOG"; }
NICE="nice -n 15"

# 1. clean belief dataset
if [ ! -f "$WORK/collect.done" ]; then
    slog "collecting $DATA_RUNS clean episodes -> data/$SOURCE"
    rm -rf "data/${SOURCE:?}"
    $NICE "$PY" -u run.py --command marl_collect_data --config "$DATA_CFG" --model_dir "$MODEL_DIR" --save_dir "$SOURCE" \
        --runs "$DATA_RUNS" --n_workers "$COLLECT_WORKERS" > "$WORK/collect.log" 2>&1 \
        || { slog "collection FAILED (see $WORK/collect.log)"; exit 1; }
    touch "$WORK/collect.done"
fi

# 2. one noise level: build -> train -> evaluate -> keep the rollouts, delete the rest
level() {
    local s=$1 gpu=$2 L=$WORK/level_$1
    mkdir -p "$L"
    if [ ! -f "$L/build.env" ]; then
        $NICE "$PY" -u evals/marl/noise_sweep_build.py --env "$SWEEP_ENV" --sigma "$s" --source "$SOURCE" --work "$L" \
            ${EPOCHS:+--epochs $EPOCHS} > "$L/build.log" 2>&1 \
            && grep -E '^[A-Z_]+=' "$L/build.log" > "$L/build.env" \
            || { slog "$s: build FAILED (see $L/build.log)"; rm -f "$L/build.env"; return 1; }
    fi
    source "$L/build.env"
    if [ ! -f "$L/train.done" ]; then
        slog "$s: training belief model (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu $NICE "$PY" -u run.py --command belief_train --config "$BELIEF_CFG" > "$L/train.log" 2>&1 || true
        ls "$BELIEF_DIR"/*.ckpt > /dev/null 2>&1 || { slog "$s: training FAILED (see $L/train.log)"; return 1; }
        touch "$L/train.done"
    fi
    if [ ! -f "$WORK/eval_$s.done" ]; then
        slog "$s: evaluating $EVAL_RUNS episodes"
        local res=$MODEL_DIR/results_allocentric$RESULTS_SUFFIX
        CUDA_VISIBLE_DEVICES=$gpu $NICE "$PY" -u run.py --command marl_eval_belief --config "$EVAL_CFG" \
            --model_dir "$MODEL_DIR" --belief_config "$BELIEF_CFG" --belief_dir "$BELIEF_DIR" --runs "$EVAL_RUNS" \
            --n_workers 10 --torch_threads 1 --ray_cpus 12 --no_eval_videos --collect_results > "$L/eval.log" 2>&1 || true
        [ "$(ls "$res" 2>/dev/null | wc -l)" = "$EVAL_RUNS" ] || { slog "$s: evaluation FAILED (see $L/eval.log)"; return 1; }
        mv "$res" "$WORK/rollouts/allocentric_sigma_$s"
        touch "$WORK/eval_$s.done"
        slog "$s: evaluated ($(grep -m1 'Success rate' "$L/eval.log"))"
    fi
    # this level's dataset, configs and belief model are no longer needed
    rm -rf "data/${DATA_NAME:?}" "data/datainfo/${DATA_NAME:?}" "${L:?}/belief"
}

slog "start: sigmas $SIGMAS"
i=0
for s in $SIGMAS; do
    [ -f "$WORK/eval_$s.done" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n || true; done
    level "$s" $((i % 2)) &
    i=$((i + 1)); sleep 2
done
wait
for s in $SIGMAS; do
    [ -f "$WORK/eval_$s.done" ] || { slog "not every level finished; rerun to resume (summary skipped)"; exit 1; }
done

# 3. CCA sweep -> per-level summary .npz
slog "CCA sweep over $(echo $SIGMAS | wc -w) levels"
runs=("allocentric:0=$SIGMA0" "egocentric=$EGO")
for s in $SIGMAS; do runs+=("allocentric:$s=$WORK/rollouts/allocentric_sigma_$s"); done
rm -rf "${WORK:?}/npz"
$NICE "$PY" -u evals/marl/cca_analysis.py --env "$ENV" --data-dir "$DATA_ROOT" --checkpoint "$MODEL_DIR" --out-dir "$WORK/cca" \
    --npz-dir "$WORK/npz" --noise "$NOISE" --workers 16 ${CCA_ARGS:-} --sweep "${runs[@]}" > "$WORK/cca.log" 2>&1 \
    || { slog "CCA sweep FAILED (see $WORK/cca.log)"; exit 1; }
# the noisy-level rollouts are deleted below: record that instead of their (work folder) paths
"$PY" -c "import glob, sys, numpy as np
for f in glob.glob(sys.argv[1] + '/allocentric_sigma_*.npz'):
    z = dict(np.load(f, allow_pickle=True))
    if float(z['noise_level']) > 0:
        z['results_dir'] = np.array('not kept (rollouts deleted after the sweep)'); np.savez(f, **z)" "$WORK/npz"
rm -rf "${OUT_DIR:?}/noise_sweep_results"
mv "$WORK/npz" "$OUT_DIR/noise_sweep_results"
slog "summary -> $OUT_DIR/noise_sweep_results (plot with scripts/noise_sweep_plot.sh $ENV)"

# 4. clean up: only the summary stays (rollouts, belief models and datasets are deleted)
if [ "${KEEP_WORK:-0}" != 1 ]; then
    rm -rf "data/${SOURCE:?}" "data/datainfo/${SOURCE:?}"
    cp "$LOG" "$OUT_DIR/sweep.log"
    rm -rf "${WORK:?}"
    log "removed data/$SOURCE and $WORK (log kept as $OUT_DIR/sweep.log)"
fi
log "$ENV noise sweep finished"
