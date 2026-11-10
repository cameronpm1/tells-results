# tells-results

Trained models, evaluation results, videos and run scripts for the communications-denied multi-agent experiments. The training and analysis code is the separate [tells-train](https://github.com/cameronpm1/tells-train) repository, cloned into this folder.

## Layout

```
tells-results/
├── tells-train/   code: run.py, confs/, envs/, learn/, evals/, plotting/ (separate git repository)
├── scripts/       bash pipelines, described below
├── models/        marl/<run>/checkpoint<N>/ policies, belief/_<run>_NN_<i>/epoch<N>/ belief models
├── results/       <env>/results_{fully_obs,allocentric,pf}/ rollouts, cca/, noise_sweep/, controller_results/
└── videos/        supplementary grid GIFs and MP4s
```

Environments: `predator_prey`, `drones`, `football`, `fire`, `market_game`, `market_game_large`, `predator_prey_swarm`.

## Downloading Datasets

The evaluation rollouts for the main seed (`results/<env>/results_{fully_obs,allocentric,pf}/`) are available on [Zenodo](https://zenodo.org/records/24114589). After cloning this repository, run the following from its root to download them and add them to the existing `results/` folder. Existing files are never overwritten, so everything from the repository is kept. This needs about 57 GB of free disk space (a 4.7 GB download that unpacks to 52 GB).

```bash
curl -L -o results.zip "https://zenodo.org/records/24114589/files/results.zip?download=1"
echo "42ef356490efde6014e72d980016b05d  results.zip" | md5sum -c -
unzip -n results.zip -x "__MACOSX/*" "*.DS_Store"
rm results.zip
```

## Installation

Clone the code into this folder, then install from inside it:

```bash
git clone git@github.com:cameronpm1/tells-train.git tells-train

cd tells-train
```

The default sync installs the predator-prey/MARL stack. The pybullet drone
environment is required for the caravan env. We recommend using a conda environment as it is necessary for google-football. Google-football has not been updated in several years, and requires an outdated version of setup-tools, as a result installation can be tricky depending on your machine. Set up instructions acter cloning git-repository below:

```bash
git submodule update --init --recursive

conda env create -f environment.yml

conda activate tells

python -m pip install "pip<23" "setuptools==65.5.0" "wheel<0.40"

python -m pip install -r requirements.txt

pip install -e external/pybullet-drones
```

Check if google-football is if not download it from git manually:

```bash
cd external

git clone git@github.com:google-research/football.git

cd football

pip install -r requirements.txt

python -m pip install .
```

If google-football hits an error in install that says "likely not an error with pip," it could me an issue with you environment pointing to newer versions of boost and setup tools outside of your conda environment. In order to fix this, run the following command, you can copy and paste the entire prompt into your linux terminal. The rerun "python -m pip install .".

```bash
cp gfootball/build_game_engine.sh gfootball/build_game_engine.sh.bak

python - <<'PY'
from pathlib import Path

p = Path("gfootball/build_game_engine.sh")
s = p.read_text()

old = "cmake ."
new = (
    "cmake . "
    "-DPython_EXECUTABLE=$CONDA_PREFIX/bin/python "
    "-DPython_ROOT_DIR=$CONDA_PREFIX "
    "-DPython_FIND_STRATEGY=LOCATION "
    "-DPython_FIND_VIRTUALENV=FIRST "
    "-DBOOST_ROOT=$CONDA_PREFIX "
    "-DBoost_NO_SYSTEM_PATHS=ON "
    "-DCMAKE_PREFIX_PATH=$CONDA_PREFIX "
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
)

if old not in s:
    raise SystemExit("Could not find plain 'cmake .' in build_game_engine.sh")

p.write_text(s.replace(old, new))
print("patched")
PY

rm -rf build
rm -rf third_party/gfootball_engine/CMakeFiles
rm -f third_party/gfootball_engine/CMakeCache.txt
```

The scripts default to the author's interpreter path, so point them at your conda environment first:

```bash
conda activate tells

export PY=$(which python)
```

## Generating Evaluation Rollouts

The evaluation rollouts (`results/<env>/results_{fully_obs,allocentric,pf}/`, about 49 GB) are too large for this repository and must be generated before running any analysis. Collect them for each environment with the included models:

```bash
scripts/collect_eval_data.sh predator_prey ../models/marl/predator_prey_fully_obs/checkpoint1000
scripts/collect_eval_data.sh drones ../models/marl/drones_fully_obs/checkpoint1500
scripts/collect_eval_data.sh football ../models/marl/football_fully_obs/checkpoint3000
scripts/collect_eval_data.sh fire ../models/marl/fire_fully_obs/checkpoint5000
scripts/collect_eval_data.sh market_game ../models/marl/market_game_fully_obs/checkpoint829
scripts/collect_eval_data.sh market_game_large ../models/marl/market_game_large_fully_obs/checkpoint250
scripts/collect_eval_data.sh predator_prey_swarm ../models/marl/predator_prey_swarm_fully_obs/checkpoint250
```

Each command collects 1000 episodes per method (set `RUNS` to change this) and replaces any existing rollouts for that environment. The newest belief model is used unless you pass a checkpoint folder as a third argument, and `MODES="allocentric"` collects only some of `fully_obs`, `egocentric` and `allocentric`.

The noise-sweep summaries are included in `results/<env>/noise_sweep/`, so the sweep figures can be drawn without any rollouts.

## Running the Scripts

The bash scripts in `scripts/` provide access to all of the functionality in the `tells-train` repository. To customize model training, evaluation, or data collection further, see `tells-train/README.md`. The scripts change into `tells-train`, so checkpoint and config arguments are relative to `tells-train` (hence `../models/...`), or absolute.

| Script | What it does |
|---|---|
| `train_<env>.sh` | Trains a decentralized policy, collects a belief dataset with it, and trains a belief model. |
| `collect_eval_data.sh` | Collects fully observable, egocentric and allocentric rollouts for a policy into `results/<env>/`. |
| `analyze_env.sh` | Runs the CCA analysis, belief error/std figures and success metrics for one environment. |
| `noise_sweep_run.sh` | Runs the belief-noise sweep for a policy and keeps only the summary files. |
| `noise_sweep_plot.sh` | Draws the per-environment and combined noise-sweep figures from those summaries. |
| `common.sh` | Shared settings and helpers sourced by the other scripts; not run directly. |

Most scripts also read `RUNS` (episodes per collection or evaluation, default 1000), `N_WORKERS` (default 10) and `RAY_ARGS` (extra `run.py` flags). Each script's header lists all its options.

## Creating New Models

Train a policy, its belief dataset and its belief model from scratch:

```bash
scripts/train_football.sh
```

The policy is written to `tells-train/logs/marl/<run>/`, the dataset to `tells-train/data/`, and the belief model to `models/belief/_<run>_NN_<i>/epoch<N>/`, where `i` is the next free index.

Train only a new belief model for an existing policy:

```bash
STEPS="data belief" MODEL_DIR=../models/marl/football_fully_obs/checkpoint3000 scripts/train_football.sh
```

## Reproducibility Over Seeds

We trained two additional seeds for every experiment to show that the results are reproducible across random seeds. To run your own analysis on these results, follow the instructions in [Generating Evaluation Rollouts](#generating-evaluation-rollouts) to collect data from the models provided in `models/marl/<run>/additional_seeds/` and `models/belief/_<run>_NN_1/additional_seeds/`.

| Experiment | Metric | Main | Seed 1 | Seed 2 |
|---|---|---|---|---|
| `predator_prey` | Seed | 23 | 53548 | 48405 |
| | Average belief error | 0.990 | 0.923 | 0.955 |
| | Belief std | 0.371 | 0.342 | 0.356 |
| | CCA weighted R² | 0.930 | 0.943 | 0.936 |
| | Allocentric success | 92.0% | 92.4% | 94.8% |
| `drones` | Seed | 22 | 53548 | 31907 |
| | Average belief error | 0.609 | 0.635 | 0.630 |
| | Belief std | 0.195 | 0.197 | 0.204 |
| | CCA weighted R² | 0.937 | 0.948 | 0.952 |
| | Allocentric success | 74.7% | 66.1% | 63.9% |
| `football` | Seed | 22 | 53548 | 48405 |
| | Average belief error | 0.088 | 0.078 | 0.081 |
| | Belief std | 0.050 | 0.046 | 0.044 |
| | CCA weighted R² | 0.724 | 0.800 | 0.747 |
| | Allocentric success | 36.8% | 28.0% | 26.0% |
| `fire` | Seed | 22 | 23 | 25 |
| | Average belief error | 6.629 | 5.100 | 6.816 |
| | Belief std | 2.955 | 2.698 | 4.123 |
| | CCA weighted R² | 0.880 | 0.945 | 0.884 |
| | Allocentric success | 71.0% | 58.0% | 73.0% |
| `market_game` | Seed | 23 | 53548 | 48405 |
| | Average belief error | 3.490 | 3.553 | 3.289 |
| | Belief std | 1.878 | 1.633 | 1.269 |
| | CCA weighted R² | 0.870 | 0.877 | 0.880 |
| | Allocentric success | 97.5% | 95.5% | 98.2% |
| `market_game_large` | Seed | 23 | 53548 | 48405 |
| | Average belief error | 1.72 | 2.64 | 2.12 |
| | Belief std | 0.654 | 1.032 | 0.795 |
| | Allocentric success | 91.9% | 88.5% | 80.4% |
| `predator_prey_swarm` | Seed | 23 | 53548 | 48405 |
| | Average belief error | 0.685 | 0.768 | 0.722 |
| | Belief std | 0.219 | 0.225 | 0.215 |
| | Allocentric success | 97.3% | 91.7% | 94.8% |

## Generating Results and Plots

**From the rollouts** (see [Generating Evaluation Rollouts](#generating-evaluation-rollouts)). Outputs go to `results/<env>/analysis/<checkpoint>/`:

```bash
STEPS="cca belief success" scripts/analyze_env.sh football ../models/marl/football_fully_obs/checkpoint3000
```

Use `STEPS` to run only part of it, for example `STEPS="success"`. `market_game_large` and `predator_prey_swarm` have no CCA or egocentric baseline, so they produce their own figures instead. The football CCA figures in `results/football/cca` use the first 100 steps of each episode; add `CCA_ARGS="--max-steps 100"` to match them.

**Collect new rollouts**, which replaces `results/<env>/results_*`. The newest belief model is used unless you pass a checkpoint folder as a third argument:

```bash
scripts/collect_eval_data.sh football ../models/marl/football_fully_obs/checkpoint3000
```

To collect only one method, set `MODES` to `fully_obs`, `egocentric` or `allocentric`:

```bash
MODES="egocentric" scripts/collect_eval_data.sh football ../models/marl/football_fully_obs/checkpoint3000
```

**Plot the noise sweeps** from the included summaries (no models or rollouts needed):

```bash
scripts/noise_sweep_plot.sh
```

**Rerun a noise sweep** (expensive, and needs the rollouts above for the clean and egocentric baselines). `DATA_RUNS` is the number of clean episodes to collect, and `FORCE=1` replaces existing summaries:

```bash
FORCE=1 DATA_RUNS=1000 scripts/noise_sweep_run.sh ../models/marl/football_fully_obs/checkpoint3000 confs/football/5a_game.yaml
```

Sweeps are available for `predator_prey`, `drones`, `fire`, `market_game` and `football`.

**Render a supplementary video** (run from `tells-train`). Each environment has a script in `plotting/video_gen/`:

```bash
cd tells-train

python plotting/video_gen/football_grid_gif.py --save_path ../videos/football_grid.gif
```

Not all environment physics engines are deterministic, so the same seed may not always produce identical results.
