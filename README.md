# LPSforECNN

Lifelong learning experiments for RF signal classification, with deterministic, Bayesian, and evidential (DST) model variants.

**Repo Structure**
- `main.py`: Training entrypoint. Handles task-by-task training, ADMM pruning, and masked retraining.
- `models/`: Model implementations.
- `TrainValTest.py`: Data loaders and evaluation utilities.
- `run_experiments.sh`: Example multi-task and single-task runs.
- `run_single_experiment.sh`: One-off experiment example.
- `radar/`, `dronerc/`, `rfmls/`, `usrp/`, `mixed/`: Dataset roots, task folders, and experiment outputs.
- `logs/`: Training curves and metrics saved per experiment.

**Data And Tasks Layout**

Datasets are hosted on Zenodo and should be downloaded locally preserving the
folder structure described below.

Each experiment reads data from:
`<base-path>/<task-folder>/task{N}/...`

Examples:
- `radar/tasks-no-noise/task0/radar_dataset.npz`
- `dronerc/tasks-1024sl/task1/train.npz`

Expected file formats:
- Radar tasks: `radar_dataset.npz` with `xtr`, `ytr`, `xte`, `yte` arrays.
- DroneRC / RFMLS / USRP / Mixed tasks: `train.npz` and `test.npz` with `X` and `y` arrays.

**Configuration**

`main.py` supports YAML defaults. If `--config` is not provided, it defaults to `dronerc/args.yaml`.

Common flags:
- `--arch`: `rfnet` (deterministic), `bayes_rfnet` (Bayesian), `evidential` (DST).
- `--base-path`: Dataset root (e.g., `radar/`).
- `--task-folder`: Task folder under the dataset root (e.g., `tasks-no-noise`).
- `--tasks`: Number of tasks (e.g., `2` for `task0` and `task1`).
- `--classes`: Total number of classes across all tasks.

**Run Experiments**

All examples below assume you are in the repo root.

Lifelong Learning (multi-task)
```bash
python main.py --exp-name "radar-lps" \
  --arch "rfnet" \
  --base-path "radar/" \
  --task-folder "tasks-no-noise" \
  --classes 10 \
  --tasks 2 \
  --adaptive-ratio 0.9 \
  --epochs 60 \
  --epochs-prune 10 --rho-num 4 \
  --epochs-mask-retrain 10 \
  --adaptive-mask true \
  --config-shrink 0.5 \
  --multi-head true
```

Bayesian model (multi-task)
```bash
python main.py --exp-name "radar-blc" \
  --arch "bayes_rfnet" \
  --base-path "radar/" \
  --task-folder "tasks-no-noise" \
  --classes 10 \
  --tasks 2 \
  --adaptive-ratio 0.9 \
  --epochs 60 \
  --epochs-prune 10 --rho-num 4 \
  --epochs-mask-retrain 10 \
  --adaptive-mask true \
  --config-shrink 0.5 \
  --multi-head true
```

Evidential (DST) model (multi-task)
```bash
python main.py --exp-name "radar-elc" \
  --arch "evidential" \
  --base-path "radar/" \
  --task-folder "tasks-no-noise" \
  --classes 10 \
  --tasks 2 \
  --epochs 60 \
  --epochs-prune 10 --rho-num 4 \
  --epochs-mask-retrain 10 \
  --adaptive-mask true \
  --config-shrink 0.5 \
  --multi-head true
```

**Outputs**
- Model checkpoints are saved under `<base-path>/<exp-name>/task{N}/`.
- Metrics are saved under `logs/<base-path>/<exp-name>/` as `.npz` files.

**Scripts**

Plotting scripts expect pickle payloads produced by the evaluation helpers in `scripts/` or top-level utilities such as `conf_matrix.py` and `evaluate_snr.py`. These scripts have hardcoded paths near the bottom; update them to your experiment outputs before running.

Selective Recall vs SNR
- Generator: `scripts/conf_matrix_elc.py` (ELC) or `conf_matrix.py` (BLC) produces `*_snr_results_*.pkl` files under `eu_comparison/...`.
- Plot: `plot_selective_acc_snr.py` reads those pickles and draws base vs selective accuracy across SNR.

Example:
```bash
python scripts/conf_matrix_elc.py
python plot_selective_acc_snr.py
```

Coverage plots (accuracy–coverage)
- Generator: `scripts/mod_conf_matrix.py` (or `conf_matrix.py`) produces pickles with `results` / `per_snr`.
- Plot: `plot_coverage.py` expects those pickles and plots coverage vs selective recall.

Example:
```bash
python scripts/mod_conf_matrix.py
python plot_coverage.py
```

Uncertainty plots
- Generator: `scripts/mod_conf_matrix.py` or `conf_matrix.py` can save `*_snr_results_distrib.pkl` (uncertainty distributions).
- Plot: `plot_uncertainty_distr.py` generates density and ROC plots for uncertainty vs correctness.

Example:
```bash
python scripts/mod_conf_matrix.py
python plot_uncertainty_distr.py
```

Helper files
- `scripts/helper.py` and `scripts/helper_elc.py` are dataset registries used by `scripts/conf_matrix_elc.py` and related evaluation utilities. They centralize paths to trained models, task folders, class offsets, and SNR split locations.
- Update these when you run a new experiment or move data:
  - `dataset`: pick a key from `datasets` to select which config is active.
  - `datasets[...]["models"]`: path to the experiment output directory containing `task*/cumu_model.pt`.
  - `datasets[...]["tasks"]`: path to the task folder (used for loading train/test data).
  - `datasets[...]["offset"]` or `["cpt"]`: class offsets or classes-per-task splits.
  - `task_configs`: list of tasks and checkpoint filenames to evaluate.
  - `dataset_paths`: absolute paths to SNR split folders (e.g., `.../snr_splits_*`).
  - `snr_range`: SNR values expected in those folders.
