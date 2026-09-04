# CoGameReg

Official implementation of **CoGameReg**, a cooperative game-guided
semi-supervised regression method. This repository provides the core algorithm
and the benchmark entry point used to reproduce the CoGameReg results reported
in the associated manuscript.

## Scope

This public repository contains:

- the CoGameReg implementation;
- the model-pool configuration used in the benchmark experiments;
- the preprocessing and evaluation protocol for ten public regression
  datasets;
- reference RMSE results and lightweight verification tests.

Independent implementations of the comparison baselines and the complete
ablation suite are not included in this public repository.

## Repository structure

```text
.
├── cogamereg.py                  # Core CoGameReg implementation
├── run_benchmark.py              # Ten-dataset benchmark entry point
├── requirements.txt              # Compatible direct dependencies
├── requirements-lock.txt         # Versions used for local verification
├── data/
│   ├── README.md                 # Data provenance and preparation
│   └── SHA256SUMS                # Expected dataset checksums
├── results/
│   └── reference_results.md      # Reference manuscript results
├── scripts/
│   └── verify_data.py            # Cross-platform checksum verification
├── tests/
│   └── test_model_initialization.py
└── .github/workflows/smoke.yml   # Dependency and model-initialization CI
```

The benchmark data files are deliberately not redistributed here. See
[`data/README.md`](data/README.md) for their source and preparation procedure.

## Environment

The reference local environment uses Python 3.11. Create an isolated
environment and install the locked dependencies:

```bash
conda create -n cogamereg python=3.11 -y
conda activate cogamereg
python -m pip install -r requirements-lock.txt
```

For a less restrictive installation that accepts compatible package updates,
use:

```bash
python -m pip install -r requirements.txt
```

Exact package versions are recommended when comparing against the reported
numbers.

## Prepare the benchmark data

Download the data from the official S2RMS repository and copy the following ten
files from its `data/` directory into this repository's `data/` directory:

```text
Folds5x2_pp.arff
abalone.arff
bank32nh.arff
elevators.arff
kin8nm.arff
parkinsons.arff
puma8NH.arff
space_ga.arff
wind.arff
wine_quality.arff
```

Then verify that the files match the benchmark snapshot:

```bash
python scripts/verify_data.py
```

The source repository and detailed provenance are documented in
[`data/README.md`](data/README.md).

## Run

Run all ten datasets at label ratios 0.025, 0.05, and 0.10 with ten seeds
(42--51):

```bash
python run_benchmark.py
```

Run a short one-seed check on Abalone:

```bash
python -c "from run_benchmark import run_calibration_experiment; run_calibration_experiment('data/abalone.arff', label_ratio=0.025, runs=1)"
```

In the locked reference environment, this check produces an RMSE that rounds to
`0.0898` for seed 42.

Run the model-initialization tests:

```bash
python -m unittest discover -s tests -v
```

## Benchmark protocol

For each dataset and seed, the data are shuffled once. Up to 2,000 samples form
the training pool and all remaining samples form the test set. The requested
fraction of the training pool is treated as labeled; 20% of those labeled
samples form the validation set. Feature standardization is fitted using the
remaining labeled training samples. Results are reported as the mean RMSE and
population standard deviation over seeds 42--51.

The released model pool contains XGBoost, RBF-SVR, MLP, ridge regression,
ordinary least squares, and KNN. CoGameReg uses fixed initial Tri-Game Shapley
weights during collaborative training (`ablation_mode='undecouple_shap'`).

See [`results/reference_results.md`](results/reference_results.md) for the
reference result matrix.

## Reproducibility note

The two Python implementation files are the cleaned public version of the code
used for the reported experiments. The model configuration, seeds, split
procedure, candidate selection, early-stopping rule, retraining procedure, and
metric calculation are unchanged.

Small floating-point differences can occur across operating systems, numerical
libraries, thread schedules, and XGBoost builds. Differences on the order of
`1e-4` to `2e-4` in four-decimal summary values have been observed across
otherwise matching environments. The manuscript retains the results produced
in the reference server environment.

Reference source hashes at repository preparation time:

```text
daac43ba37902c355a9d11074a2c23b614951e846e202f513c744b71db1cb632  cogamereg.py
e1882a1b848e8dfa6983dc5dfc0d192e94292ba395a34eb45d2325c1f221d100  run_benchmark.py
```

## Citation

The formal paper citation and archival DOI will be added when the manuscript or
preprint record becomes publicly available.

## License and third-party material

A software license has not yet been selected. Until a `LICENSE` file is added,
the source is publicly visible but no open-source license is granted. Dataset
files are not distributed by this repository and will not be covered by any
future software license. See [`data/README.md`](data/README.md).
