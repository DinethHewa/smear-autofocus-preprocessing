# CFM4 Preprocessing Optimization for Smear Microscopy Autofocus

This repository is a reproducible research-software compendium for optimizing
preprocessing workflows used before autofocus scoring in smear microscopy.
It supports two linked studies:

1. **Generalized preprocessing:** search for one workflow that transfers across
   smear datasets using leave-one-dataset-out (LODO) evaluation.
2. **Dataset-specific preprocessing:** learn one workflow per dataset, evaluate
   cross-dataset transfer, and quantify improvement, overfitting, runtime, and
   complexity relative to the generalized workflow.

The production run covers BMA, PBS, TBF, TBI, and WBC. Raw microscopy images and
source stack arrays are not redistributed in this repository. Small synthetic
stacks are included for smoke testing.

## Method

The focus measure is:

```text
CFM4 = FTSI - WDE-db1 * psqrt(psqrt(BG - GSE))
psqrt(x) = sqrt(abs(x) + epsilon)
```

where BG is Brenner Gradient, GSE is Gradient Squared Energy, FTSI is the
Fourier Transform Sharpness Index, and WDE-db1 is db1 wavelet detail energy.

For every stack/workflow pair, the pipeline stores the raw focus curve,
per-stack min-max normalized curve, predicted focus index, reference focus
index, label provenance, runtime, and failure information. Evaluation uses ten
autofocus metrics:

1. absolute peak localization error;
2. full width at half maximum;
3. curvature at peak;
4. steep-slope width;
5. steep-to-gradual slope ratio;
6. false maxima count;
7. noise level;
8. relative RMSE under additive noise;
9. high-response range around the global maximum; and
10. execution time per slice.

The primary endpoint is an equal-dataset, rank-based generalization score over
all ten metrics. Secondary analyses include value-based aggregation with
`alpha = 0.7`, bootstrap confidence intervals, Friedman tests, Nemenyi critical
difference data, pairwise Wilcoxon tests with Holm correction, sensitivity
analyses, runtime, failures, and workflow complexity.

## Repository Layout

```text
configs/             Q1 generalized, dataset-specific, and smoke configs
data/                synthetic smoke-test stacks and manifests
docs/                methodology, data-access, output, and release guidance
optimized_safe/      optional caching/backend/parallel helpers
performance_audit/   profile-first optimization audit tools and reports
results/             curated publication outputs from the production run
scripts/q1_pipeline/ staged Q1 pipeline entry points
src/autofocus/q1/    focus curves, metrics, labels, GP, statistics, and exports
tests/               deterministic and smoke tests
```

Complete large per-stack curves, transfer rows, and execution logs are stored
in the companion Zenodo results record, not in Git history.

## Installation

Python 3.10 or later is recommended. The archived production environment used
Python 3.12.3.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For the optional Torch CUDA CFM4 backend, install a Torch build appropriate for
your CUDA/driver environment, then install the project requirements. CPU
fallback remains available.

## Smoke Test

```bash
pytest -q

python scripts/q1_pipeline/10_run_full_q1_pipeline.py \
  --config configs/q1_smoke.yaml \
  --smoke
```

The release was validated with 39 passing tests.

## Full Experiments

First edit the five dataset and label paths in
`configs/q1_dataset_specific.yaml` or `configs/q1_generalized.yaml`. See
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Both generalized and dataset-specific experiments
python scripts/q1_pipeline/10_run_full_q1_pipeline.py \
  --config configs/q1_dataset_specific.yaml \
  --run-id q1_cfm4_reproduction

# Resume an interrupted run
python scripts/q1_pipeline/10_run_full_q1_pipeline.py \
  --config configs/q1_dataset_specific.yaml \
  --run-id q1_cfm4_reproduction \
  --resume
```

Stages 05 and 06 run true genetic-programming searches. Every GP candidate is
evaluated by drawing fresh preprocessed focus curves and computing fresh Q1
metrics; GP scores are not reused from single-operator results.

## Archived Production Run

Run ID: `q1_cfm4_20260525_042149`

```text
Configuration hash: 3dc8033161e2f87517552571cb061a3ec25436f1ffc2f985d802d168425060fd
Manifest hash:      12b0c71ec15a4e3d09857050fc8f985563ccbd8aa91039a9d650530f1f54cc9e
Source-tree hash:   ce8873ae6cefca8b901dc4e25a23e859d5c40eaa8254aa2033aea7f051d75514
Random seed:        123
Pipeline failures:  0 (as recorded in run_summary.json)
```

See [`docs/OUTPUTS.md`](docs/OUTPUTS.md) for the GitHub/Zenodo output split and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for important limitations.

## Citation and DOI

Before publishing, replace the creator, ORCID, affiliation, repository URL, and
DOI placeholders in `CITATION.cff` and `.zenodo.json`. GitHub will expose a
"Cite this repository" link when `CITATION.cff` is on the default branch.

## License

Software is released under the MIT License. Original derived numerical outputs,
tables, figures, and documentation are released under CC BY 4.0. These licenses
do not apply to the external microscopy datasets, which are not included.

