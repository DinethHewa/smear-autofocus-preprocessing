# Q1 Preprocessing Pipeline Upgrade Audit

## 1. Current `autofocus_preprocess` Pipeline Stages

The current repository already contains a useful experimental scaffold:

- Data ingestion from direct `.npy` stack paths or manifest CSVs via `src/autofocus/eval/evaluate.py`.
- Deterministic nested group splits with `outer_split` and `inner_split` via `src/autofocus/data/splits.py`.
- Preprocessing operator definitions in `src/autofocus/preprocess/ops.py`.
- Per-operator Optuna tuning via `src/autofocus/preprocess/optuna_single_op.py` and `src/autofocus/cli/run_optuna_per_op.py`.
- GP workflow search over preprocessing categories via `src/autofocus/gp/preprocess_gp.py` and `src/autofocus/cli/run_gp.py`.
- A combined tuning-to-GP workflow in `src/autofocus/cli/run_full.py`.
- A lightweight publication exporter in `src/autofocus/cli/build_q1_paper_outputs.py`.
- Tests covering deterministic behavior, split leakage, objective behavior, pipeline failures, and paper-output formatting.

The current staged flow is therefore:

1. Build/validate manifest.
2. Assign nested train/validation/test splits.
3. Tune each preprocessing operator independently.
4. Use tuned operator parameters inside GP workflow search.
5. Re-evaluate the selected GP workflow.
6. Export summary tables and figures.

## 2. Current Objective and Why It Is Insufficient for Q1 Claims

The current default configuration has no `evaluation.reference_paths`, so evaluation falls back to an unsupervised objective:

```yaml
evaluation:
  mode: raw
  reference_paths: {}
tuning:
  objective_mode: reference_or_unsupervised
  unsupervised_objective: unimodal_peak
```

The unsupervised objective in `src/autofocus/metrics/unsupervised.py` mainly rewards peak prominence and penalizes multiple peaks/flat curves:

```text
score = -peak_prominence + multi_peak_penalty + 10 * flat_penalty
```

This is useful for exploration, but insufficient as a Q1 journal endpoint because:

- It is not anchored to source or surrogate best-focus reference labels.
- It can reward high-amplitude focus curves even when the predicted focus plane is wrong.
- It does not compute the 10 autofocus metrics used in the Q1 revision workflow.
- It does not export raw and normalized curves as auditable artifacts.
- It does not include label provenance or leave-one-out surrogate disclosure.
- It does not use leave-one-dataset-out validation for cross-dataset generalization.
- It does not provide bootstrap confidence intervals, Friedman/Nemenyi/Wilcoxon-Holm tests, or sensitivity analysis.
- The current `run_full.py` evaluates the selected GP workflow on the train/validation pool, not the outer test split, so those outputs must not be described as held-out test results.

## 3. Exact Q1 Metric Framework to Port

The reference methodology in `focus_measure_q1_revision` defines the publication metric framework in:

- `config/settings.py`
- `src/evaluation/autofocus_metrics.py`
- `src/evaluation/aggregation.py`
- `src/evaluation/statistics.py`

The metric framework to port is:

1. Absolute peak localization error: lower is better.
2. Full width at half maximum (FWHM): lower is better.
3. Curvature at peak: higher is better, converted by reciprocal alignment during value aggregation.
4. Steep-slope width: lower is better.
5. Steep-to-gradual slope ratio: higher is better, converted by reciprocal alignment during value aggregation.
6. False maxima count: lower is better.
7. Noise level from second differences: lower is better.
8. Relative RMSE under additive noise: lower is better.
9. Range around global maximum: lower is better.
10. Execution time per slice/image: lower is better.

Required curve conventions:

- Compute raw focus curves per stack.
- Normalize each focus curve using per-stack min-max normalization:
  `F_norm = (F - min(F)) / (max(F) - min(F) + eps)`.
- Preserve native resolution for the main benchmark.
- Store timing per stack/image.
- Store predicted best-focus index using the global maximum with deterministic tie-breaking.
- Store reference focus index and label source/provenance.

Required aggregation conventions:

- Rank-based summary across dataset-by-metric blocks.
- Value-based summary using metric-direction alignment, per-dataset min-max normalization across methods, metric weights, and `alpha = 0.7`.
- Equal-dataset weighting as the primary summary.
- Per-stack weighting and alpha/metric/dataset-weight sensitivity as secondary checks.
- Bootstrap confidence intervals.
- Friedman test, Nemenyi critical difference data, pairwise Wilcoxon signed-rank tests, and Holm correction where assumptions allow.

## 4. Files To Create or Modify

Create:

- `docs/q1_upgrade_audit.md`
- `src/autofocus/q1/curves.py`
- `src/autofocus/q1/metrics.py`
- `src/autofocus/q1/aggregation.py`
- `src/autofocus/q1/statistics.py`
- `src/autofocus/q1/labels.py`
- `src/autofocus/q1/exports.py`
- `src/autofocus/q1/config.py`
- `src/autofocus/q1/workflows.py`
- `scripts/q1_pipeline/01_build_stacks_and_manifest.py`
- `scripts/q1_pipeline/02_build_reference_labels.py`
- `scripts/q1_pipeline/03_run_preprocessed_focus_curves.py`
- `scripts/q1_pipeline/04_evaluate_workflows_q1_metrics.py`
- `scripts/q1_pipeline/05_generalized_workflow_search_lodo.py`
- `scripts/q1_pipeline/06_dataset_specific_workflow_search.py`
- `scripts/q1_pipeline/07_compare_generalized_vs_specific.py`
- `scripts/q1_pipeline/08_statistics_and_sensitivity.py`
- `scripts/q1_pipeline/09_export_paper_assets.py`
- `scripts/q1_pipeline/10_run_full_q1_pipeline.py`
- `configs/q1_generalized.yaml`
- `configs/q1_dataset_specific.yaml`
- `configs/q1_smoke.yaml`
- Tests for parameter spaces, Q1 metrics, labels, splits, aggregation, statistics, assets, resumability, and reproducibility metadata.

Modify:

- `src/autofocus/preprocess/param_spaces.py`
  - Fix `laplacian_sharpen` so it no longer reuses `unsharp_space`.
- Potentially `src/autofocus/preprocess/ops.py` only if operator signatures or exported metadata require small additions.
- Existing tests only if necessary to preserve compatibility.

## 5. Current Artifacts: Safe, Exploratory, Not Final

Safe to use for engineering audit:

- `artifacts/*/config_used.yaml`
- `artifacts/*/manifest_used.csv`
- `artifacts/*/run_metadata.json`
- `artifacts/*/optuna_per_op/per_op_summary.csv`
- Existing tests and smoke artifacts.

Exploratory only:

- Current `optuna_per_op` scores because they use the unsupervised scalar objective.
- Current GP convergence and `best_pipeline.json` because they optimize the unsupervised scalar objective.
- `paper_outputs/20260102_014123_c6f8c289/*` because it is a useful package-format prototype but not a final Q1 evidence package.
- Dataset-oracle single-operator outputs, which can be retained only as explicitly labeled oracle/exploratory references.

Should not be used for final manuscript claims:

- Any score derived only from `total_fitness` under the unsupervised objective.
- Any result described as "held-out test" when it was computed on `trainval` rows.
- `gp/per_stack_results_gp_eval.csv` as a final selected-pipeline result. This file is overwritten during GP search and can represent the last evaluated individual, not the final hall-of-fame workflow.
- Current `git_hash: unknown` metadata as a reproducibility guarantee.

## 6. Known Implementation Bugs and Risks

- `laplacian_sharpen` parameter-space mismatch:
  - `laplacian_sharpen(img, ksize=3, amount=1.0)` does not accept `sigma`.
  - `PARAM_SPACES["laplacian_sharpen"]` currently maps to `unsharp_space`, which returns `sigma` and `amount`.
  - This caused observed all-stack failures such as `laplacian_sharpen() got an unexpected keyword argument 'sigma'`.
- Operator availability is environment-dependent:
  - `macenko_normalization`, `vahadane_normalization`, and sometimes `guided_filter` are unavailable and must be disclosed in exported parameter-space tables.
- Category labels must be consistent:
  - The code uses lowercase categories such as `color`; exports should use a single display convention such as `Color`.
- Source labels may not exist:
  - The upgraded pipeline must support source labels, surrogate labels, and leave-one-out surrogate labels, and must disclose provenance per stack.
- The repository is not a git repository:
  - Final reproducibility must compute and store a source-tree hash over `.py`, `.yaml`, and `.toml` files.
