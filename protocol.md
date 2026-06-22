# Evaluation Protocol

## 1) Scope and purpose
This protocol evaluates preprocessing pipelines for microscopy autofocus by measuring focus-curve quality and
generalization across datasets.

## 2) Datasets
Datasets: WBC, TBI, PBS, BMA, TBF. Each dataset is provided as z-stacks. Each stack has a group_id used to
prevent leakage in split assignment.

## 3) Data ingestion
Two ingestion modes are supported:
- direct_inputs: datasets are read from the configured paths and an in-memory manifest is built.
- manifest_path: a CSV manifest is loaded from disk.
For reproducibility, the manifest used for each run must be written to `artifacts/<run_id>/manifest_used.csv`.

## 4) Split protocol (frozen)
Splits are group-disjoint and deterministic given the configured seed.
- Outer split: `outer_split` in {trainval, test}. Groups in test never appear in trainval.
- Inner split: within trainval only, `inner_split` in {train, val}. Groups in test remain test and are never used for tuning.
Default sizes: `outer_test_size=0.2`, `inner_val_size=0.2`. Assignment is seeded by `seed` in the config.

## 5) Preprocessing pipeline definition
A pipeline is an ordered list of steps with `name`, `enabled`, `params`, and `category`.
The pipeline fingerprint is logged via `pipeline.fingerprint()`.

## 6) Reference peaks (ground truth)
Reference peaks are provided by `evaluation.reference_paths` as a mapping per dataset to a CSV with
`reference_peak` and optional `confidence` columns.

## 7) Primary endpoint
Per-stack fitness is aggregated to dataset mean_fitness values. The primary endpoint is:
`generalization_score = weighted_mean + weighted_std` across datasets.
`stability_ratio = |weighted_mean| / (weighted_std + eps)` is computed for stability.
Penalties apply for nonfinite curves, hard failure rate, and stability threshold violations.

## 8) Runtime and hardware constraints
Runs must report normalized ms/image and store this metric in artifacts or logs. Target constraints:
< 5 ms/image on GPU or < 30 ms/image on CPU. These are constraints, not performance claims.

## 9) Reproducibility checklist
- `artifacts/<run_id>/run_metadata.json` includes seed, config hash, and backend info.
- `artifacts/<run_id>/manifest_used.csv` is required.
- `configs_used.yaml` must be stored in the run directory.
- Record git commit hash if running from a git clone.

## 10) Change control
Protocol changes require a version bump and must not be made after starting final experiments.
