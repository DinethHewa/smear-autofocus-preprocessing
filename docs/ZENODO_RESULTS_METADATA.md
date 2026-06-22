# Zenodo Metadata: Complete Results Record

## Basic Information

**Do you already have a DOI for this upload?** No. Reserve a separate Zenodo DOI.

**Resource type:** Dataset

**Title:** Complete Outputs for CFM4-Based Generalized and Dataset-Specific Preprocessing Optimization in Smear Microscopy Autofocus

**Publication date:** Use the actual first-publication date of this results package.

**Creators:** Use the same creator list and order as the software record unless
contributions justify a documented difference. Add ORCIDs and affiliations.

## Description

Complete derived outputs for production run `q1_cfm4_20260525_042149` of a
reproducible smear microscopy autofocus preprocessing study. The package covers
BMA, PBS, TBF, TBI, and WBC and contains label provenance, reference labels,
raw and normalized CFM4 focus curves, per-stack values for ten autofocus
metrics, generalized leave-one-dataset-out genetic-programming searches,
dataset-specific genetic-programming searches, fresh cross-dataset transfer
evaluation, rank- and value-based aggregation, runtime and failure summaries,
bootstrap confidence intervals, Friedman/Nemenyi and Wilcoxon-Holm outputs,
sensitivity analyses, workflow definitions, paper-ready figures/tables,
execution logs, and reproducibility metadata.

The primary aggregation gives equal weight to each dataset. The production GP
budget used 8 generations, population size 8, elite size 3, maximum pipeline
length 3, and a screening sample of at most 10 stacks per training dataset. The
run used random seed 123 and recorded zero pipeline failures in its final run
summary.

Raw microscopy images and source `.npy` stacks are not redistributed. Reference
labels are recorded as `q1_revision_surrogate`; they are surrogate labels rather
than independent manual expert annotations. Original absolute WSL paths are
retained in exact reproducibility metadata and must be replaced for reruns.

## Additional Description: Archive Parts

- **Core:** labels, summaries, GP states/candidates, statistics, transfer
  matrices, paper assets, configuration, manifest, environment, and hashes.
- **Large CSV:** full preprocessed focus curves, per-stack Q1 metrics, LODO
  per-stack results/curves, dataset-specific per-stack results/curves, and full
  transfer results/curves.
- **Logs:** all stage execution logs, including the long checkpoint/progress log
  from generalized GP search.

Use `ARCHIVE_MANIFEST.csv` and `SHA256SUMS` to verify content and integrity.

## Keywords and Subjects

- microscopy autofocus
- smear microscopy
- focus curves
- autofocus metrics
- preprocessing optimization
- CFM4
- genetic programming
- generalization
- leave-one-dataset-out
- dataset-specific workflow
- transfer matrix
- sensitivity analysis
- reproducibility

Controlled subjects to search for: Microscopy; Image Processing,
Computer-Assisted; Algorithms; Reproducibility of Results.

## Other Fields

**Language:** English

**Version:** `1.0.0`

**Publisher:** Zenodo

**License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`)

**Dates:**

- Created: `2026-05-25`
- Data/results generated: `2026-05-25/2026-06-12`
- Updated/package prepared: `2026-06-22` or actual preparation date

**Funding:** Leave blank unless an actual grant applies; otherwise enter the
same verified grant information as the software record.

**Related works:**

- Software Zenodo DOI: `Is derived from` or nearest available equivalent
- GitHub repository URL: `Is supplemented by`
- Paper 1 DOI, once available: `Is supplement to`
- Paper 2 DOI, once available: `Is supplement to`

**Visibility:** Public only after confirming rights to redistribute all derived
outputs. Use restricted/embargoed files if required by journal or dataset terms.

