# Zenodo Metadata: Software Record

## Basic Information

**Do you already have a DOI for this upload?** No. Reserve a Zenodo DOI.

**Resource type:** Software

**Title:** CFM4-Based Preprocessing Optimization for Smear Microscopy Autofocus: Generalized and Dataset-Specific Workflows

**Publication date:** Use the actual date on which this version is first made public. Use `2026-06-22` only if publishing on that date.

**Creators:** Add every software/research creator in agreed publication order.

For each person enter:

- Family name: `REPLACE_WITH_FAMILY_NAME`
- Given names: `Dineth` (replace/extend as needed)
- ORCID: `REPLACE_WITH_ORCID`
- Affiliation: `REPLACE_WITH_AFFILIATION`
- ROR: select the matching institution if available

## Description

Version 1.0.0 of a reproducible Python pipeline for preprocessing optimization
in smear microscopy autofocus. The software implements stack and manifest
construction, source/surrogate label provenance, raw and per-stack min-max
normalized focus curves, polarity alignment, the CFM4 focus measure, ten
autofocus metrics, leave-one-dataset-out generalized genetic-programming
search, dataset-specific genetic-programming search, fresh cross-dataset
transfer evaluation, bootstrap and nonparametric statistics, sensitivity
analysis, checkpointing, resumability, optional Torch CUDA acceleration with
CPU fallback, and paper-ready figure/table export.

The workflow supports BMA, PBS, TBF, TBI, and WBC through user-supplied stack
and label paths. Raw microscopy datasets are not included. Complete derived
outputs for production run q1_cfm4_20260525_042149 are available in a companion
Zenodo results record.

## Additional Description: Methods

CFM4 is implemented as `FTSI - WDE-db1 * psqrt(psqrt(BG - GSE))`, where
`psqrt(x) = sqrt(abs(x) + epsilon)`. Evaluation uses ten autofocus metrics:
peak localization error, FWHM, curvature at peak, steep-slope width,
steep-to-gradual ratio, false maxima count, noise level, relative RMSE under
additive noise, high-response range, and execution time. The primary endpoint
is an equal-dataset rank-based generalization score. Secondary analyses include
value-based aggregation at alpha 0.7, runtime, failures, complexity, bootstrap
confidence intervals, Friedman/Nemenyi analysis, Wilcoxon-Holm tests, and
sensitivity analysis.

## Keywords and Subjects

Add these custom keywords:

- microscopy autofocus
- smear microscopy
- preprocessing optimization
- focus measure
- focus curves
- CFM4
- genetic programming
- leave-one-dataset-out
- dataset-specific workflow
- image processing
- computational microscopy
- reproducible research
- CUDA

Search Zenodo's controlled vocabularies for these subjects and select exact
matches where available:

- Microscopy (MeSH)
- Image Processing, Computer-Assisted (MeSH)
- Algorithms (MeSH)
- Reproducibility of Results (MeSH)

## Other Fields

**Language:** English

**Version:** `1.0.0`

**Publisher:** Zenodo

**License:** MIT

**Dates:**

- Created: `2026-05-25`
- Updated: `2026-06-22` or the actual release-preparation date

**Funding:** Leave blank if no grant funded the work. Otherwise select the
actual funder and enter the exact grant number/title. Do not invent a grant or
enter personal funding as an institutional award.

**Software fields:**

- Programming language: Python
- Repository: `https://github.com/REPLACE_WITH_GITHUB_USERNAME/smear-autofocus-preprocessing`
- Code repository/release: link tag `v1.0.0`
- Operating system: Linux; tested under WSL2
- Runtime: CPU, with optional NVIDIA CUDA through PyTorch
- Development status: Active

**Related works:**

- GitHub `v1.0.0` release URL: `Is identical to`
- Companion Zenodo results DOI: `Is supplemented by` or nearest equivalent
- Paper 1 DOI, once available: `Is supplement to`
- Paper 2 DOI, once available: `Is supplement to`

**Visibility:** Public, unless journal/data agreements require an embargo.

**Communities:** Add relevant institutional or microscopy/open-science
communities only if their scope and submission rules match.

