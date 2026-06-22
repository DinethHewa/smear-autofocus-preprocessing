# GitHub Publishing Instructions

## Repository Metadata

**Repository name:** `smear-autofocus-preprocessing`

**Description:** Reproducible CFM4-based preprocessing optimization for smear microscopy autofocus: ten focus-curve metrics, label provenance, generalized LODO GP, dataset-specific GP, transfer analysis, statistics, sensitivity tests, CUDA fallback, and paper-ready exports.

**Topics:** `autofocus`, `microscopy`, `image-processing`, `focus-measure`,
`genetic-programming`, `computer-vision`, `scientific-python`,
`reproducible-research`, `cuda`, `medical-imaging`

## Before the First Commit

1. Replace every `REPLACE_WITH_...` placeholder.
2. Confirm that you may publish all derived outputs.
3. Confirm that no raw clinical/patient data are present.
4. Run `pytest -q`.
5. Run `find . -type f -size +99M -print`; it should return nothing.

## Create and Push

Create an empty public repository on GitHub without initializing README,
license, or `.gitignore`, then run inside `PreprocessingGithub`:

```bash
git init
git branch -M main
git add .
git commit -m "Release reproducible CFM4 autofocus preprocessing pipeline v1.0.0"
git remote add origin https://github.com/REPLACE_WITH_GITHUB_USERNAME/smear-autofocus-preprocessing.git
git push -u origin main
```

## Create the Software Release

1. Open **Releases** and select **Draft a new release**.
2. Create tag `v1.0.0` targeting `main`.
3. Release title: `v1.0.0 - Q1 preprocessing optimization pipeline`.
4. Summarize the generalized and dataset-specific pipelines.
5. State that complete large outputs are deposited in the companion Zenodo
   results record.
6. Publish only after the Zenodo metadata placeholders are resolved.

This release package uses a manually prepared Zenodo software archive. Do not
also enable automatic Zenodo ingestion for the same `v1.0.0` release. If you
choose automatic GitHub archiving instead, omit the manual software record.
`.zenodo.json` takes precedence over `CITATION.cff` during automatic ingestion,
so keep both synchronized in either case.
