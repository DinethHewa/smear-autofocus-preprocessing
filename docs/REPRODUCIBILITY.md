# Reproducibility Notes

## Production Run

- Run ID: `q1_cfm4_20260525_042149`
- Final completion: 2026-06-12
- Seed: `123`
- Datasets: BMA, PBS, TBF, TBI, WBC
- Primary weighting: equal dataset
- GP search budget: 8 generations, population 8, elite size 3
- GP screening sample: maximum 10 stacks per training dataset
- Focus measure: CFM4
- Backend: `torch_cuda_auto` with CPU fallback

The archived run summary reports zero pipeline failures. The source directory
was not a Git repository, so reproducibility uses a source-tree SHA-256 hash.

## Important Interpretation Limits

1. Reference labels are recorded as `q1_revision_surrogate`, not independent
   manual expert annotations.
2. Dataset-oracle single-operator results are exploratory and must not be
   presented as deployable held-out baselines.
3. Dataset-specific improvements should be interpreted with transfer,
   overfitting, runtime, and complexity results.
4. The production CUDA config had backend validation disabled. The code offers
   optional CPU/CUDA curve validation; backend-equivalence claims require a
   separate reported validation experiment.
5. Exact archived metadata contains original absolute WSL paths for provenance.
   These paths must be replaced in a new environment.
6. Raw microscopy datasets are not included, so a full independent rerun also
   requires legitimate access to the source stacks and labels.

## Verification

```bash
pytest -q
sha256sum -c SHA256SUMS
```

