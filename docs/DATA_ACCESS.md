# Data Access and Configuration

## What Is Included

`data/stack1.npy`, `data/stack2.npy`, and `data/q1_smoke_manifest.csv` are small
synthetic fixtures used only for tests and smoke runs.

## What Is Not Included

The BMA, PBS, TBF, TBI, and WBC microscopy images/source stacks are not
redistributed. Their ownership and licenses must be checked with the original
dataset providers before public redistribution.

The archived production run used surrogate labels imported from the companion
`focus_measure_q1_revision` workflow. This provenance is recorded as
`q1_revision_surrogate` in the production metadata.

## Configure Local Data

Place local arrays in `data/external/` or edit the config paths directly. Each
stack array should have shape `(N, Z, H, W)`. Each label array should contain
one reference focus index per stack.

Expected portable names:

```text
data/external/WBC_stacks.npy
data/external/WBC_surrogate_labels.npy
data/external/TBI_stacks.npy
data/external/TBI_surrogate_labels.npy
data/external/PBS_stacks.npy
data/external/PBS_surrogate_labels.npy
data/external/BMA_stacks.npy
data/external/BMA_surrogate_labels.npy
data/external/TBF_stacks.npy
data/external/TBF_surrogate_labels.npy
```

Alternatively, set each dataset `path` to a directory containing one image
folder per z-stack. Do not describe evaluations as held-out unless the config
and manifest identify a genuine outer test split.

