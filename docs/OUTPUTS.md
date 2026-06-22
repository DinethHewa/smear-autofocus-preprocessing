# Output Inventory

## GitHub Curated Outputs

`results/q1_cfm4_20260525_042149/` contains:

- label provenance and reference labels;
- dataset/method metric summaries and metric definitions;
- generalized and dataset-specific GP state, selected pipelines, and candidate summaries;
- LODO and dataset-specific dataset-level results;
- transfer matrices, gaps, overfitting summaries, and statistical tests;
- bootstrap, Friedman, Nemenyi, Wilcoxon-Holm, and sensitivity outputs;
- paper-ready SVG, PNG, PDF, CSV, and LaTeX assets;
- exact run environment, config, manifest, stage summaries, and source-tree hash.

## Zenodo Complete Outputs

The companion results record additionally contains:

- `q1_curves/preprocessed_focus_curves.csv`;
- `q1_metrics/per_stack_metrics.csv`;
- `paper1_generalized/lodo_results_per_stack.csv`;
- `paper1_generalized/lodo_selected_focus_curves.csv`;
- `paper2_dataset_specific/transfer_results_per_stack.csv`;
- `paper2_dataset_specific/transfer_focus_curves.csv`; and
- all execution logs.

These files are excluded from Git because several exceed GitHub's 100 MiB
per-file limit and generated data should not inflate repository history.

