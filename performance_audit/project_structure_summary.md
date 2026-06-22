# Project Structure Summary

| File | Purpose | Main Dependencies | Possible Bottleneck | Risk Level |
|---|---|---|---|---|
| `scripts/q1_pipeline/10_run_full_q1_pipeline.py` | Full staged Q1 publication pipeline entry point | `src/autofocus/q1/stages.py` | Runs all stages, including repeated evaluation and plotting | Medium |
| `scripts/q1_pipeline/03_run_preprocessed_focus_curves.py` | Stage 03 focus-curve generation wrapper | Q1 stages, preprocessing pipelines | Stack/method loops, repeated stack reads, CSV append/checkpoint I/O | High |
| `scripts/q1_pipeline/04_evaluate_workflows_q1_metrics.py` | Stage 04 Q1 metric evaluation wrapper | Q1 stages, metrics | Recomputes focus curves/metrics after Stage 03 | High |
| `scripts/q1_pipeline/05_generalized_workflow_search_lodo.py` | Paper 1 LODO generalized workflow search | Q1 stages, aggregation | Repeated method evaluation per held-out dataset | High |
| `scripts/q1_pipeline/06_dataset_specific_workflow_search.py` | Paper 2 dataset-specific workflow search | Q1 stages, aggregation | Repeated evaluation per dataset and candidate workflow | High |
| `scripts/q1_pipeline/07_compare_generalized_vs_specific.py` | Transfer comparison for generalized vs specific workflows | Q1 stages, aggregation | Transfer matrix evaluation repeats workflow scoring | Medium |
| `scripts/q1_pipeline/08_statistics_and_sensitivity.py` | Statistical tests and sensitivity analysis | Q1 statistics, pandas/scipy | Pairwise tests and bootstrap/sensitivity loops | Medium |
| `scripts/q1_pipeline/09_export_paper_assets.py` | Paper-ready figure/table export | Q1 exports, matplotlib, pandas | Matplotlib text/layout and repeated CSV reads | Medium |
| `src/autofocus/q1/stages.py` | Implementation of the staged Q1 workflow | pandas, numpy, Q1 modules | Stage orchestration, repeated reads/writes, repeated method evaluation | High |
| `src/autofocus/q1/workflows.py` | Configured workflows, evaluation, summaries | stack loading, preprocessing, focus measures, Q1 metrics | Nested method/stack loops and stack loading per method | High |
| `src/autofocus/q1/labels.py` | Manifest creation and reference-label provenance | pandas, stack IO, focus measures | Manifest scans, surrogate label stack loading | Medium |
| `src/autofocus/q1/metrics.py` | Ten Q1 autofocus metric calculations | numpy, scipy optional | Noise/RRMSE and curve metric calculations per stack/method | Medium |
| `src/autofocus/q1/aggregation.py` | Rank/value-based Q1 summaries | pandas, numpy | Groupby/rank aggregation over all methods/datasets | Low |
| `src/autofocus/q1/statistics.py` | Bootstrap, Friedman, Wilcoxon/Holm, sensitivity | numpy, pandas, scipy | Pairwise tests and sensitivity sweeps | Medium |
| `src/autofocus/q1/exports.py` | Figure/table export helpers | matplotlib, pandas | Plot rendering and repeated table reads | Medium |
| `src/autofocus/metrics/focus_measures.py` | Focus measure library, including CFM4 | numpy, cv2, pywt optional | Per-image focus measure computation, FFT/wavelet operations | High |
| `src/autofocus/preprocess/ops.py` | Preprocessing operator implementations | cv2, skimage optional, numpy | Per-slice image preprocessing loops | High |
| `src/autofocus/preprocess/param_spaces.py` | Operator search spaces and defaults | preprocessing ops | Low runtime impact; correctness-sensitive | Low |
| `src/autofocus/data/io.py` | Stack/image loading | numpy, cv2/PIL | Repeated `.npy` or image-folder reads | High |
| `src/autofocus/data/manifests.py` | Legacy manifest loading/direct-input expansion | pandas, numpy | CSV reads, group-map row iteration | Low |
| `src/autofocus/eval/evaluate.py` | Legacy evaluation pipeline | stack IO, preprocessing, metrics | Stack loops, row iteration, optional cache use | Medium |
| `src/autofocus/eval/summary.py` | Legacy summary generation | pandas, objectives | CSV reads and row iteration | Low |
| `configs/q1_generalized.yaml` | Paper 1 experiment configuration | Q1 pipeline | Runtime driven by candidate methods and full datasets | High |
| `configs/q1_dataset_specific.yaml` | Paper 2 experiment configuration | Q1 pipeline | Runtime driven by per-dataset workflow searches | High |
| `configs/q1_smoke.yaml` | Fast smoke-test configuration | Q1 pipeline | Small workload, not representative for full performance | Low |
| `tests/` | Regression and smoke tests | pytest | Full suite currently passes with the restored minimal legacy artifact fixture | Low |
