# Validation Report

| Output | Match Status | Difference | Acceptable? | Notes |
|---|---|---|---|---|
| `labels/` | Passed | None | Yes | `reference_labels.csv` and `label_provenance_summary.csv` matched between baseline and optimized smoke runs. |
| `q1_curves/` | Passed | Runtime column ignored | Yes | Raw and normalized curves matched with `runtime_sec_per_stack` ignored. |
| `q1_metrics/per_stack_metrics.csv` | Passed | Runtime columns ignored | Yes | Non-runtime Q1 metric values matched with `runtime_sec_per_stack` and `execution_time_per_slice` ignored. |
| `q1_metrics/dataset_metric_summary.csv` | Passed | Runtime columns ignored | Yes | Non-runtime dataset metric summaries matched. |
| `q1_metrics/method_summary.csv` | Passed | Runtime and aggregate score/rank columns ignored | Yes | Failure counts/rates and non-runtime summary fields matched. |
| `q1_metrics/method_rank_summary.csv` | Expected difference | Runtime-sensitive aggregate rank fields differed | Review | Execution time is a Q1 endpoint, so timing changes can alter aggregate rank/value summaries. |
| `q1_metrics/method_value_summary.csv` | Expected difference | Runtime-sensitive aggregate value fields differed | Review | Do not use runtime-sensitive aggregate equality as a deterministic validation criterion. |

Validation commands used:

```bash
../.venv/bin/python performance_audit/validate_outputs.py --original paper_outputs/perf_original_20260525_040825/labels --optimized paper_outputs/perf_optimized_20260525_041229/labels
../.venv/bin/python performance_audit/validate_outputs.py --original paper_outputs/perf_original_20260525_040825/q1_curves --optimized paper_outputs/perf_optimized_20260525_041229/q1_curves --ignore-columns runtime_sec_per_stack
../.venv/bin/python performance_audit/validate_outputs.py --original paper_outputs/perf_original_20260525_040825/q1_metrics --optimized paper_outputs/perf_optimized_20260525_041229/q1_metrics --ignore-columns runtime_sec_per_stack,execution_time_per_slice,execution_time_per_slice_mean,execution_time_per_slice_std,mean_execution_time,rank_generalization_score,rank_final_rank,value_generalization_score,value_final_rank
```
