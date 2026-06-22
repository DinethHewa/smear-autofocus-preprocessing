# Optimization Report

## 1. Project Structure Summary

See `performance_audit/project_structure_summary.md`.

## 2. Environment Report

Environment detection was run with:

```bash
../.venv/bin/python performance_audit/environment_report.py
```

Key result: Python 3.12.3 on WSL2/Linux, 32 logical CPU cores, NVIDIA GeForce RTX 4060 Laptop GPU visible through `nvidia-smi`, PyTorch CUDA available, but `polars`, `cudf`, `cupy`, `cuml`, `dask`, `numba`, and `psutil` are not installed.

## 3. GPU Availability Report

GPU hardware is visible, but the dataframe/array GPU libraries needed for low-risk acceleration are absent. The current safe path is CPU-only with optional future GPU acceleration for large batched numeric focus-measure calculations.

## 4. Detected Bottlenecks

The static scanner reported 127 findings before source edits and 116 after the safe iteration changes. The most important bottlenecks are:

| Bottleneck | Location | Severity | Applied Fix | Expected Speedup | Measured Speedup | Risk |
|---|---|---:|---|---|---|---|
| Repeated stack/method focus evaluation | `src/autofocus/q1/workflows.py`, `src/autofocus/q1/stages.py` | High | Only row-iteration cleanup applied | Small in smoke, larger on huge manifests | Smoke full pipeline 1.13x | Low for applied part |
| Repeated stack loading across methods/stages | `evaluate_methods()` calls from stages 03-07 | High | Recommendation only | Potentially large | Not measured | High |
| Matplotlib text/layout rendering | Paper asset export via `src/autofocus/q1/exports.py` | Medium | Recommendation only | Moderate | Profile shows dominant overhead under cProfile | Medium |
| Pairwise statistical testing loops | `src/autofocus/q1/statistics.py` | Medium | Recommendation only | Small/medium | Not measured separately | Medium |
| Repeated CSV reads/writes | Q1 stages and exports | Medium | Optional cache helpers added | Moderate on large outputs | Not measured on full output | Low if enabled with validation |
| Pandas row iteration | Q1 labels/workflows, legacy manifest/eval summary | Low/Medium | Replaced read-only `iterrows()` with `to_dict("records")` | Small/medium | Included in smoke 1.13x | Low |

## 5. Runtime Profiling Results

Profiling command:

```bash
../.venv/bin/python performance_audit/profile_runner.py --script scripts/q1_pipeline/10_run_full_q1_pipeline.py -- --config configs/q1_smoke.yaml --smoke --run-id perf_profile_original_20260525_041000
```

Profile output: `performance_audit/results/cprofile_scripts_q1_pipeline_10_run_full_q1_pipeline_py.txt`.

Important cumulative-time findings under cProfile:

| Function/Area | Cumulative Time |
|---|---:|
| Matplotlib text extent/layout | ~28.7 s under cProfile |
| Module imports | ~9.7 s |
| `stage08_statistics_and_sensitivity` | ~5.6 s |
| `workflows.evaluate_methods` | ~3.1 s |
| `statistics.friedman_wilcoxon_holm` | ~2.8 s |

The cProfile run took 32.45 s due profiler overhead; normal smoke runtime is the benchmark source of truth.

## 6. Memory Profiling Results

`tracemalloc` peak during cProfile was 180,953,327 bytes. The subprocess benchmark RSS fallback measured 384,507,904 bytes on the optimized smoke run. Original peak RSS was unavailable because `psutil` is not installed and the `/proc` fallback was added after the first baseline benchmark.

## 7. Safe Optimizations Applied

- Replaced read-only `iterrows()` loops with deterministic `to_dict("records")` loops in Q1 workflow/label paths.
- Replaced a `range(len(df))`-style manifest ID fallback with index enumeration.
- Applied the same low-risk row-iteration cleanup to legacy manifest/evaluation summary paths.
- Added performance audit tooling for environment detection, static scanning, profiling, benchmarking, and output validation.
- Added optional helper modules for dataframe caches, Parquet conversion, Polars fallback selection, GPU detection, deterministic parallel mapping, and a compatibility pipeline wrapper.

## 8. Optimizations Not Applied Due To Risk

- No GPU rewrite was applied.
- No Polars rewrite was applied because Polars is not installed and pandas behavior is currently tested.
- No multiprocessing was inserted into scientific stages.
- No stage-level cache was used to skip focus-curve recomputation.
- No change was made to labels, splits, metric equations, preprocessing definitions, random seeds, or paper output layout.

## 9. Recommended Future Optimizations

See `performance_audit/remaining_recommendations.md`.

## 10. CPU-Only Acceleration Path

Current safe path:

```bash
../.venv/bin/python scripts/q1_pipeline/10_run_full_q1_pipeline.py --config configs/q1_smoke.yaml --smoke
```

The current pipeline is CPU-safe and remains fully reproducible.

## 11. GPU-Optional Acceleration Path

GPU should only be used for future large numeric batches, for example batched FFT/wavelet/focus-measure operations. The code now includes `optimized_safe/gpu_backend.py` for safe detection, but no Q1 scientific stage depends on GPU libraries.

## 12. Multiprocessing Opportunities

Best candidate: independent stack-level or stack-method-level focus-curve jobs. This was not enabled because full `.npy` stacks can be memory-heavy and output order/checkpoint behavior must remain deterministic.

## 13. Disk I/O And Caching Improvements

`optimized_safe/cache_utils.py` and `optimized_safe/build_cache.py` support optional Parquet caches for internal repeated CSV reads. Final paper outputs remain CSV/LaTeX/SVG/PNG/PDF.

## 14. Before/After Benchmark Table

| Stage | Original Runtime | Optimized Runtime | Speedup | Validation Status | Notes |
|---|---:|---:|---:|---|---|
| Full Q1 smoke pipeline | 7.274736 s | 6.439032 s | 1.13x | Partially passed | Labels and curves passed; runtime-sensitive aggregate scores expectedly differ |
| Full Q1 smoke pipeline with RSS fallback | N/A | 6.813186 s | N/A | Not compared | Peak RSS 384,507,904 bytes |

Raw benchmark file: `performance_audit/speedup_summary.csv`.

## 15. Output Validation Result

Validation is partial and honest:

- `labels/`: passed, 2 files.
- `q1_curves/`: passed when ignoring `runtime_sec_per_stack`.
- `q1_metrics/`: per-stack, dataset summary, and method summary passed when ignoring runtime/timing columns and final aggregate score columns.
- Runtime-sensitive aggregate rank/value summaries differ because execution time is one of the Q1 endpoints and timing naturally changes between runs.

Test status:

```bash
../.venv/bin/python -m pytest -q
# 33 passed in 6.71s
```

## 16. Remaining Bottlenecks

The largest remaining bottleneck is architectural: the full Q1 pipeline repeatedly loads stacks and recomputes focus curves across stage boundaries and method comparisons. That should be optimized only after defining a curve-cache contract that preserves failure provenance and runtime endpoint interpretation.

## 17. Reproduction Commands

```bash
../.venv/bin/python performance_audit/environment_report.py
../.venv/bin/python performance_audit/bottleneck_scanner.py
../.venv/bin/python performance_audit/profile_runner.py --script scripts/q1_pipeline/10_run_full_q1_pipeline.py -- --config configs/q1_smoke.yaml --smoke --run-id perf_profile_original_20260525_041000
../.venv/bin/python performance_audit/benchmark_runner.py --target original --script scripts/q1_pipeline/10_run_full_q1_pipeline.py -- --config configs/q1_smoke.yaml --smoke --run-id perf_original_20260525_040825
../.venv/bin/python performance_audit/benchmark_runner.py --target optimized --script scripts/q1_pipeline/10_run_full_q1_pipeline.py -- --config configs/q1_smoke.yaml --smoke --run-id perf_optimized_20260525_041229
../.venv/bin/python performance_audit/validate_outputs.py --original paper_outputs/perf_original_20260525_040825/labels --optimized paper_outputs/perf_optimized_20260525_041229/labels
../.venv/bin/python performance_audit/validate_outputs.py --original paper_outputs/perf_original_20260525_040825/q1_curves --optimized paper_outputs/perf_optimized_20260525_041229/q1_curves --ignore-columns runtime_sec_per_stack
```
