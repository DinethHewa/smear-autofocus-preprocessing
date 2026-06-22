# Remaining Recommendations

These changes were not applied automatically because they can alter execution semantics, memory use, runtime metrics, or checkpoint behavior. They should be implemented only behind explicit validation gates.

| Recommendation | Expected Benefit | Why Not Applied Automatically | Required Validation |
|---|---|---|---|
| Reuse Stage 03 focus curves in Stage 04/05/06/07 instead of recomputing curves from stacks | Large speedup on full runs | The Q1 execution-time metric, RRMSE-under-noise, failure provenance, and method-level runtime summaries must remain defensible | Compare all non-runtime Q1 metrics and explicitly redefine/runtime-measure timing outputs |
| Load each stack once and evaluate all methods for that stack before moving to the next stack | Large speedup from reduced `.npy`/image-folder reads | Current checkpoint order is method-major; changing to stack-major changes resumability behavior and memory pressure | Bitwise curve comparison, checkpoint resume test, memory benchmark on largest dataset |
| Add bounded multiprocessing for independent stack-method or stack-level tasks | Potentially high speedup on 32 CPU cores | Large stacks may duplicate memory, and parallel disk reads can be slower or unstable | Fixed-order output validation, peak RSS benchmark, seed/noise determinism test |
| Add Parquet caches for repeated internal CSV reads in Q1 stages | Moderate speedup for large paper outputs | Final paper outputs must remain CSV/LaTeX, and cache invalidation must be tied to config/source hashes | Row/column/dtype validation and cache-staleness tests |
| Install and use Polars for large summary tables | Moderate speedup for large aggregation/statistics tables | Polars is not installed in this environment; pandas semantics are currently known and tested | Pandas-vs-Polars result equivalence on full outputs |
| Use CuPy/RAPIDS for batched FFT/wavelet/numeric focus operations | Possible speedup for large numeric batches | `cupy`, `cudf`, and `cuml` are not installed; current pipeline is image/IO-loop dominated | CPU/GPU numerical tolerance tests and transfer-overhead benchmark |
| Optimize matplotlib asset export | Moderate speedup in smoke/profile runs | Asset export is publication-facing; simplifying rendering can affect figure quality | Visual review plus file-existence/format tests |
