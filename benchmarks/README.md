# Benchmarks

Performance benchmarks for the Talwani computation engine.

## Running the benchmarks

From the project root (with the virtual environment activated):

```bash
python -m benchmarks.bench_talwani
python -m benchmarks.bench_calculator
```

Or with `uv run`:

```bash
uv run python -m benchmarks.bench_talwani
uv run python -m benchmarks.bench_calculator
```

## bench_talwani

Times `compute_polygon_anomaly()` directly for two synthetic cases:

| Case | Obs points | Vertices |
|------|-----------|----------|
| Baseline | 201 | 4 |
| Large | 1000 | 8 |

Reports the best wall-time (ms) per call over 5 × 20 iterations.

## bench_calculator

Times the full `calculate_anomaly()` pipeline in both serial and parallel
modes across three model sizes:

| Case | Bodies | Obs points |
|------|--------|-----------|
| Simple | 1 | 201 |
| Multi | 6 | 201 |
| Large | 20 | 1000 |

Reports serial time, parallel time, and speedup ratio.

## Interpreting results

- **Speedup ≥ 2×** on the 500+ obs-point cases validates the vectorization goal.
- Parallel speedup depends on the number of CPU cores and process-spawn overhead;
  gains are most visible for multi-body models with large observation grids.
- Run on a quiet machine (no other heavy processes) for reproducible numbers.

## pytest-benchmark integration

If `pytest-benchmark` is installed (listed as an optional dev dependency), you
can also run benchmarks via pytest:

```bash
uv run pytest --benchmark-only
```
