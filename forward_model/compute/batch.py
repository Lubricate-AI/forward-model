"""Batch processing of multiple forward models."""

import csv
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from os import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.calculator import calculate_anomaly
from forward_model.io.loaders import load_model, load_model_from_csv
from forward_model.io.writers import write_csv, write_json, write_numpy


def _new_str_list() -> list[str]:
    return []


def _new_str_dict() -> dict[str, str]:
    return {}


@dataclass
class BatchResult:
    """Result of a batch processing run."""

    succeeded: list[str] = field(default_factory=_new_str_list)
    failed: dict[str, str] = field(default_factory=_new_str_dict)
    summary: NDArray[np.float64] | None = None


def _process_single(
    model_path: Path,
    output_dir: Path,
    fmt: Literal["csv", "json", "npy"],
    plot: bool,
) -> tuple[NDArray[np.float64], list[float]]:
    """Process a single model file.

    Top-level function to support pickling in ProcessPoolExecutor.

    Args:
        model_path: Path to the model file (JSON or CSV).
        output_dir: Directory to write output to.
        fmt: Output format (csv, json, npy).
        plot: Whether to save a combined plot.

    Returns:
        Tuple of (anomaly array, observation_x list).
    """
    if model_path.suffix.lower() == ".csv":
        model = load_model_from_csv(model_path)
    else:
        model = load_model(model_path)

    anomaly = calculate_anomaly(model)

    stem = model_path.stem
    out_path = output_dir / f"{stem}.{fmt}"

    if fmt == "csv":
        write_csv(out_path, model.observation_x, anomaly)
    elif fmt == "json":
        write_json(out_path, model, anomaly)
    elif fmt == "npy":
        write_numpy(out_path, model.observation_x, anomaly)

    if plot:
        import matplotlib.pyplot as plt

        from forward_model.viz.plotter import plot_combined

        fig = plot_combined(model, anomaly, save_path=output_dir / f"{stem}.png")
        plt.close(fig)

    return anomaly, model.observation_x


def batch_calculate(
    model_paths: Sequence[str | Path],
    output_dir: str | Path,
    fmt: Literal["csv", "json", "npy"] = "csv",
    parallel: bool = False,
    max_workers: int | None = None,
    continue_on_error: bool = True,
    write_summary: bool = False,
    plot: bool = False,
) -> BatchResult:
    """Process multiple model files in batch.

    Args:
        model_paths: List of model file paths (JSON or CSV).
        output_dir: Directory for output files. Created if absent.
        fmt: Output format for results (csv, json, npy).
        parallel: If True, process models concurrently via ProcessPoolExecutor.
        max_workers: Number of parallel workers. Defaults to min(cpu_count, N).
        continue_on_error: If True, record errors and continue; else re-raise.
        write_summary: If True, write batch_summary.csv with cross-model statistics.
        plot: If True, save a combined plot for each model.

    Returns:
        BatchResult with succeeded paths, failed paths with errors, and optional
        summary array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in model_paths]
    result = BatchResult()
    anomaly_arrays: list[NDArray[np.float64]] = []
    obs_x_lists: list[list[float]] = []

    if parallel:
        n_workers = max_workers or min((cpu_count() or 1), len(paths))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single, p, output_dir, fmt, plot): p
                for p in paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    anomaly, obs_x = future.result()
                    result.succeeded.append(str(path))
                    anomaly_arrays.append(anomaly)
                    obs_x_lists.append(obs_x)
                except Exception as e:
                    if continue_on_error:
                        result.failed[str(path)] = str(e)
                    else:
                        raise
    else:
        for path in paths:
            try:
                anomaly, obs_x = _process_single(path, output_dir, fmt, plot)
                result.succeeded.append(str(path))
                anomaly_arrays.append(anomaly)
                obs_x_lists.append(obs_x)
            except Exception as e:
                if continue_on_error:
                    result.failed[str(path)] = str(e)
                else:
                    raise

    if write_summary and len(anomaly_arrays) >= 2:
        first_x = obs_x_lists[0]
        grids_match = all(x == first_x for x in obs_x_lists[1:])

        if grids_match:
            stacked = np.stack(anomaly_arrays, axis=0)
            x_arr = np.array(first_x)
            mean_arr = stacked.mean(axis=0)
            min_arr = stacked.min(axis=0)
            max_arr = stacked.max(axis=0)
            std_arr = stacked.std(axis=0)

            summary = np.column_stack([x_arr, mean_arr, min_arr, max_arr, std_arr])
            result.summary = summary

            summary_path = output_dir / "batch_summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_m", "mean_nT", "min_nT", "max_nT", "std_nT"])
                for row in summary:
                    writer.writerow(row.tolist())

    return result
