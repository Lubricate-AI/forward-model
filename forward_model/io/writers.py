"""Result export to CSV and JSON files."""

import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from forward_model.models.model import ForwardModel


def write_csv(
    filepath: str | Path,
    observation_x: list[float],
    anomaly: NDArray[np.float64],
) -> None:
    """Write anomaly results to CSV file.

    Creates a simple tabular CSV format with columns for x-coordinate
    and magnetic anomaly value.

    Args:
        filepath: Output file path. Can be a string or Path object.
        observation_x: List of x-coordinates for observation points (meters).
        anomaly: Array of magnetic anomaly values (nanoTesla).

    Example:
        >>> write_csv("results.csv", model.observation_x, anomaly)
    """
    filepath = Path(filepath)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "anomaly_nT"])
        for x, anom in zip(observation_x, anomaly, strict=True):
            writer.writerow([x, anom])


def write_json(
    filepath: str | Path,
    model: ForwardModel,
    anomaly: NDArray[np.float64],
) -> None:
    """Write model and anomaly results to JSON file.

    Creates a JSON file containing both the model definition and
    the computed anomaly results.

    Args:
        filepath: Output file path. Can be a string or Path object.
        model: The forward model that was used for computation.
        anomaly: Array of magnetic anomaly values (nanoTesla).

    Example:
        >>> write_json("results.json", model, anomaly)
    """
    filepath = Path(filepath)

    output = {
        "model": model.model_dump(),
        "results": {
            "observation_x": model.observation_x,
            "anomaly_nT": anomaly.tolist(),
        },
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


def write_numpy(
    filepath: str | Path,
    observation_x: list[float],
    anomaly: NDArray[np.float64],
    compressed: bool = False,
) -> None:
    """Write anomaly results to NumPy format.

    Supports both .npy (single array) and .npz (named arrays) formats.
    The format is determined by the file extension.

    For .npy files:
        Creates a 2D array with shape (N, 2) where columns are [x, anomaly].

    For .npz files:
        Creates a dictionary with keys 'x' and 'anomaly'.
        Use compressed=True to create a compressed .npz file.

    Args:
        filepath: Output file path (.npy or .npz). Can be string or Path.
        observation_x: List of x-coordinates for observation points (meters).
        anomaly: Array of magnetic anomaly values (nanoTesla).
        compressed: If True and filepath ends with .npz, uses compression.

    Example:
        >>> write_numpy("results.npy", model.observation_x, anomaly)
        >>> write_numpy("results.npz", model.observation_x, anomaly, compressed=True)
    """
    filepath = Path(filepath)

    # Determine format from extension
    if filepath.suffix == ".npz":
        # NPZ format: dictionary with named arrays
        x_array = np.array(observation_x)

        if compressed:
            np.savez_compressed(filepath, x=x_array, anomaly=anomaly)
        else:
            np.savez(filepath, x=x_array, anomaly=anomaly)

    else:
        # NPY format (or default): 2D array with [x, anomaly] columns
        x_array = np.array(observation_x)
        combined = np.column_stack([x_array, anomaly])
        np.save(filepath, combined)
