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
