"""Magnetic anomaly computation algorithms."""

from forward_model.compute.batch import BatchResult, batch_calculate
from forward_model.compute.calculator import calculate_anomaly
from forward_model.compute.talwani import (
    compute_demagnetization_factor,
    compute_polygon_anomaly,
    field_to_magnetization,
)

__all__ = [
    "BatchResult",
    "batch_calculate",
    "calculate_anomaly",
    "compute_demagnetization_factor",
    "compute_polygon_anomaly",
    "field_to_magnetization",
]
