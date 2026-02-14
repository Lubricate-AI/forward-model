"""Magnetic anomaly computation algorithms."""

from forward_model.compute.calculator import calculate_anomaly
from forward_model.compute.talwani import (
    compute_polygon_anomaly,
    field_to_magnetization,
)

__all__ = ["calculate_anomaly", "compute_polygon_anomaly", "field_to_magnetization"]
