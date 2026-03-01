"""Magnetic, gravity, and heat flow anomaly computation algorithms."""

from forward_model.compute.batch import BatchResult, batch_calculate
from forward_model.compute.calculator import calculate_anomaly
from forward_model.compute.gravity import GravityComponents, calculate_gravity
from forward_model.compute.heatflow_talwani import (
    HeatFlowComponents,
    calculate_heat_flow,
)
from forward_model.compute.talwani import (
    AnomalyComponents,
    MagneticComponents,
    compute_demagnetization_factor,
    compute_polygon_anomaly,
    compute_polygon_anomaly_2_5d,
    compute_polygon_anomaly_2_75d,
    field_to_magnetization,
)

__all__ = [
    "AnomalyComponents",
    "BatchResult",
    "batch_calculate",
    "calculate_anomaly",
    "calculate_gravity",
    "compute_demagnetization_factor",
    "compute_polygon_anomaly",
    "compute_polygon_anomaly_2_5d",
    "compute_polygon_anomaly_2_75d",
    "field_to_magnetization",
    "GravityComponents",
    "HeatFlowComponents",
    "calculate_heat_flow",
    "MagneticComponents",
]
