"""Data models for forward modeling."""

from forward_model.models.base import ObservationModel
from forward_model.models.body import GeologicBody
from forward_model.models.field import MagneticField
from forward_model.models.gravity_model import GravityModel
from forward_model.models.heatflow_model import HeatFlowModel
from forward_model.models.model import ForwardModel
from forward_model.models.properties import (
    GravityProperties,
    MagneticProperties,
    ThermalProperties,
)

__all__ = [
    "ObservationModel",
    "GeologicBody",
    "MagneticField",
    "ForwardModel",
    "GravityModel",
    "HeatFlowModel",
    "MagneticProperties",
    "GravityProperties",
    "ThermalProperties",
]
