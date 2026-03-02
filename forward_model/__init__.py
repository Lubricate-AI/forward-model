"""Forward magnetic modeling package.

A Python package for conducting Talwani-style 2D forward magnetic models.

Example:
    >>> from forward_model import load_model, calculate_anomaly, plot_combined
    >>> model = load_model("model.json")
    >>> anomaly = calculate_anomaly(model)
    >>> plot_combined(model, anomaly, save_path="output.png")
"""

# Configuration
# Computation
from forward_model.compute import AnomalyComponents as AnomalyComponents
from forward_model.compute import BatchResult as BatchResult
from forward_model.compute import GravityComponents as GravityComponents
from forward_model.compute import HeatFlowComponents as HeatFlowComponents
from forward_model.compute import MagneticComponents as MagneticComponents
from forward_model.compute import batch_calculate as batch_calculate
from forward_model.compute import calculate_anomaly as calculate_anomaly
from forward_model.config import Config as Config
from forward_model.config import load_config as load_config

# I/O
from forward_model.io import (
    load_model,
    load_model_from_csv,
    write_csv,
    write_json,
    write_numpy,
)
from forward_model.models import (
    AnyForwardModel,
    MagneticModel,
    GeologicBody,
    GravityModel,
    GravityProperties,
    HeatFlowModel,
    MagneticField,
    MagneticProperties,
    ThermalProperties,
)

# Visualization
from forward_model.viz import plot_anomaly, plot_combined, plot_model

__all__ = [
    # Configuration
    "Config",
    "load_config",
    # Models
    "GeologicBody",
    "MagneticField",
    "MagneticModel",
    "GravityModel",
    "HeatFlowModel",
    "AnyForwardModel",
    "MagneticProperties",
    "GravityProperties",
    "ThermalProperties",
    # Computation
    "AnomalyComponents",
    "BatchResult",
    "batch_calculate",
    "calculate_anomaly",
    "GravityComponents",
    "HeatFlowComponents",
    "MagneticComponents",
    # I/O
    "load_model",
    "load_model_from_csv",
    "write_csv",
    "write_json",
    "write_numpy",
    # Visualization
    "plot_model",
    "plot_anomaly",
    "plot_combined",
]

__version__ = "2.8.1"
