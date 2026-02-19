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
from forward_model.compute import batch_calculate as batch_calculate
from forward_model.compute import calculate_anomaly
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
from forward_model.models import ForwardModel, GeologicBody, MagneticField

# Visualization
from forward_model.viz import plot_anomaly, plot_combined, plot_model

__all__ = [
    # Configuration
    "Config",
    "load_config",
    # Models
    "GeologicBody",
    "MagneticField",
    "ForwardModel",
    # Computation
    "AnomalyComponents",
    "BatchResult",
    "batch_calculate",
    "calculate_anomaly",
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

__version__ = "0.1.0"
