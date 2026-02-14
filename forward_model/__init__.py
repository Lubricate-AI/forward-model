"""Forward magnetic modeling package.

A Python package for conducting Talwani-style 2D forward magnetic models.

Example:
    >>> from forward_model import load_model, calculate_anomaly, plot_combined
    >>> model = load_model("model.json")
    >>> anomaly = calculate_anomaly(model)
    >>> plot_combined(model, anomaly, save_path="output.png")
"""

# Data models
# Computation
from forward_model.compute import calculate_anomaly

# I/O
from forward_model.io import load_model, write_csv, write_json
from forward_model.models import ForwardModel, GeologicBody, MagneticField

# Visualization
from forward_model.viz import plot_anomaly, plot_combined, plot_model

__all__ = [
    # Models
    "GeologicBody",
    "MagneticField",
    "ForwardModel",
    # Computation
    "calculate_anomaly",
    # I/O
    "load_model",
    "write_csv",
    "write_json",
    # Visualization
    "plot_model",
    "plot_anomaly",
    "plot_combined",
]

__version__ = "0.1.0"
