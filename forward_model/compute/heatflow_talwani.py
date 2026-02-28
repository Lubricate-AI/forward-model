"""2D Talwani-style algorithm for heat flow anomalies."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.geometry import (
    edge_geometry_2_5d,
    edge_geometry_2_75d,
    edge_geometry_2d,
)
from forward_model.models.heatflow_model import HeatFlowModel


@dataclass
class HeatFlowComponents:
    """Heat flow anomaly components at each observation point.

    Attributes:
        heat_flow: Vertical heat flow perturbation (mW/m²).
        heat_flow_x: Horizontal heat flow perturbation (mW/m²).
        heat_flow_gradient: Horizontal gradient of vertical heat flow (mW/m³).
    """

    heat_flow: NDArray[np.float64]
    heat_flow_x: NDArray[np.float64]
    heat_flow_gradient: NDArray[np.float64]
