"""Tests for heat flow anomaly computation (2D Talwani-style)."""

import numpy as np
import pytest

from forward_model.compute.heatflow_talwani import HeatFlowComponents


class TestHeatFlowComponents:
    """Test HeatFlowComponents dataclass."""

    def test_fields_exist(self) -> None:
        """Test that HeatFlowComponents has expected fields."""
        qz = np.array([1.0, 2.0], dtype=np.float64)
        qx = np.array([0.1, 0.2], dtype=np.float64)
        grad = np.array([0.01, 0.02], dtype=np.float64)
        comp = HeatFlowComponents(heat_flow=qz, heat_flow_x=qx, heat_flow_gradient=grad)
        assert comp.heat_flow is qz
        assert comp.heat_flow_x is qx
        assert comp.heat_flow_gradient is grad

    def test_dtype_is_float64(self) -> None:
        """Test that HeatFlowComponents uses float64."""
        qz = np.array([1.0], dtype=np.float64)
        qx = np.array([0.0], dtype=np.float64)
        grad = np.array([0.0], dtype=np.float64)
        comp = HeatFlowComponents(heat_flow=qz, heat_flow_x=qx, heat_flow_gradient=grad)
        assert comp.heat_flow.dtype == np.float64
        assert comp.heat_flow_x.dtype == np.float64
        assert comp.heat_flow_gradient.dtype == np.float64
