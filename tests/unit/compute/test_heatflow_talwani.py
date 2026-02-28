"""Tests for heat flow anomaly computation (2D Talwani-style)."""

import numpy as np
import pytest

from forward_model.compute.heatflow_talwani import HeatFlowComponents
from forward_model.models import GeologicBody, HeatFlowModel, ThermalProperties


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


class TestHeatFlowKernel:
    """Tests for 2D heat flow conductive kernel."""

    def test_nonzero_output_with_conductivity_contrast(self) -> None:
        """A body with conductivity contrast produces non-zero heat flow."""
        from forward_model.compute.heatflow_talwani import _apply_heatflow_kernel

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array([[0.0, 0.0]], dtype=np.float64)
        qz = _apply_heatflow_kernel(
            vertices, obs, conductivity_contrast=1.0, background_heat_flow=65.0
        )
        assert np.any(qz != 0.0)

    def test_zero_contrast_gives_zero(self) -> None:
        """Zero conductivity contrast produces zero perturbation."""
        from forward_model.compute.heatflow_talwani import _apply_heatflow_kernel

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = np.linspace(-100.0, 100.0, 21)
        obs = np.column_stack([obs_x, np.zeros_like(obs_x)])
        qz = _apply_heatflow_kernel(
            vertices, obs, conductivity_contrast=0.0, background_heat_flow=65.0
        )
        np.testing.assert_allclose(qz, 0.0, atol=1e-12)

    def test_symmetric_body_symmetric_anomaly(self) -> None:
        """Symmetric polygon produces symmetric heat flow perturbation."""
        from forward_model.compute.heatflow_talwani import _apply_heatflow_kernel

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = [-50.0, -25.0, 0.0, 25.0, 50.0]
        obs = np.column_stack([obs_x, [0.0] * 5])
        qz = _apply_heatflow_kernel(
            vertices, obs, conductivity_contrast=1.0, background_heat_flow=65.0
        )
        np.testing.assert_allclose(qz[0], qz[4], rtol=1e-10)
        np.testing.assert_allclose(qz[1], qz[3], rtol=1e-10)

    def test_all_finite(self) -> None:
        """All output values are finite."""
        from forward_model.compute.heatflow_talwani import _apply_heatflow_kernel

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = np.linspace(-200.0, 200.0, 41)
        obs = np.column_stack([obs_x, np.zeros_like(obs_x)])
        qz = _apply_heatflow_kernel(
            vertices, obs, conductivity_contrast=1.0, background_heat_flow=65.0
        )
        assert np.all(np.isfinite(qz))

    def test_output_shape_matches_observation_points(self) -> None:
        """Output shape matches number of observation points."""
        from forward_model.compute.heatflow_talwani import _apply_heatflow_kernel

        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 11))
        obs = np.column_stack([obs_x, [0.0] * 11])
        qz = _apply_heatflow_kernel(
            vertices, obs, conductivity_contrast=1.0, background_heat_flow=65.0
        )
        assert qz.shape == (11,)


class TestHeatFlowStrikeKernels:
    """Tests for 2.5D and 2.75D heat flow kernels."""

    def test_2_5d_attenuates_vs_2d(self) -> None:
        """Finite strike (2.5D) produces smaller amplitude than infinite (2D)."""
        from forward_model.compute.heatflow_talwani import (
            _apply_heatflow_kernel,
            _apply_heatflow_kernel_2_5d,
        )

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = np.linspace(-100.0, 100.0, 21)
        obs = np.column_stack([obs_x, np.zeros_like(obs_x)])
        qz_2d = _apply_heatflow_kernel(vertices, obs, 1.0, 65.0)
        qz_25d = _apply_heatflow_kernel_2_5d(
            vertices, obs, 1.0, 65.0, strike_half_length=150.0
        )
        assert np.max(np.abs(qz_25d)) < np.max(np.abs(qz_2d))

    def test_2_75d_produces_finite_values(self) -> None:
        """Asymmetric 2.75D strike produces finite heat flow."""
        from forward_model.compute.heatflow_talwani import _apply_heatflow_kernel_2_75d

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0]], dtype=np.float64)
        qz = _apply_heatflow_kernel_2_75d(vertices, obs, 1.0, 65.0, 500.0, 200.0)
        assert np.all(np.isfinite(qz))

    def test_large_strike_approaches_2d(self) -> None:
        """Very large strike half-length should approach 2D result."""
        from forward_model.compute.heatflow_talwani import (
            _apply_heatflow_kernel,
            _apply_heatflow_kernel_2_5d,
        )

        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array([[0.0, 0.0]], dtype=np.float64)
        qz_2d = _apply_heatflow_kernel(vertices, obs, 1.0, 65.0)
        qz_large_strike = _apply_heatflow_kernel_2_5d(
            vertices, obs, 1.0, 65.0, strike_half_length=1e7
        )
        np.testing.assert_allclose(qz_large_strike, qz_2d, rtol=1e-3)
