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


class TestCalculateHeatFlow:
    """Tests for calculate_heat_flow() dispatcher."""

    def _make_model(self) -> HeatFlowModel:
        return HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[
                        [-25.0, 100.0],
                        [25.0, 100.0],
                        [25.0, 200.0],
                        [-25.0, 200.0],
                    ],
                    thermal=ThermalProperties(conductivity=1.0, heat_generation=0.0),
                    name="Block",
                )
            ],
            observation_x=list(np.linspace(-100.0, 100.0, 21)),
            observation_z=0.0,
        )

    def test_returns_heat_flow_components(self) -> None:
        """calculate_heat_flow returns a HeatFlowComponents instance."""
        from forward_model.compute.heatflow_talwani import (
            HeatFlowComponents,
            calculate_heat_flow,
        )

        result = calculate_heat_flow(self._make_model())
        assert isinstance(result, HeatFlowComponents)

    def test_shape_matches_observation_points(self) -> None:
        """Output arrays have the same length as observation_x."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        result = calculate_heat_flow(self._make_model())
        assert result.heat_flow.shape == (21,)
        assert result.heat_flow_x.shape == (21,)
        assert result.heat_flow_gradient.shape == (21,)

    def test_all_finite(self) -> None:
        """All output values are finite."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        result = calculate_heat_flow(self._make_model())
        assert np.all(np.isfinite(result.heat_flow))
        assert np.all(np.isfinite(result.heat_flow_x))
        assert np.all(np.isfinite(result.heat_flow_gradient))

    def test_parallel_matches_serial(self) -> None:
        """parallel=True produces the same result as serial."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        model = self._make_model()
        serial = calculate_heat_flow(model, parallel=False)
        parallel = calculate_heat_flow(model, parallel=True)
        np.testing.assert_allclose(serial.heat_flow, parallel.heat_flow, rtol=1e-10)
        np.testing.assert_allclose(
            serial.heat_flow_gradient, parallel.heat_flow_gradient, rtol=1e-10
        )

    def test_missing_thermal_properties_raises(self) -> None:
        """ValueError is raised when a body has no thermal properties."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow
        from forward_model.models import GravityProperties

        model = HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
                    gravity=GravityProperties(density_contrast=300.0),
                    name="NoThermal",
                )
            ],
            observation_x=[0.0],
            observation_z=0.0,
        )
        with pytest.raises(ValueError, match="NoThermal"):
            calculate_heat_flow(model)

    def test_superposition_two_bodies(self) -> None:
        """Two bodies run together equals sum of individual runs."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        verts1 = [[-50.0, 100.0], [0.0, 100.0], [0.0, 200.0], [-50.0, 200.0]]
        verts2 = [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]]
        obs_x = list(np.linspace(-100.0, 100.0, 21))
        thermal = ThermalProperties(conductivity=1.0)

        model1 = HeatFlowModel(
            bodies=[GeologicBody(vertices=verts1, thermal=thermal, name="B1")],
            observation_x=obs_x,
            observation_z=0.0,
        )
        model2 = HeatFlowModel(
            bodies=[GeologicBody(vertices=verts2, thermal=thermal, name="B2")],
            observation_x=obs_x,
            observation_z=0.0,
        )
        model_both = HeatFlowModel(
            bodies=[
                GeologicBody(vertices=verts1, thermal=thermal, name="B1"),
                GeologicBody(vertices=verts2, thermal=thermal, name="B2"),
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )
        r1 = calculate_heat_flow(model1)
        r2 = calculate_heat_flow(model2)
        r_both = calculate_heat_flow(model_both)
        np.testing.assert_allclose(
            r_both.heat_flow, r1.heat_flow + r2.heat_flow, rtol=1e-10
        )

    def test_sanity_check_order_of_magnitude(self) -> None:
        """Heat flow perturbation is in a physically plausible mW/m² range.

        A 50m-wide body (Δk=1.0 W/m·K) at 100m depth in a 65 mW/m² background
        should produce a surface perturbation of order 0.01–100 mW/m².
        """
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        model = HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[
                        [-25.0, 100.0],
                        [25.0, 100.0],
                        [25.0, 200.0],
                        [-25.0, 200.0],
                    ],
                    thermal=ThermalProperties(conductivity=1.0),
                    name="Granite",
                )
            ],
            observation_x=[0.0],
            observation_z=0.0,
            background_heat_flow=65.0,
        )
        result = calculate_heat_flow(model)
        peak = float(np.abs(result.heat_flow[0]))
        assert 0.01 < peak < 100.0, (
            f"Heat flow perturbation {peak:.4f} mW/m² outside expected range"
        )

    def test_radiogenic_contribution_increases_heat_flow(self) -> None:
        """A body with heat_generation > 0 produces a larger heat flow signal."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        model_no_rad = HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[
                        [-25.0, 100.0],
                        [25.0, 100.0],
                        [25.0, 200.0],
                        [-25.0, 200.0],
                    ],
                    thermal=ThermalProperties(conductivity=1.0, heat_generation=0.0),
                    name="NoRad",
                )
            ],
            observation_x=[0.0],
            observation_z=0.0,
        )
        model_rad = HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[
                        [-25.0, 100.0],
                        [25.0, 100.0],
                        [25.0, 200.0],
                        [-25.0, 200.0],
                    ],
                    thermal=ThermalProperties(conductivity=1.0, heat_generation=2.5),
                    name="WithRad",
                )
            ],
            observation_x=[0.0],
            observation_z=0.0,
        )
        r_no_rad = calculate_heat_flow(model_no_rad)
        r_rad = calculate_heat_flow(model_rad)
        assert r_rad.heat_flow[0] > r_no_rad.heat_flow[0], (
            "Radiogenic heat generation should increase surface heat flow"
        )

    def test_single_observation_point_gradient_is_zero(self) -> None:
        """With one observation point, gradient is undefined and returned as zero."""
        from forward_model.compute.heatflow_talwani import calculate_heat_flow

        model = HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
                    thermal=ThermalProperties(conductivity=1.0),
                    name="Block",
                )
            ],
            observation_x=[25.0],
            observation_z=0.0,
        )
        result = calculate_heat_flow(model)
        np.testing.assert_allclose(result.heat_flow_gradient, 0.0, atol=1e-12)
