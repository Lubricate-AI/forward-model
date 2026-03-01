"""Tests for high-level anomaly calculator."""

import numpy as np
import pytest

from forward_model.compute import (
    GravityComponents,
    MagneticComponents,
    calculate_anomaly,
)
from forward_model.models import (
    ForwardModel,
    GeologicBody,
    GravityModel,
    GravityProperties,
    HeatFlowModel,
    MagneticField,
    MagneticProperties,
)


class TestCalculateAnomaly:
    """Tests for high-level calculate_anomaly function."""

    def test_simple_model(self, simple_model: ForwardModel) -> None:
        """Test calculation with simple forward model."""
        anomaly = calculate_anomaly(simple_model)

        # Check basic properties
        assert anomaly.shape == (7,)  # 7 observation points
        assert np.all(np.isfinite(anomaly))

    def test_zero_susceptibility(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test that zero susceptibility produces zero anomaly."""
        # Create body with zero susceptibility
        zero_body = GeologicBody(
            vertices=simple_rectangle.vertices,
            magnetic=MagneticProperties(susceptibility=0.0),
            name="Zero",
        )
        model = ForwardModel(
            bodies=[zero_body],
            field=earth_field,
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )

        anomaly = calculate_anomaly(model)
        assert np.allclose(anomaly, 0.0)

    def test_superposition(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test superposition principle: two bodies = sum of individual anomalies."""
        # Create second body (shifted)
        body2 = GeologicBody(
            vertices=[
                [100.0, 100.0],
                [150.0, 100.0],
                [150.0, 200.0],
                [100.0, 200.0],
            ],
            magnetic=MagneticProperties(susceptibility=0.05),
            name="Body2",
        )

        obs_x = [0.0, 50.0, 100.0, 150.0]

        # Model with body 1 only
        model1 = ForwardModel(
            bodies=[simple_rectangle],
            field=earth_field,
            observation_x=obs_x,
            observation_z=0.0,
        )
        anomaly1 = calculate_anomaly(model1)

        # Model with body 2 only
        model2 = ForwardModel(
            bodies=[body2],
            field=earth_field,
            observation_x=obs_x,
            observation_z=0.0,
        )
        anomaly2 = calculate_anomaly(model2)

        # Model with both bodies
        model_both = ForwardModel(
            bodies=[simple_rectangle, body2],
            field=earth_field,
            observation_x=obs_x,
            observation_z=0.0,
        )
        anomaly_both = calculate_anomaly(model_both)

        # Check superposition
        assert np.allclose(anomaly_both, anomaly1 + anomaly2, rtol=1e-10)

    def test_parallel_matches_serial(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test that parallel=True produces the same result as parallel=False."""
        body2 = GeologicBody(
            vertices=[
                [100.0, 100.0],
                [150.0, 100.0],
                [150.0, 200.0],
                [100.0, 200.0],
            ],
            magnetic=MagneticProperties(susceptibility=0.05),
            name="Body2",
        )
        model = ForwardModel(
            bodies=[simple_rectangle, body2],
            field=earth_field,
            observation_x=np.linspace(-100.0, 250.0, 50).tolist(),
            observation_z=0.0,
        )

        serial = calculate_anomaly(model, parallel=False)
        parallel = calculate_anomaly(model, parallel=True)
        assert np.allclose(serial, parallel, rtol=1e-12)

    def test_multiple_observation_points(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test with varying number of observation points."""
        for n_points in [1, 5, 10, 50]:
            obs_x = np.linspace(-100, 200, n_points).tolist()
            model = ForwardModel(
                bodies=[simple_rectangle],
                field=earth_field,
                observation_x=obs_x,
                observation_z=0.0,
            )
            anomaly = calculate_anomaly(model)
            assert anomaly.shape == (n_points,)
            assert np.all(np.isfinite(anomaly))

    def test_calculate_anomaly_all_returns_magnetic_components(
        self, simple_model: ForwardModel
    ) -> None:
        """calculate_anomaly with component='all' returns MagneticComponents."""
        result = calculate_anomaly(simple_model, component="all")
        assert isinstance(result, MagneticComponents)

    def test_no_magnetic_properties_raises(self, earth_field: MagneticField) -> None:
        """ValueError is raised when a body has no magnetic properties set."""
        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            gravity=GravityProperties(density_contrast=2670.0),
            name="DensityOnly",
        )
        model = ForwardModel(
            bodies=[body],
            field=earth_field,
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )
        with pytest.raises(ValueError, match="DensityOnly"):
            calculate_anomaly(model)


class TestCalculateAnomalyDispatch:
    """Tests for type-dispatched calculate_anomaly."""

    def test_gravity_model_returns_gravity_components(
        self, gravity_model: GravityModel
    ) -> None:
        """calculate_anomaly(gravity_model) returns GravityComponents."""
        result = calculate_anomaly(gravity_model)
        assert isinstance(result, GravityComponents)
        assert result.gz.shape == (7,)
        assert np.all(np.isfinite(result.gz))

    def test_gravity_model_parallel(self, gravity_model: GravityModel) -> None:
        """calculate_anomaly(gravity_model, parallel=True) matches serial result."""
        serial = calculate_anomaly(gravity_model, parallel=False)
        parallel = calculate_anomaly(gravity_model, parallel=True)
        assert np.allclose(serial.gz, parallel.gz, rtol=1e-12)

    def test_gravity_model_correct_units(self, gravity_model: GravityModel) -> None:
        """Gravity result is in mGal (reasonable range for crustal anomalies)."""
        result = calculate_anomaly(gravity_model)
        assert np.all(np.abs(result.gz) < 1000.0)

    def test_heat_flow_model_returns_heat_flow_components(
        self, heat_flow_model: HeatFlowModel
    ) -> None:
        """calculate_anomaly(heat_flow_model) returns HeatFlowComponents."""
        from forward_model.compute.heatflow_talwani import HeatFlowComponents

        result = calculate_anomaly(heat_flow_model)
        assert isinstance(result, HeatFlowComponents)
        assert result.heat_flow.shape == (7,)
        assert np.all(np.isfinite(result.heat_flow))

    def test_heat_flow_model_parallel(self, heat_flow_model: HeatFlowModel) -> None:
        """calculate_anomaly(heat_flow_model, parallel=True) matches serial."""
        serial = calculate_anomaly(heat_flow_model, parallel=False)
        parallel = calculate_anomaly(heat_flow_model, parallel=True)
        np.testing.assert_allclose(serial.heat_flow, parallel.heat_flow, rtol=1e-10)

    def test_forward_model_unchanged(self, simple_model: ForwardModel) -> None:
        """Backward compat: ForwardModel still returns NDArray by default."""
        result = calculate_anomaly(simple_model)
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)

    def test_gravity_model_ignores_component(self, gravity_model: GravityModel) -> None:
        """component keyword is silently ignored for GravityModel."""
        result = calculate_anomaly(gravity_model, component="all")
        assert isinstance(result, GravityComponents)

    def test_gravity_model_returns_gz_gradient(
        self, gravity_model: GravityModel
    ) -> None:
        """GravityComponents includes gz_gradient with same shape as gz."""
        result = calculate_anomaly(gravity_model)
        assert result.gz_gradient.shape == result.gz.shape
        assert np.all(np.isfinite(result.gz_gradient))


def test_magnetic_components_exported_from_compute() -> None:
    """MagneticComponents is importable from forward_model.compute."""
    from forward_model.compute import MagneticComponents as MC

    assert MC is MagneticComponents


def test_top_level_exports_magnetic_and_gravity_components() -> None:
    """MagneticComponents and GravityComponents are importable from forward_model."""
    from forward_model import GravityComponents as GC
    from forward_model import MagneticComponents as MC

    assert MC is MagneticComponents
    assert GC is GravityComponents


def test_heat_flow_components_exported_from_compute() -> None:
    """HeatFlowComponents is importable from forward_model.compute."""
    from forward_model.compute import HeatFlowComponents

    assert HeatFlowComponents is not None


def test_top_level_exports_heat_flow_components() -> None:
    """HeatFlowComponents is importable from forward_model."""
    from forward_model import HeatFlowComponents

    assert HeatFlowComponents is not None
