"""Tests for magnetic anomaly computation."""

import numpy as np

from forward_model.compute import (
    calculate_anomaly,
    compute_polygon_anomaly,
    field_to_magnetization,
)
from forward_model.models import ForwardModel, GeologicBody, MagneticField


class TestFieldToMagnetization:
    """Tests for field to magnetization conversion."""

    def test_vertical_field(self) -> None:
        """Test with vertical inducing field (inclination = 90)."""
        mag = field_to_magnetization(
            susceptibility=0.1,
            field_intensity=50000.0,  # nT
            field_inclination=90.0,
            field_declination=0.0,
        )
        assert mag.shape == (2,)
        # Vertical field: Mx should be ~0, Mz should be negative (down)
        assert np.abs(mag[0]) < 1e-10  # Mx ~ 0
        assert mag[1] < 0  # Mz negative (field points down)

    def test_horizontal_field(self) -> None:
        """Test with horizontal inducing field (inclination = 0)."""
        mag = field_to_magnetization(
            susceptibility=0.1,
            field_intensity=50000.0,
            field_inclination=0.0,
            field_declination=0.0,
        )
        assert mag.shape == (2,)
        # Horizontal field: Mx non-zero, Mz should be ~0
        assert mag[0] > 0  # Mx positive
        assert np.abs(mag[1]) < 1e-10  # Mz ~ 0

    def test_zero_susceptibility(self) -> None:
        """Test that zero susceptibility gives zero magnetization."""
        mag = field_to_magnetization(
            susceptibility=0.0,
            field_intensity=50000.0,
            field_inclination=60.0,
            field_declination=0.0,
        )
        assert np.allclose(mag, [0.0, 0.0])


class TestComputePolygonAnomaly:
    """Tests for Talwani polygon anomaly computation."""

    def test_simple_rectangle(self) -> None:
        """Test anomaly from a simple rectangular body."""
        # Rectangle centered at x=25, depth 100-200m
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )

        # Observation points along surface
        obs = np.array(
            [[-100.0, 0.0], [0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            dtype=np.float64,
        )

        # Vertical magnetization (simple case)
        mag = np.array([0.0, -1000.0], dtype=np.float64)  # A/m

        anomaly = compute_polygon_anomaly(vertices, obs, mag)

        # Basic checks
        assert anomaly.shape == (5,)
        assert np.all(np.isfinite(anomaly))

        # Anomaly should be largest near center of body
        center_idx = 2  # x=25, directly above center
        # Greater than far field
        assert np.abs(anomaly[center_idx]) > np.abs(anomaly[0])

    def test_symmetry(self) -> None:
        """Test that symmetric body produces symmetric anomaly."""
        # Rectangle centered at x=0
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )

        # Symmetric observation points
        obs = np.array(
            [[-100.0, 0.0], [-50.0, 0.0], [0.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            dtype=np.float64,
        )

        # Vertical magnetization
        mag = np.array([0.0, -1000.0], dtype=np.float64)

        anomaly = compute_polygon_anomaly(vertices, obs, mag)

        # Check symmetry: anomaly[-100] ≈ anomaly[100], etc.
        assert np.allclose(anomaly[0], anomaly[4], rtol=1e-10)  # x=-100 vs x=100
        assert np.allclose(anomaly[1], anomaly[3], rtol=1e-10)  # x=-50 vs x=50

    def test_zero_magnetization(self) -> None:
        """Test that zero magnetization gives zero anomaly."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float64)
        mag = np.array([0.0, 0.0], dtype=np.float64)

        anomaly = compute_polygon_anomaly(vertices, obs, mag)
        assert np.allclose(anomaly, 0.0)

    def test_far_field_decay(self) -> None:
        """Test that anomaly decreases with distance."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )

        # Observation points at increasing distances
        obs = np.array(
            [[25.0, 0.0], [125.0, 0.0], [500.0, 0.0], [1000.0, 0.0]],
            dtype=np.float64,
        )
        mag = np.array([0.0, -1000.0], dtype=np.float64)

        anomaly = compute_polygon_anomaly(vertices, obs, mag)

        # Anomaly should decay with distance
        assert np.abs(anomaly[0]) > np.abs(anomaly[1])
        assert np.abs(anomaly[1]) > np.abs(anomaly[2])
        assert np.abs(anomaly[2]) > np.abs(anomaly[3])

    def test_degenerate_edge(self) -> None:
        """Test handling of degenerate edges (same vertex repeated)."""
        # Include a duplicate vertex
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array([[25.0, 0.0]], dtype=np.float64)
        mag = np.array([0.0, -1000.0], dtype=np.float64)

        # Should not raise an error
        anomaly = compute_polygon_anomaly(vertices, obs, mag)
        assert np.isfinite(anomaly[0])


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
            susceptibility=0.0,
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
            susceptibility=0.05,
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
