"""Tests for high-level anomaly calculator."""

import numpy as np

from forward_model.compute import calculate_anomaly
from forward_model.models import ForwardModel, GeologicBody, MagneticField


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
            susceptibility=0.05,
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
