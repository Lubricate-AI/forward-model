"""Tests for ForwardModel container."""

import numpy as np
import pytest
from pydantic import ValidationError

from forward_model.models import ForwardModel, GeologicBody, MagneticField


class TestForwardModel:
    """Tests for ForwardModel container."""

    def test_valid_model(self, simple_model: ForwardModel) -> None:
        """Test creating a valid forward model."""
        assert len(simple_model.bodies) == 1
        assert simple_model.field.intensity == 50000.0
        assert len(simple_model.observation_x) == 7
        assert simple_model.observation_z == 0.0

    def test_empty_bodies_list(self, earth_field: MagneticField) -> None:
        """Test that empty bodies list is rejected."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            ForwardModel(
                bodies=[],
                field=earth_field,
                observation_x=[0.0, 10.0],
                observation_z=0.0,
            )

    def test_non_finite_observation_x(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test that non-finite observation x coordinates are rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            ForwardModel(
                bodies=[simple_rectangle],
                field=earth_field,
                observation_x=[0.0, float("inf"), 10.0],
                observation_z=0.0,
            )

    def test_non_finite_observation_z(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test that non-finite observation z is rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            ForwardModel(
                bodies=[simple_rectangle],
                field=earth_field,
                observation_x=[0.0, 10.0],
                observation_z=float("nan"),
            )

    def test_get_observation_points(self, simple_model: ForwardModel) -> None:
        """Test getting observation points as NumPy array."""
        points = simple_model.get_observation_points()
        assert isinstance(points, np.ndarray)
        assert points.shape == (7, 2)
        assert points.dtype == np.float64
        assert np.allclose(points[:, 0], [-100.0, -50.0, 0.0, 25.0, 50.0, 100.0, 150.0])
        assert np.allclose(points[:, 1], 0.0)

    def test_immutability(self, simple_model: ForwardModel) -> None:
        """Test that ForwardModel is immutable."""
        with pytest.raises(ValidationError):
            simple_model.observation_z = 10.0  # type: ignore

    def test_multiple_bodies(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Test model with multiple bodies."""
        body2 = GeologicBody(
            vertices=[[100.0, 50.0], [150.0, 50.0], [150.0, 100.0], [100.0, 100.0]],
            susceptibility=0.1,
            name="Body2",
        )
        model = ForwardModel(
            bodies=[simple_rectangle, body2],
            field=earth_field,
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )
        assert len(model.bodies) == 2
        assert model.bodies[0].name == "Rectangle"
        assert model.bodies[1].name == "Body2"
