"""Tests for data models."""

import numpy as np
import pytest
from pydantic import ValidationError

from forward_model.models import ForwardModel, GeologicBody, MagneticField


class TestGeologicBody:
    """Tests for GeologicBody model."""

    def test_valid_body(self, simple_rectangle: GeologicBody) -> None:
        """Test creating a valid geologic body."""
        assert len(simple_rectangle.vertices) == 4
        assert simple_rectangle.susceptibility == 0.05
        assert simple_rectangle.name == "Rectangle"

    def test_minimum_vertices(self) -> None:
        """Test that at least 3 vertices are required."""
        with pytest.raises(ValidationError, match="at least 3 items"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0]],
                susceptibility=0.01,
                name="Invalid",
            )

    def test_vertex_dimension_validation(self) -> None:
        """Test that vertices must be 2D coordinates."""
        with pytest.raises(ValidationError, match="exactly 2 coordinates"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0, 2.0]],
                susceptibility=0.01,
                name="Invalid",
            )

    def test_non_finite_vertices(self) -> None:
        """Test that non-finite vertex coordinates are rejected."""
        with pytest.raises(ValidationError, match="non-finite values"):
            GeologicBody(
                vertices=[[0.0, 0.0], [float("inf"), 0.0], [1.0, 1.0]],
                susceptibility=0.01,
                name="Invalid",
            )

    def test_non_finite_susceptibility(self) -> None:
        """Test that non-finite susceptibility is rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                susceptibility=float("nan"),
                name="Invalid",
            )

    def test_to_numpy(self, simple_rectangle: GeologicBody) -> None:
        """Test conversion to NumPy array."""
        arr = simple_rectangle.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 2)
        assert arr.dtype == np.float64
        assert np.allclose(arr[0], [0.0, 100.0])

    def test_immutability(self, simple_rectangle: GeologicBody) -> None:
        """Test that GeologicBody is immutable."""
        with pytest.raises(ValidationError):
            simple_rectangle.susceptibility = 0.1  # type: ignore


class TestMagneticField:
    """Tests for MagneticField model."""

    def test_valid_field(self, earth_field: MagneticField) -> None:
        """Test creating a valid magnetic field."""
        assert earth_field.intensity == 50000.0
        assert earth_field.inclination == 60.0
        assert earth_field.declination == 0.0

    def test_negative_intensity(self) -> None:
        """Test that negative intensity is rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MagneticField(intensity=-1000.0, inclination=60.0, declination=0.0)

    def test_zero_intensity(self) -> None:
        """Test that zero intensity is rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MagneticField(intensity=0.0, inclination=60.0, declination=0.0)

    def test_inclination_range(self) -> None:
        """Test that inclination must be in valid range."""
        # Valid boundary values
        MagneticField(intensity=50000.0, inclination=-90.0, declination=0.0)
        MagneticField(intensity=50000.0, inclination=90.0, declination=0.0)

        # Invalid values
        with pytest.raises(ValidationError, match="greater than or equal to -90"):
            MagneticField(intensity=50000.0, inclination=-91.0, declination=0.0)
        with pytest.raises(ValidationError, match="less than or equal to 90"):
            MagneticField(intensity=50000.0, inclination=91.0, declination=0.0)

    def test_declination_range(self) -> None:
        """Test that declination must be in valid range."""
        # Valid boundary values
        MagneticField(intensity=50000.0, inclination=60.0, declination=-180.0)
        MagneticField(intensity=50000.0, inclination=60.0, declination=180.0)

        # Invalid values
        with pytest.raises(ValidationError, match="greater than or equal to -180"):
            MagneticField(intensity=50000.0, inclination=60.0, declination=-181.0)
        with pytest.raises(ValidationError, match="less than or equal to 180"):
            MagneticField(intensity=50000.0, inclination=60.0, declination=181.0)

    def test_non_finite_values(self) -> None:
        """Test that non-finite field values are rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            MagneticField(intensity=float("inf"), inclination=60.0, declination=0.0)
        # NaN fails range validation before finite check
        with pytest.raises(ValidationError):
            MagneticField(intensity=50000.0, inclination=float("nan"), declination=0.0)

    def test_immutability(self, earth_field: MagneticField) -> None:
        """Test that MagneticField is immutable."""
        with pytest.raises(ValidationError):
            earth_field.intensity = 60000.0  # type: ignore


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
