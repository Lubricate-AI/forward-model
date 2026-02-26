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


class TestForwardModelOverlapValidation:
    """Tests for the body overlap model_validator on ForwardModel."""

    def _make_model(
        self,
        bodies: list[GeologicBody],
        earth_field: MagneticField,
    ) -> ForwardModel:
        return ForwardModel(
            bodies=bodies,
            field=earth_field,
            observation_x=[0.0, 50.0],
            observation_z=0.0,
        )

    def test_partial_overlap_raises(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Partially overlapping rectangles raise ValidationError."""
        body_b = GeologicBody(
            vertices=[[25.0, 50.0], [75.0, 50.0], [75.0, 150.0], [25.0, 150.0]],
            susceptibility=0.05,
            name="PartialOverlap",
        )
        with pytest.raises(ValidationError, match="overlap"):
            self._make_model([simple_rectangle, body_b], earth_field)

    def test_full_containment_raises(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """A body fully containing another raises ValidationError."""
        outer = GeologicBody(
            vertices=[[-50.0, 50.0], [150.0, 50.0], [150.0, 300.0], [-50.0, 300.0]],
            susceptibility=0.02,
            name="Outer",
        )
        with pytest.raises(ValidationError, match="overlap"):
            self._make_model([simple_rectangle, outer], earth_field)

    def test_adjacent_shared_edge_does_not_raise(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Bodies sharing only an edge are accepted (valid geological contact)."""
        adjacent = GeologicBody(
            vertices=[[50.0, 100.0], [100.0, 100.0], [100.0, 200.0], [50.0, 200.0]],
            susceptibility=0.03,
            name="Adjacent",
        )
        model = self._make_model([simple_rectangle, adjacent], earth_field)
        assert len(model.bodies) == 2

    def test_corner_touch_does_not_raise(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Bodies touching only at a single corner are accepted."""
        corner = GeologicBody(
            vertices=[[50.0, 200.0], [100.0, 200.0], [100.0, 300.0], [50.0, 300.0]],
            susceptibility=0.04,
            name="CornerTouch",
        )
        model = self._make_model([simple_rectangle, corner], earth_field)
        assert len(model.bodies) == 2

    def test_coincident_polygons_raise(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Two bodies with exactly the same footprint raise ValidationError."""
        coincident = GeologicBody(
            vertices=list(simple_rectangle.vertices),
            susceptibility=0.07,
            name="Coincident",
        )
        with pytest.raises(ValidationError, match="overlap"):
            self._make_model([simple_rectangle, coincident], earth_field)

    def test_boundary_overlap_with_collinear_vertices_raises(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Same footprint with extra collinear vertices raises ValidationError."""
        boundary_overlap = GeologicBody(
            vertices=[
                [0.0, 100.0],
                [25.0, 100.0],
                [50.0, 100.0],
                [50.0, 150.0],
                [50.0, 200.0],
                [25.0, 200.0],
                [0.0, 200.0],
                [0.0, 150.0],
            ],
            susceptibility=0.08,
            name="BoundaryOverlap",
        )
        with pytest.raises(ValidationError, match="overlap"):
            self._make_model([simple_rectangle, boundary_overlap], earth_field)

    def test_non_overlapping_multiple_bodies_passes(
        self, simple_rectangle: GeologicBody, earth_field: MagneticField
    ) -> None:
        """Non-overlapping bodies pass validation."""
        body2 = GeologicBody(
            vertices=[[100.0, 50.0], [150.0, 50.0], [150.0, 100.0], [100.0, 100.0]],
            susceptibility=0.1,
            name="Body2",
        )
        model = self._make_model([simple_rectangle, body2], earth_field)
        assert len(model.bodies) == 2
