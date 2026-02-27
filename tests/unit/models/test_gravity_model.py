"""Tests for GravityModel container."""

from typing import TypedDict

import numpy as np
import pytest
from pydantic import ValidationError

from forward_model.models import GeologicBody, GravityModel


class ObservationData(TypedDict):
    observation_x: list[float]
    observation_z: float


class TestGravityModel:
    """Tests for GravityModel container."""

    _VERTS = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

    def _body(self, **kwargs: object) -> GeologicBody:
        return GeologicBody(
            vertices=self._VERTS,
            density=2670.0,
            name="Body",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_valid_model_with_density_body(self) -> None:
        """GravityModel accepts a body with only density set."""
        body = self._body()
        model = GravityModel(
            bodies=[body],
            observation_x=[0.0, 10.0, 20.0],
            observation_z=0.0,
        )
        assert len(model.bodies) == 1
        assert model.bodies[0].density == 2670.0
        assert len(model.observation_x) == 3
        assert model.observation_z == 0.0

    def test_valid_model_with_susceptibility_body(
        self, simple_rectangle: GeologicBody
    ) -> None:
        """GravityModel accepts a body that has susceptibility (but no density)."""
        model = GravityModel(
            bodies=[simple_rectangle],
            observation_x=[0.0, 10.0],
            observation_z=0.0,
        )
        assert len(model.bodies) == 1

    def test_no_field_attribute(self) -> None:
        """GravityModel has no 'field' attribute (no MagneticField required)."""
        model = GravityModel(
            bodies=[self._body()],
            observation_x=[0.0],
            observation_z=0.0,
        )
        assert not hasattr(model, "field")

    def test_empty_bodies_list_rejected(self) -> None:
        """Empty bodies list raises ValidationError."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            GravityModel(
                bodies=[],
                observation_x=[0.0, 10.0],
                observation_z=0.0,
            )

    def test_non_finite_observation_x_rejected(self) -> None:
        """Non-finite observation_x raises ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            GravityModel(
                bodies=[self._body()],
                observation_x=[0.0, float("inf"), 10.0],
                observation_z=0.0,
            )

    def test_non_finite_observation_z_rejected(self) -> None:
        """Non-finite observation_z raises ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            GravityModel(
                bodies=[self._body()],
                observation_x=[0.0, 10.0],
                observation_z=float("nan"),
            )

    def test_get_observation_points(self) -> None:
        """get_observation_points returns correct Nx2 NumPy array."""
        model = GravityModel(
            bodies=[self._body()],
            observation_x=[-100.0, 0.0, 100.0],
            observation_z=-5.0,
        )
        points = model.get_observation_points()
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 2)
        assert points.dtype == np.float64
        assert np.allclose(points[:, 0], [-100.0, 0.0, 100.0])
        assert np.allclose(points[:, 1], -5.0)

    def test_immutability(self) -> None:
        """GravityModel is immutable (frozen=True)."""
        model = GravityModel(
            bodies=[self._body()],
            observation_x=[0.0],
            observation_z=0.0,
        )
        with pytest.raises(ValidationError):
            model.observation_z = 10.0  # type: ignore

    def test_multiple_bodies(self) -> None:
        """GravityModel accepts multiple bodies."""
        body1 = GeologicBody(
            vertices=self._VERTS,
            density=2670.0,
            name="Body1",
        )
        body2 = GeologicBody(
            vertices=[[5.0, 0.0], [6.0, 0.0], [6.0, 1.0]],
            density=-200.0,
            name="Body2",
        )
        model = GravityModel(
            bodies=[body1, body2],
            observation_x=[0.0, 5.0],
            observation_z=0.0,
        )
        assert len(model.bodies) == 2
        assert model.bodies[0].name == "Body1"
        assert model.bodies[1].name == "Body2"


class TestGravityModelOverlapValidation:
    """Overlap validation is inherited from ObservationModel."""

    _OBS: ObservationData = {"observation_x": [0.0, 50.0], "observation_z": 0.0}

    def _body(self, vertices: list[list[float]], name: str) -> GeologicBody:
        return GeologicBody(vertices=vertices, density=2670.0, name=name)

    def test_overlapping_bodies_rejected(self) -> None:
        """Partially overlapping bodies raise ValidationError."""
        a = self._body([[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]], "A")
        b = self._body([[25.0, 50.0], [75.0, 50.0], [75.0, 150.0], [25.0, 150.0]], "B")
        with pytest.raises(ValidationError, match="overlap"):
            GravityModel(bodies=[a, b], **self._OBS)

    def test_adjacent_shared_edge_accepted(self) -> None:
        """Bodies sharing only an edge are accepted."""
        a = self._body([[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]], "A")
        b = self._body(
            [[50.0, 100.0], [100.0, 100.0], [100.0, 200.0], [50.0, 200.0]], "B"
        )
        model = GravityModel(bodies=[a, b], **self._OBS)
        assert len(model.bodies) == 2

    def test_corner_touch_accepted(self) -> None:
        """Bodies touching only at a corner are accepted."""
        a = self._body([[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]], "A")
        b = self._body(
            [[50.0, 200.0], [100.0, 200.0], [100.0, 300.0], [50.0, 300.0]], "B"
        )
        model = GravityModel(bodies=[a, b], **self._OBS)
        assert len(model.bodies) == 2
