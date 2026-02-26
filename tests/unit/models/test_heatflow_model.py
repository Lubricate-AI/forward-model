"""Tests for HeatFlowModel container."""

import numpy as np
import pytest
from pydantic import ValidationError

from forward_model.models import GeologicBody, HeatFlowModel


class TestHeatFlowModel:
    """Tests for HeatFlowModel container."""

    def test_valid_model(self, heat_flow_model: HeatFlowModel) -> None:
        """Test creating a valid heat flow model."""
        assert len(heat_flow_model.bodies) == 1
        assert len(heat_flow_model.observation_x) == 7
        assert heat_flow_model.observation_z == 0.0
        assert heat_flow_model.background_heat_flow == 65.0

    def test_empty_bodies_list_rejected(self) -> None:
        """Test that empty bodies list is rejected."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            HeatFlowModel(
                bodies=[],
                observation_x=[0.0, 10.0],
                observation_z=0.0,
            )

    def test_non_finite_observation_x_rejected(
        self, thermal_body: GeologicBody
    ) -> None:
        """Test that non-finite observation x coordinates are rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            HeatFlowModel(
                bodies=[thermal_body],
                observation_x=[0.0, float("inf"), 10.0],
                observation_z=0.0,
            )

    def test_non_finite_observation_z_rejected(
        self, thermal_body: GeologicBody
    ) -> None:
        """Test that non-finite observation z is rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            HeatFlowModel(
                bodies=[thermal_body],
                observation_x=[0.0, 10.0],
                observation_z=float("nan"),
            )

    def test_background_heat_flow_default(self, thermal_body: GeologicBody) -> None:
        """background_heat_flow defaults to 65.0 mW/m²."""
        model = HeatFlowModel(
            bodies=[thermal_body],
            observation_x=[0.0],
            observation_z=0.0,
        )
        assert model.background_heat_flow == 65.0

    def test_background_heat_flow_custom(self, thermal_body: GeologicBody) -> None:
        """Custom background_heat_flow value is stored correctly."""
        model = HeatFlowModel(
            bodies=[thermal_body],
            observation_x=[0.0],
            observation_z=0.0,
            background_heat_flow=80.0,
        )
        assert model.background_heat_flow == 80.0

    def test_background_heat_flow_non_finite_rejected(
        self, thermal_body: GeologicBody
    ) -> None:
        """Non-finite background_heat_flow is rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            HeatFlowModel(
                bodies=[thermal_body],
                observation_x=[0.0],
                observation_z=0.0,
                background_heat_flow=float("inf"),
            )

    def test_get_observation_points_shape(self, heat_flow_model: HeatFlowModel) -> None:
        """get_observation_points() returns an Nx2 array."""
        points = heat_flow_model.get_observation_points()
        assert isinstance(points, np.ndarray)
        assert points.shape == (7, 2)

    def test_get_observation_points_dtype(self, heat_flow_model: HeatFlowModel) -> None:
        """get_observation_points() returns float64 array."""
        points = heat_flow_model.get_observation_points()
        assert points.dtype == np.float64

    def test_get_observation_points_values(
        self, heat_flow_model: HeatFlowModel
    ) -> None:
        """x column matches observation_x; z column is constant observation_z."""
        points = heat_flow_model.get_observation_points()
        assert np.allclose(points[:, 0], [-100.0, -50.0, 0.0, 25.0, 50.0, 100.0, 150.0])
        assert np.allclose(points[:, 1], 0.0)

    def test_immutability(self, heat_flow_model: HeatFlowModel) -> None:
        """HeatFlowModel is immutable (frozen=True)."""
        with pytest.raises(ValidationError):
            heat_flow_model.observation_z = 10.0  # type: ignore

    def test_json_roundtrip(self, heat_flow_model: HeatFlowModel) -> None:
        """model_dump() → HeatFlowModel(**d) round-trip preserves all values."""
        d = heat_flow_model.model_dump()
        restored = HeatFlowModel(**d)
        assert restored.observation_z == heat_flow_model.observation_z
        assert restored.background_heat_flow == heat_flow_model.background_heat_flow
        assert len(restored.bodies) == len(heat_flow_model.bodies)
        assert restored.observation_x == heat_flow_model.observation_x

    def test_multiple_bodies(self, thermal_body: GeologicBody) -> None:
        """Model accepts multiple geologic bodies."""
        body2 = GeologicBody(
            vertices=[[100.0, 50.0], [150.0, 50.0], [150.0, 100.0], [100.0, 100.0]],
            susceptibility=0.0,
            thermal_conductivity=1.5,
            name="Shale",
        )
        model = HeatFlowModel(
            bodies=[thermal_body, body2],
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )
        assert len(model.bodies) == 2
        assert model.bodies[0].name == "Granite"
        assert model.bodies[1].name == "Shale"
