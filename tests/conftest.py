"""Pytest fixtures for testing."""

import pytest

from forward_model.models import (
    MagneticModel,
    GeologicBody,
    GravityModel,
    GravityProperties,
    HeatFlowModel,
    MagneticField,
    MagneticProperties,
    ThermalProperties,
)


@pytest.fixture
def simple_rectangle() -> GeologicBody:
    """A simple rectangular body centered at x=25, depth 100-200m."""
    return GeologicBody(
        vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
        magnetic=MagneticProperties(susceptibility=0.05),
        name="Rectangle",
    )


@pytest.fixture
def earth_field() -> MagneticField:
    """Typical Earth magnetic field (mid-latitude Northern hemisphere)."""
    return MagneticField(
        intensity=50000.0,  # nT
        inclination=60.0,  # degrees
        declination=0.0,  # degrees
    )


@pytest.fixture
def simple_model(
    simple_rectangle: GeologicBody, earth_field: MagneticField
) -> MagneticModel:
    """A simple forward model with one rectangular body."""
    return MagneticModel(
        bodies=[simple_rectangle],
        field=earth_field,
        observation_x=[-100.0, -50.0, 0.0, 25.0, 50.0, 100.0, 150.0],
        observation_z=0.0,
    )


@pytest.fixture
def gravity_body() -> GeologicBody:
    """A simple rectangular gravity body."""
    return GeologicBody(
        vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
        gravity=GravityProperties(density_contrast=300.0),
        name="GravityBody",
    )


@pytest.fixture
def gravity_model(gravity_body: GeologicBody) -> GravityModel:
    """A simple gravity model with one body."""
    return GravityModel(
        bodies=[gravity_body],
        observation_x=[-100.0, -50.0, 0.0, 25.0, 50.0, 100.0, 150.0],
        observation_z=0.0,
    )


@pytest.fixture
def body_2_5d() -> GeologicBody:
    """2.5D body with strike_half_length=5000 m."""
    return GeologicBody(
        vertices=[[0, 100], [50, 100], [50, 200], [0, 200]],
        magnetic=MagneticProperties(susceptibility=0.05),
        name="Body2_5D",
        strike_half_length=5000.0,
    )


@pytest.fixture
def body_2_75d() -> GeologicBody:
    """2.75D body with asymmetric strike (forward=8000, backward=3000)."""
    return GeologicBody(
        vertices=[[0, 100], [50, 100], [50, 200], [0, 200]],
        magnetic=MagneticProperties(susceptibility=0.08),
        name="Body2_75D",
        strike_forward=8000.0,
        strike_backward=3000.0,
    )


@pytest.fixture
def model_2_5d(body_2_5d: GeologicBody, earth_field: MagneticField) -> MagneticModel:
    """Forward model with a 2.5D body."""
    return MagneticModel(
        bodies=[body_2_5d],
        field=earth_field,
        observation_x=[-100.0, 0.0, 100.0],
        observation_z=0.0,
    )


@pytest.fixture
def model_2_75d(body_2_75d: GeologicBody, earth_field: MagneticField) -> MagneticModel:
    """Forward model with a 2.75D body."""
    return MagneticModel(
        bodies=[body_2_75d],
        field=earth_field,
        observation_x=[-100.0, 0.0, 100.0],
        observation_z=0.0,
    )


@pytest.fixture
def thermal_body() -> GeologicBody:
    """A body with thermal properties set (granite-like: 3.3 W/m·K, 2.5 µW/m³)."""
    return GeologicBody(
        vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
        thermal=ThermalProperties(conductivity=3.3, heat_generation=2.5),
        name="Granite",
    )


@pytest.fixture
def heat_flow_model(thermal_body: GeologicBody) -> HeatFlowModel:
    """A minimal HeatFlowModel with one thermal body."""
    return HeatFlowModel(
        bodies=[thermal_body],
        observation_x=[-100.0, -50.0, 0.0, 25.0, 50.0, 100.0, 150.0],
        observation_z=0.0,
    )
