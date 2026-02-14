"""Pytest fixtures for testing."""

import pytest

from forward_model.models import ForwardModel, GeologicBody, MagneticField


@pytest.fixture
def simple_rectangle() -> GeologicBody:
    """A simple rectangular body centered at x=25, depth 100-200m."""
    return GeologicBody(
        vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
        susceptibility=0.05,
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
) -> ForwardModel:
    """A simple forward model with one rectangular body."""
    return ForwardModel(
        bodies=[simple_rectangle],
        field=earth_field,
        observation_x=[-100.0, -50.0, 0.0, 25.0, 50.0, 100.0, 150.0],
        observation_z=0.0,
    )
