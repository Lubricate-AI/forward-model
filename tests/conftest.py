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


@pytest.fixture
def body_2_5d() -> GeologicBody:
    """2.5D body with strike_half_length=5000 m."""
    return GeologicBody(
        vertices=[[0, 100], [50, 100], [50, 200], [0, 200]],
        susceptibility=0.05,
        name="Body2_5D",
        strike_half_length=5000.0,
    )


@pytest.fixture
def body_2_75d() -> GeologicBody:
    """2.75D body with asymmetric strike (forward=8000, backward=3000)."""
    return GeologicBody(
        vertices=[[0, 100], [50, 100], [50, 200], [0, 200]],
        susceptibility=0.08,
        name="Body2_75D",
        strike_forward=8000.0,
        strike_backward=3000.0,
    )


@pytest.fixture
def model_2_5d(body_2_5d: GeologicBody, earth_field: MagneticField) -> ForwardModel:
    """Forward model with a 2.5D body."""
    return ForwardModel(
        bodies=[body_2_5d],
        field=earth_field,
        observation_x=[-100.0, 0.0, 100.0],
        observation_z=0.0,
    )


@pytest.fixture
def model_2_75d(body_2_75d: GeologicBody, earth_field: MagneticField) -> ForwardModel:
    """Forward model with a 2.75D body."""
    return ForwardModel(
        bodies=[body_2_75d],
        field=earth_field,
        observation_x=[-100.0, 0.0, 100.0],
        observation_z=0.0,
    )
