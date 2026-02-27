"""Heat flow model container."""

import math

from pydantic import Field, field_validator

from forward_model.models.base import ObservationModel


class HeatFlowModel(ObservationModel, frozen=True):
    """Complete heat flow model specification.

    Attributes:
        bodies: List of geologic bodies to include in the model.
                Must contain at least one body.
        observation_x: List of x-coordinates for observation points (meters).
        observation_z: Fixed z-coordinate for all observation points (meters).
                      Typically 0 for surface observations.
        background_heat_flow: Regional background heat flow (mW/m²). Added to
                             the computed anomaly to give absolute heat flow.
                             Default is 65.0 (continental average).
    """

    background_heat_flow: float = Field(default=65.0)

    @field_validator("background_heat_flow")
    @classmethod
    def validate_background_heat_flow(cls, v: float) -> float:
        """Validate background heat flow is finite."""
        if not math.isfinite(v):
            raise ValueError(f"background_heat_flow must be finite, got {v}")
        return v
