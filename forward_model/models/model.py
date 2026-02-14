"""Forward model container."""

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from forward_model.models.body import GeologicBody
from forward_model.models.field import MagneticField


class ForwardModel(BaseModel, frozen=True):
    """Complete forward magnetic model specification.

    Attributes:
        bodies: List of geologic bodies to include in the model.
                Must contain at least one body.
        field: Earth's magnetic field parameters.
        observation_x: List of x-coordinates for observation points (meters).
        observation_z: Fixed z-coordinate for all observation points (meters).
                      Typically 0 for surface observations.
    """

    bodies: list[GeologicBody] = Field(min_length=1)
    field: MagneticField
    observation_x: list[float]
    observation_z: float

    @field_validator("observation_x")
    @classmethod
    def validate_observation_x(cls, v: list[float]) -> list[float]:
        """Validate observation x coordinates are finite."""
        if not all(math.isfinite(x) for x in v):
            raise ValueError("All observation x coordinates must be finite")
        return v

    @field_validator("observation_z")
    @classmethod
    def validate_observation_z(cls, v: float) -> float:
        """Validate observation z coordinate is finite."""
        if not math.isfinite(v):
            raise ValueError(f"observation_z must be finite, got {v}")
        return v

    def get_observation_points(self) -> NDArray[np.float64]:
        """Get observation points as NumPy array.

        Returns:
            Nx2 array where N is the number of observation points.
            Each row is [x, z] coordinates.
        """
        n_points = len(self.observation_x)
        points = np.zeros((n_points, 2), dtype=np.float64)
        points[:, 0] = self.observation_x
        points[:, 1] = self.observation_z
        return points
