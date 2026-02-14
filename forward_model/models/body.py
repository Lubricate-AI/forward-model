"""Geologic body data model."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator


class GeologicBody(BaseModel, frozen=True):
    """A 2D polygonal geologic body with magnetic susceptibility.

    Attributes:
        vertices: List of [x, z] coordinate pairs defining the polygon boundary.
                 Minimum 3 vertices required. Coordinates in meters.
        susceptibility: Magnetic susceptibility (SI units, dimensionless).
                       Must be finite.
        name: Human-readable identifier for this body.
    """

    vertices: list[list[float]] = Field(min_length=3)
    susceptibility: float
    name: str

    @field_validator("vertices")
    @classmethod
    def validate_vertices(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate vertex list structure and values."""
        if len(v) < 3:
            raise ValueError("At least 3 vertices required for a polygon")

        for i, vertex in enumerate(v):
            if len(vertex) != 2:
                msg = (
                    f"Vertex {i} must have exactly 2 coordinates (x, z), "
                    f"got {len(vertex)}"
                )
                raise ValueError(msg)
            if not all(math.isfinite(coord) for coord in vertex):
                raise ValueError(f"Vertex {i} contains non-finite values: {vertex}")

        return v

    @field_validator("susceptibility")
    @classmethod
    def validate_susceptibility(cls, v: float) -> float:
        """Validate susceptibility is finite."""
        if not math.isfinite(v):
            raise ValueError(f"Susceptibility must be finite, got {v}")
        return v

    def to_numpy(self) -> NDArray[np.float64]:
        """Convert vertices to NumPy array.

        Returns:
            Nx2 array of vertex coordinates where N is the number of vertices.
        """
        return np.asarray(self.vertices, dtype=np.float64)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to ensure consistent serialization."""
        return super().model_dump(**kwargs)
