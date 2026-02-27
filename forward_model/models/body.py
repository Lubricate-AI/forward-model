"""Geologic body data model."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from forward_model.models.properties import (
    GravityProperties,
    MagneticProperties,
    ThermalProperties,
)


class GeologicBody(BaseModel, frozen=True):
    """A 2D polygonal geologic body with physical properties for forward modeling.

    Attributes:
        vertices: List of [x, z] coordinate pairs defining the polygon boundary.
                 Minimum 3 vertices required. Coordinates in meters.
        magnetic: Magnetic physical properties (susceptibility, remanence,
                 demagnetization). Optional; set to enable magnetic forward
                 modeling. At least one of ``magnetic``, ``gravity``, or
                 ``thermal`` must be provided.
        gravity: Gravity physical properties (density contrast in kg/m³).
                Optional; set to enable gravity forward modeling. At least one
                of ``magnetic``, ``gravity``, or ``thermal`` must be provided.
        thermal: Thermal physical properties (conductivity in W/m·K, heat
                generation in µW/m³). Optional; set to enable heat flow forward
                modeling. At least one of ``magnetic``, ``gravity``, or
                ``thermal`` must be provided.
        name: Human-readable identifier for this body.
        label_loc: Optional [x, z] override for the label position. When set,
                  the plotter uses this location directly (no clamping applied).
        color: Optional matplotlib color for this body. Accepts any named color
               string (e.g. ``"red"``, ``"#87CEEB"``) or an RGB/RGBA list of
               floats in ``[0.0, 1.0]``. When set, overrides the global colormap.
        hatch: Optional matplotlib hatch pattern string (e.g. ``"///"``,
               ``"\\\\"``) applied as a fill pattern. ``None`` means no hatch.
        strike_half_length: Half-length of the body in the strike direction (m).
                           When ``None`` (default), the standard 2D (infinite-strike)
                           Talwani (1965) algorithm is used. When a positive finite
                           value is provided, the Won & Bevis (1987) 2.5D formulation
                           is used: the standard Talwani vertex functions are replaced
                           with modified versions (Θₖ, Λₖ) that account for finite
                           strike, attenuating the anomaly for bodies with limited
                           lateral extent. Must be strictly positive and finite.
        strike_forward: Forward (+y) half-extent of the body along strike (m).
                       Must be set together with ``strike_backward``; both ``None``
                       or both positive finite values. When set, the Won & Bevis
                       (1987) 2.75D asymmetric formulation is used:
                       ``ΔB = (ΔB_2.5D(y₁) + ΔB_2.5D(y₂)) / 2``. Must be
                       strictly positive and finite.
        strike_backward: Backward (−y) half-extent of the body along strike (m).
                        Must be set together with ``strike_forward``. See
                        ``strike_forward`` for details.
    """

    vertices: list[list[float]] = Field(min_length=3)
    magnetic: MagneticProperties | None = None
    gravity: GravityProperties | None = None
    thermal: ThermalProperties | None = None
    name: str
    label_loc: list[float] | None = Field(default=None)
    color: str | list[float] | None = Field(default=None)
    hatch: str | None = Field(default=None)
    strike_half_length: float | None = Field(default=None, gt=0.0)
    strike_forward: float | None = Field(default=None, gt=0.0)
    strike_backward: float | None = Field(default=None, gt=0.0)

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

    @field_validator("label_loc")
    @classmethod
    def validate_label_loc(cls, v: list[float] | None) -> list[float] | None:
        """Validate label_loc is a 2-element list of finite floats."""
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("label_loc must have exactly 2 coordinates [x, z]")
        if not all(math.isfinite(c) for c in v):
            raise ValueError("label_loc contains non-finite values")
        return v

    @field_validator("strike_half_length")
    @classmethod
    def validate_strike_half_length(cls, v: float | None) -> float | None:
        """Validate strike_half_length is finite."""
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError(f"strike_half_length must be finite, got {v}")
        return v

    @field_validator("strike_forward", "strike_backward")
    @classmethod
    def validate_strike_asymmetric_fields(cls, v: float | None) -> float | None:
        """Validate strike_forward and strike_backward are finite."""
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError(f"strike field must be finite, got {v}")
        return v

    @model_validator(mode="after")
    def validate_strike_fields(self) -> "GeologicBody":
        """Validate that strike_forward and strike_backward are paired."""
        if (self.strike_forward is None) != (self.strike_backward is None):
            raise ValueError(
                "strike_forward and strike_backward must both be set or both be None"
            )
        return self

    @model_validator(mode="after")
    def validate_physical_property(self) -> "GeologicBody":
        """Validate that at least one property group is provided."""
        if self.magnetic is None and self.gravity is None and self.thermal is None:
            raise ValueError(
                "At least one of 'magnetic', 'gravity', or 'thermal' must be provided"
            )
        return self

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: str | list[float] | None) -> str | list[float] | None:
        """Validate color field for list[float] RGB/RGBA values."""
        if v is None or isinstance(v, str):
            return v
        if len(v) not in (3, 4):
            raise ValueError(
                f"color list must have 3 (RGB) or 4 (RGBA) elements, got {len(v)}"
            )
        for component in v:
            if not math.isfinite(component):
                raise ValueError(f"color component {component!r} is non-finite")
            if not (0.0 <= component <= 1.0):
                raise ValueError(
                    f"color component {component!r} is out of range [0.0, 1.0]"
                )
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
