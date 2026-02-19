"""Geologic body data model."""

import math
import warnings
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
        label_loc: Optional [x, z] override for the label position. When set,
                  the plotter uses this location directly (no clamping applied).
        color: Optional matplotlib color for this body. Accepts any named color
               string (e.g. ``"red"``, ``"#87CEEB"``) or an RGB/RGBA list of
               floats in ``[0.0, 1.0]``. When set, overrides the global colormap.
        hatch: Optional matplotlib hatch pattern string (e.g. ``"///"``,
               ``"\\\\"``) applied as a fill pattern. ``None`` means no hatch.
        remanent_intensity: Remanent magnetization intensity in A/m. Must be
                           non-negative and finite. Default is 0.0 (no remanence).
        remanent_inclination: Inclination of the remanent magnetization vector
                             in degrees (-90 to 90). Positive downward. Default 0.0.
        remanent_declination: Declination of the remanent magnetization vector
                             in degrees (-180 to 180). Default 0.0.
        demagnetization_factor: Demagnetization factor N_d in [0.0, 1.0]. Controls
                               the reduction of effective susceptibility for
                               high-susceptibility bodies via
                               χ_eff = χ / (1 + N_d·χ). Default is 0.0 (no
                               correction). For 2D infinite-strike bodies the
                               physically meaningful range is [0.0, 0.5]; values
                               above 0.5 are accepted but trigger a UserWarning.
        strike_half_length: Half-length of the body in the strike direction (m).
                           When ``None`` (default), the standard 2D (infinite-strike)
                           Talwani (1965) algorithm is used. When a positive finite
                           value is provided, the Won & Bevis (1987) 2.5D correction
                           factor ``y0 / sqrt(r² + y0²)`` is applied per edge,
                           attenuating the anomaly for bodies with limited lateral
                           extent. Must be strictly positive and finite.
    """

    vertices: list[list[float]] = Field(min_length=3)
    susceptibility: float
    name: str
    label_loc: list[float] | None = Field(default=None)
    color: str | list[float] | None = Field(default=None)
    hatch: str | None = Field(default=None)
    remanent_intensity: float = Field(default=0.0, ge=0.0)
    remanent_inclination: float = Field(default=0.0, ge=-90.0, le=90.0)
    remanent_declination: float = Field(default=0.0, ge=-180.0, le=180.0)
    demagnetization_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    strike_half_length: float | None = Field(default=None, gt=0.0)

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

    @field_validator(
        "remanent_intensity", "remanent_inclination", "remanent_declination"
    )
    @classmethod
    def validate_remanent_fields(cls, v: float) -> float:
        """Validate remanent fields are finite."""
        if not math.isfinite(v):
            raise ValueError(f"Remanent field must be finite, got {v}")
        return v

    @field_validator("demagnetization_factor")
    @classmethod
    def validate_demagnetization_factor(cls, v: float) -> float:
        """Validate demagnetization factor is finite; warn if above 2D limit."""
        if not math.isfinite(v):
            raise ValueError(f"demagnetization_factor must be finite, got {v}")
        if v > 0.5:
            warnings.warn(
                f"demagnetization_factor={v} exceeds 0.5, which is the physical "
                "upper bound for 2D infinite-strike bodies. Values above 0.5 are "
                "only valid for 3D geometries.",
                UserWarning,
                stacklevel=2,
            )
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
