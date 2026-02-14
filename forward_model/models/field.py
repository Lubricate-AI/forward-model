"""Magnetic field data model."""

import math

from pydantic import BaseModel, Field, field_validator


class MagneticField(BaseModel, frozen=True):
    """Earth's magnetic field parameters.

    Attributes:
        intensity: Field intensity in nanoTesla (nT). Must be positive.
        inclination: Field inclination in degrees. Range: -90 to 90.
                    Positive is downward (Northern hemisphere typical).
        declination: Field declination in degrees. Range: -180 to 180.
                    Positive is east of north.
    """

    intensity: float = Field(gt=0.0)
    inclination: float = Field(ge=-90.0, le=90.0)
    declination: float = Field(ge=-180.0, le=180.0)

    @field_validator("intensity", "inclination", "declination")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Validate all field values are finite."""
        if not math.isfinite(v):
            raise ValueError(f"Field value must be finite, got {v}")
        return v
