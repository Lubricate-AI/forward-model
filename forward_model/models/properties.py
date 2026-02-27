"""Physical property group sub-models for geologic bodies."""

import math
import warnings

from pydantic import BaseModel, Field, field_validator


class MagneticProperties(BaseModel, frozen=True):
    """Magnetic physical properties for a geologic body.

    Attributes:
        susceptibility: Magnetic susceptibility (SI units, dimensionless).
                       Must be finite.
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
    """

    susceptibility: float
    remanent_intensity: float = Field(default=0.0, ge=0.0)
    remanent_inclination: float = Field(default=0.0, ge=-90.0, le=90.0)
    remanent_declination: float = Field(default=0.0, ge=-180.0, le=180.0)
    demagnetization_factor: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("susceptibility")
    @classmethod
    def validate_susceptibility(cls, v: float) -> float:
        """Validate susceptibility is finite."""
        if not math.isfinite(v):
            raise ValueError(f"susceptibility must be finite, got {v}")
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


class GravityProperties(BaseModel, frozen=True):
    """Gravity physical properties for a geologic body.

    Attributes:
        density_contrast: Density contrast in kg/m³. Accepts any finite value
                         (positive, negative, or zero — density contrasts can
                         be signed).
    """

    density_contrast: float

    @field_validator("density_contrast")
    @classmethod
    def validate_density_contrast(cls, v: float) -> float:
        """Validate density_contrast is finite."""
        if not math.isfinite(v):
            raise ValueError(f"density_contrast must be finite, got {v}")
        return v


class ThermalProperties(BaseModel, frozen=True):
    """Thermal physical properties for a geologic body.

    Attributes:
        conductivity: Thermal conductivity in W/m·K. Must be strictly positive
                     and finite.
        heat_generation: Radiogenic heat generation in µW/m³. Must be
                        non-negative and finite. Default is 0.0.
    """

    conductivity: float = Field(gt=0.0)
    heat_generation: float = Field(default=0.0, ge=0.0)

    @field_validator("conductivity", "heat_generation")
    @classmethod
    def validate_thermal_fields(cls, v: float) -> float:
        """Validate thermal fields are finite."""
        if not math.isfinite(v):
            raise ValueError(f"Thermal field must be finite, got {v}")
        return v
