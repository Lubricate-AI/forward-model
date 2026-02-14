"""Tests for MagneticField model."""

import pytest
from pydantic import ValidationError

from forward_model.models import MagneticField


class TestMagneticField:
    """Tests for MagneticField model."""

    def test_valid_field(self, earth_field: MagneticField) -> None:
        """Test creating a valid magnetic field."""
        assert earth_field.intensity == 50000.0
        assert earth_field.inclination == 60.0
        assert earth_field.declination == 0.0

    def test_negative_intensity(self) -> None:
        """Test that negative intensity is rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MagneticField(intensity=-1000.0, inclination=60.0, declination=0.0)

    def test_zero_intensity(self) -> None:
        """Test that zero intensity is rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MagneticField(intensity=0.0, inclination=60.0, declination=0.0)

    def test_inclination_range(self) -> None:
        """Test that inclination must be in valid range."""
        # Valid boundary values
        MagneticField(intensity=50000.0, inclination=-90.0, declination=0.0)
        MagneticField(intensity=50000.0, inclination=90.0, declination=0.0)

        # Invalid values
        with pytest.raises(ValidationError, match="greater than or equal to -90"):
            MagneticField(intensity=50000.0, inclination=-91.0, declination=0.0)
        with pytest.raises(ValidationError, match="less than or equal to 90"):
            MagneticField(intensity=50000.0, inclination=91.0, declination=0.0)

    def test_declination_range(self) -> None:
        """Test that declination must be in valid range."""
        # Valid boundary values
        MagneticField(intensity=50000.0, inclination=60.0, declination=-180.0)
        MagneticField(intensity=50000.0, inclination=60.0, declination=180.0)

        # Invalid values
        with pytest.raises(ValidationError, match="greater than or equal to -180"):
            MagneticField(intensity=50000.0, inclination=60.0, declination=-181.0)
        with pytest.raises(ValidationError, match="less than or equal to 180"):
            MagneticField(intensity=50000.0, inclination=60.0, declination=181.0)

    def test_non_finite_values(self) -> None:
        """Test that non-finite field values are rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            MagneticField(intensity=float("inf"), inclination=60.0, declination=0.0)
        # NaN fails range validation before finite check
        with pytest.raises(ValidationError):
            MagneticField(intensity=50000.0, inclination=float("nan"), declination=0.0)

    def test_immutability(self, earth_field: MagneticField) -> None:
        """Test that MagneticField is immutable."""
        with pytest.raises(ValidationError):
            earth_field.intensity = 60000.0  # type: ignore
