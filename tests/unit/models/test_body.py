"""Tests for GeologicBody model."""

import numpy as np
import pytest
from pydantic import ValidationError

from forward_model.models import GeologicBody


class TestGeologicBody:
    """Tests for GeologicBody model."""

    def test_valid_body(self, simple_rectangle: GeologicBody) -> None:
        """Test creating a valid geologic body."""
        assert len(simple_rectangle.vertices) == 4
        assert simple_rectangle.susceptibility == 0.05
        assert simple_rectangle.name == "Rectangle"

    def test_minimum_vertices(self) -> None:
        """Test that at least 3 vertices are required."""
        with pytest.raises(ValidationError, match="at least 3 items"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0]],
                susceptibility=0.01,
                name="Invalid",
            )

    def test_vertex_dimension_validation(self) -> None:
        """Test that vertices must be 2D coordinates."""
        with pytest.raises(ValidationError, match="exactly 2 coordinates"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0, 2.0]],
                susceptibility=0.01,
                name="Invalid",
            )

    def test_non_finite_vertices(self) -> None:
        """Test that non-finite vertex coordinates are rejected."""
        with pytest.raises(ValidationError, match="non-finite values"):
            GeologicBody(
                vertices=[[0.0, 0.0], [float("inf"), 0.0], [1.0, 1.0]],
                susceptibility=0.01,
                name="Invalid",
            )

    def test_non_finite_susceptibility(self) -> None:
        """Test that non-finite susceptibility is rejected."""
        with pytest.raises(ValidationError, match="must be finite"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                susceptibility=float("nan"),
                name="Invalid",
            )

    def test_to_numpy(self, simple_rectangle: GeologicBody) -> None:
        """Test conversion to NumPy array."""
        arr = simple_rectangle.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 2)
        assert arr.dtype == np.float64
        assert np.allclose(arr[0], [0.0, 100.0])

    def test_immutability(self, simple_rectangle: GeologicBody) -> None:
        """Test that GeologicBody is immutable."""
        with pytest.raises(ValidationError):
            simple_rectangle.susceptibility = 0.1  # type: ignore


class TestGeologicBodyLabelLoc:
    """Tests for the label_loc field on GeologicBody."""

    def test_label_loc_default_is_none(self) -> None:
        """label_loc defaults to None when not provided."""
        body = GeologicBody(
            vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            susceptibility=0.01,
            name="Body",
        )
        assert body.label_loc is None

    def test_label_loc_valid(self) -> None:
        """label_loc stores a valid 2-element coordinate list."""
        body = GeologicBody(
            vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            susceptibility=0.01,
            name="Body",
            label_loc=[100.0, 200.0],
        )
        assert body.label_loc == [100.0, 200.0]

    def test_label_loc_wrong_length(self) -> None:
        """label_loc with wrong number of elements raises ValidationError."""
        with pytest.raises(ValidationError, match="exactly 2 coordinates"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                susceptibility=0.01,
                name="Body",
                label_loc=[100.0, 200.0, 300.0],
            )

    def test_label_loc_non_finite(self) -> None:
        """label_loc with non-finite values raises ValidationError."""
        with pytest.raises(ValidationError, match="non-finite values"):
            GeologicBody(
                vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                susceptibility=0.01,
                name="Body",
                label_loc=[float("inf"), 200.0],
            )


class TestGeologicBodyVisualProperties:
    """Tests for the color and hatch fields on GeologicBody."""

    _VERTS = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

    def _body(self, **kwargs: object) -> GeologicBody:
        return GeologicBody(
            vertices=self._VERTS,
            susceptibility=0.01,
            name="Body",
            **kwargs,  # type: ignore[arg-type]
        )

    # --- color field ---

    def test_color_default_is_none(self) -> None:
        """color defaults to None when not provided."""
        assert self._body().color is None

    def test_color_string_named(self) -> None:
        """Named color string is accepted."""
        body = self._body(color="red")
        assert body.color == "red"

    def test_color_string_hex(self) -> None:
        """Hex color string is accepted."""
        body = self._body(color="#87CEEB")
        assert body.color == "#87CEEB"

    def test_color_rgb_list(self) -> None:
        """RGB list of floats in [0, 1] is accepted."""
        body = self._body(color=[0.5, 0.3, 0.1])
        assert body.color == [0.5, 0.3, 0.1]

    def test_color_rgba_list(self) -> None:
        """RGBA list of floats in [0, 1] is accepted."""
        body = self._body(color=[0.5, 0.3, 0.1, 0.8])
        assert body.color == [0.5, 0.3, 0.1, 0.8]

    def test_color_list_wrong_length(self) -> None:
        """Color list with wrong number of elements raises ValidationError."""
        with pytest.raises(ValidationError, match="3 \\(RGB\\) or 4 \\(RGBA\\)"):
            self._body(color=[0.5, 0.3])

    def test_color_list_out_of_range(self) -> None:
        """Color list with value outside [0, 1] raises ValidationError."""
        with pytest.raises(ValidationError, match="out of range"):
            self._body(color=[1.5, 0.3, 0.1])

    def test_color_list_non_finite(self) -> None:
        """Color list with non-finite value raises ValidationError."""
        with pytest.raises(ValidationError, match="non-finite"):
            self._body(color=[float("nan"), 0.3, 0.1])

    # --- hatch field ---

    def test_hatch_default_is_none(self) -> None:
        """hatch defaults to None when not provided."""
        assert self._body().hatch is None

    def test_hatch_valid_forward_slash(self) -> None:
        """Forward-slash hatch pattern is accepted."""
        body = self._body(hatch="///")
        assert body.hatch == "///"

    def test_hatch_valid_backslash(self) -> None:
        """Backslash hatch pattern is accepted."""
        body = self._body(hatch="\\\\")
        assert body.hatch == "\\\\"


class TestGeologicBodyRemanentMagnetization:
    """Tests for remanent magnetization fields on GeologicBody."""

    _VERTS = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

    def _body(self, **kwargs: object) -> GeologicBody:
        return GeologicBody(
            vertices=self._VERTS,
            susceptibility=0.01,
            name="Body",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_remanent_defaults_to_zero(self) -> None:
        """All three remanent fields default to 0.0."""
        body = self._body()
        assert body.remanent_intensity == 0.0
        assert body.remanent_inclination == 0.0
        assert body.remanent_declination == 0.0

    def test_remanent_intensity_valid(self) -> None:
        """Positive remanent intensity is accepted."""
        body = self._body(remanent_intensity=2.5)
        assert body.remanent_intensity == 2.5

    def test_remanent_intensity_negative_rejected(self) -> None:
        """Negative remanent intensity is rejected (ge=0.0 constraint)."""
        with pytest.raises(ValidationError):
            self._body(remanent_intensity=-0.1)

    def test_remanent_intensity_non_finite_rejected(self) -> None:
        """Non-finite remanent intensity (inf, nan) raises ValidationError."""
        # inf passes ge=0.0 but is caught by the finiteness validator
        with pytest.raises(ValidationError, match="finite"):
            self._body(remanent_intensity=float("inf"))
        # nan fails ge=0.0 directly (nan >= 0 is False); still rejected
        with pytest.raises(ValidationError):
            self._body(remanent_intensity=float("nan"))

    def test_remanent_inclination_bounds(self) -> None:
        """±90° inclination is accepted; ±91° is rejected."""
        body_pos = self._body(remanent_inclination=90.0)
        assert body_pos.remanent_inclination == 90.0
        body_neg = self._body(remanent_inclination=-90.0)
        assert body_neg.remanent_inclination == -90.0
        with pytest.raises(ValidationError):
            self._body(remanent_inclination=91.0)
        with pytest.raises(ValidationError):
            self._body(remanent_inclination=-91.0)

    def test_remanent_declination_bounds(self) -> None:
        """±180° declination is accepted; ±181° is rejected."""
        body_pos = self._body(remanent_declination=180.0)
        assert body_pos.remanent_declination == 180.0
        body_neg = self._body(remanent_declination=-180.0)
        assert body_neg.remanent_declination == -180.0
        with pytest.raises(ValidationError):
            self._body(remanent_declination=181.0)
        with pytest.raises(ValidationError):
            self._body(remanent_declination=-181.0)

    def test_json_roundtrip_preserves_remanent_fields(self) -> None:
        """model_dump() → GeologicBody(**d) round-trip preserves remanent values."""
        original = self._body(
            remanent_intensity=1.5,
            remanent_inclination=45.0,
            remanent_declination=-30.0,
        )
        d = original.model_dump()
        restored = GeologicBody(**d)
        assert restored.remanent_intensity == original.remanent_intensity
        assert restored.remanent_inclination == original.remanent_inclination
        assert restored.remanent_declination == original.remanent_declination
