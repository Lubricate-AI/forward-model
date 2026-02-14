"""Tests for plot styles."""

import pytest

from forward_model.viz.styles import PLOT_STYLES, get_style


class TestPlotStyles:
    """Tests for plot style configurations."""

    def test_default_style_exists(self) -> None:
        """Test that default style exists."""
        assert "default" in PLOT_STYLES

    def test_publication_style_exists(self) -> None:
        """Test that publication style exists."""
        assert "publication" in PLOT_STYLES

    def test_presentation_style_exists(self) -> None:
        """Test that presentation style exists."""
        assert "presentation" in PLOT_STYLES

    def test_get_style_default(self) -> None:
        """Test getting default style."""
        style = get_style("default")
        assert isinstance(style, dict)
        assert "font.size" in style
        assert "savefig.dpi" in style

    def test_get_style_publication(self) -> None:
        """Test getting publication style."""
        style = get_style("publication")
        assert isinstance(style, dict)
        assert style["savefig.dpi"] == 300
        assert "serif" in str(style["font.family"])

    def test_get_style_presentation(self) -> None:
        """Test getting presentation style."""
        style = get_style("presentation")
        assert isinstance(style, dict)
        assert style["font.size"] == 14
        assert style["savefig.dpi"] == 200

    def test_get_style_invalid(self) -> None:
        """Test that invalid style name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown style"):
            get_style("nonexistent")

    def test_publication_has_higher_dpi(self) -> None:
        """Test that publication style has higher DPI than default."""
        default = get_style("default")
        publication = get_style("publication")

        assert publication["savefig.dpi"] > default["savefig.dpi"]

    def test_all_styles_have_required_keys(self) -> None:
        """Test that all styles have essential keys."""
        required_keys = ["font.size", "savefig.dpi", "figure.facecolor"]

        for style_name in PLOT_STYLES:
            style = get_style(style_name)
            for key in required_keys:
                assert key in style, f"Style '{style_name}' missing key '{key}'"
