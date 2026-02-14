"""Tests for visualization functions."""

# pyright: reportUnknownMemberType=false

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forward_model.models import ForwardModel
from forward_model.viz import plot_anomaly, plot_combined, plot_model


class TestPlotModel:
    """Tests for plot_model function."""

    def test_plot_model_creates_axes(self, simple_model: ForwardModel) -> None:
        """Test that plot_model creates axes without errors."""
        ax = plot_model(simple_model)
        assert ax is not None
        plt.close()

    def test_plot_model_with_custom_axes(self, simple_model: ForwardModel) -> None:
        """Test providing custom axes."""
        _, ax = plt.subplots()
        result_ax = plot_model(simple_model, ax=ax)
        assert result_ax is ax
        plt.close()

    def test_plot_model_has_elements(self, simple_model: ForwardModel) -> None:
        """Test that plot contains expected elements."""
        ax = plot_model(simple_model)

        # Check that patches (polygons) were added
        assert len(ax.patches) > 0

        # Check that observation points were plotted
        assert len(ax.lines) > 0

        plt.close()

    def test_plot_model_color_by_index(self, simple_model: ForwardModel) -> None:
        """Test plotting with color_by='index'."""
        ax = plot_model(simple_model, color_by="index")
        assert ax is not None
        assert len(ax.patches) > 0
        plt.close()

    def test_plot_model_color_by_susceptibility(
        self, simple_model: ForwardModel
    ) -> None:
        """Test plotting with color_by='susceptibility'."""
        ax = plot_model(simple_model, color_by="susceptibility")
        assert ax is not None
        assert len(ax.patches) > 0
        plt.close()

    def test_plot_model_with_observation_lines(
        self, simple_model: ForwardModel
    ) -> None:
        """Test plotting with observation lines enabled."""
        ax = plot_model(simple_model, show_observation_lines=True)
        assert ax is not None
        # Lines should include observation points plus vertical lines
        assert len(ax.lines) > 0
        plt.close()

    def test_plot_model_without_observation_lines(
        self, simple_model: ForwardModel
    ) -> None:
        """Test plotting with observation lines disabled."""
        ax = plot_model(simple_model, show_observation_lines=False)
        assert ax is not None
        assert len(ax.patches) > 0
        plt.close()


class TestPlotAnomaly:
    """Tests for plot_anomaly function."""

    def test_plot_anomaly_creates_axes(self) -> None:
        """Test that plot_anomaly creates axes without errors."""
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])

        ax = plot_anomaly(obs_x, anomaly)
        assert ax is not None
        plt.close()

    def test_plot_anomaly_with_custom_axes(self) -> None:
        """Test providing custom axes."""
        _, ax = plt.subplots()
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])

        result_ax = plot_anomaly(obs_x, anomaly, ax=ax)
        assert result_ax is ax
        plt.close()

    def test_plot_anomaly_has_elements(self) -> None:
        """Test that plot contains expected elements."""
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])

        ax = plot_anomaly(obs_x, anomaly)

        # Check that lines were plotted
        assert len(ax.lines) > 0

        plt.close()


class TestPlotCombined:
    """Tests for plot_combined function."""

    def test_plot_combined_creates_figure(self, simple_model: ForwardModel) -> None:
        """Test that plot_combined creates a figure without errors."""
        anomaly = np.random.randn(len(simple_model.observation_x))

        fig = plot_combined(simple_model, anomaly)
        assert fig is not None
        assert len(fig.axes) == 2  # Two subplots
        plt.close()

    def test_plot_combined_saves_to_file(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test saving combined plot to file."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        output_file = tmp_path / "test_plot.png"

        fig = plot_combined(simple_model, anomaly, save_path=output_file)
        assert output_file.exists()
        plt.close(fig)

    def test_plot_combined_with_string_path(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test that string paths work for saving."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        output_file = tmp_path / "test_plot.png"

        fig = plot_combined(simple_model, anomaly, save_path=str(output_file))
        assert output_file.exists()
        plt.close(fig)

    def test_plot_combined_subplots(self, simple_model: ForwardModel) -> None:
        """Test that combined plot has two subplots."""
        anomaly = np.random.randn(len(simple_model.observation_x))

        fig = plot_combined(simple_model, anomaly)
        axes = fig.axes

        assert len(axes) >= 2  # At least 2 (may have colorbar)
        # First subplot should have polygons (cross-section)
        assert len(axes[0].patches) > 0
        # Second subplot should have lines (anomaly)
        assert len(axes[1].lines) > 0

        plt.close()

    def test_plot_combined_with_style(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test combined plot with different styles."""
        anomaly = np.random.randn(len(simple_model.observation_x))

        for style in ["default", "publication", "presentation"]:
            output_file = tmp_path / f"test_plot_{style}.png"
            fig = plot_combined(
                simple_model, anomaly, save_path=output_file, style=style
            )
            assert output_file.exists()
            plt.close(fig)

    def test_plot_combined_with_custom_figsize(
        self, simple_model: ForwardModel
    ) -> None:
        """Test combined plot with custom figure size."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        custom_figsize = (10, 6)

        fig = plot_combined(simple_model, anomaly, figsize=custom_figsize)
        # Check that figure size is approximately correct (allowing for tight_layout)
        assert abs(fig.get_figwidth() - custom_figsize[0]) < 0.1
        assert abs(fig.get_figheight() - custom_figsize[1]) < 0.1
        plt.close()

    def test_plot_combined_with_custom_dpi(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test combined plot with custom DPI."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        output_file = tmp_path / "test_plot_300dpi.png"

        fig = plot_combined(simple_model, anomaly, save_path=output_file, dpi=300)
        assert output_file.exists()
        plt.close(fig)

    def test_plot_combined_vector_formats(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test combined plot with vector formats (PDF, SVG)."""
        anomaly = np.random.randn(len(simple_model.observation_x))

        for fmt in ["pdf", "svg"]:
            output_file = tmp_path / f"test_plot.{fmt}"
            fig = plot_combined(simple_model, anomaly, save_path=output_file)
            assert output_file.exists()
            plt.close(fig)

    def test_plot_combined_color_options(self, simple_model: ForwardModel) -> None:
        """Test combined plot with different color options."""
        anomaly = np.random.randn(len(simple_model.observation_x))

        # Test color_by index
        fig1 = plot_combined(simple_model, anomaly, color_by="index")
        assert fig1 is not None
        plt.close(fig1)

        # Test color_by susceptibility
        fig2 = plot_combined(simple_model, anomaly, color_by="susceptibility")
        assert fig2 is not None
        plt.close(fig2)

    def test_plot_combined_observation_lines(self, simple_model: ForwardModel) -> None:
        """Test combined plot with observation lines option."""
        anomaly = np.random.randn(len(simple_model.observation_x))

        # With lines
        fig1 = plot_combined(simple_model, anomaly, show_observation_lines=True)
        assert fig1 is not None
        plt.close(fig1)

        # Without lines
        fig2 = plot_combined(simple_model, anomaly, show_observation_lines=False)
        assert fig2 is not None
        plt.close(fig2)
