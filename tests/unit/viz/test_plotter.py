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

        assert len(axes) == 2
        # First subplot should have polygons (cross-section)
        assert len(axes[0].patches) > 0
        # Second subplot should have lines (anomaly)
        assert len(axes[1].lines) > 0

        plt.close()
