"""Tests for visualization functions."""

# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
# pyright: reportOperatorIssue=false, reportUnknownArgumentType=false
# pyright: reportGeneralTypeIssues=false

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forward_model.models import ForwardModel
from forward_model.viz import plot_anomaly, plot_combined, plot_model, plot_model_3d
from forward_model.viz.plotter import _polygon_centroid


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

    def test_plot_model_with_xlim(self, simple_model: ForwardModel) -> None:
        """Test that xlim correctly sets the x-axis limits."""
        xlim = (-50.0, 75.0)
        ax = plot_model(simple_model, xlim=xlim)
        assert ax.get_xlim() == xlim
        plt.close()

    def test_plot_model_with_zlim(self, simple_model: ForwardModel) -> None:
        """Test that zlim correctly sets the depth axis limits."""
        zlim = (50.0, 250.0)
        ax = plot_model(simple_model, zlim=zlim)
        assert ax.get_ylim() == zlim
        plt.close()

    def test_plot_model_show_colorbar_false(self) -> None:
        """Test that show_colorbar=False suppresses the colorbar."""
        from forward_model.models import GeologicBody, MagneticField

        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            susceptibility=0.05,
            name="Body1",
        )
        body2 = GeologicBody(
            vertices=[[60.0, 100.0], [110.0, 100.0], [110.0, 200.0], [60.0, 200.0]],
            susceptibility=0.10,
            name="Body2",
        )
        field = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)
        model = ForwardModel(
            bodies=[body1, body2],
            field=field,
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )

        fig, ax = plt.subplots()
        plot_model(model, ax=ax, show_colorbar=False)
        # Only the original axes — no colorbar axes added
        assert len(fig.axes) == 1
        plt.close()

    def test_plot_model_equal_aspect_false(self, simple_model: ForwardModel) -> None:
        """Test that equal_aspect=False does not lock axes to equal scale."""
        ax = plot_model(simple_model, equal_aspect=False)
        assert ax.get_aspect() != "equal"
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

    def test_plot_anomaly_with_xlim(self) -> None:
        """Test that xlim correctly sets the x-axis limits."""
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])
        xlim = (5.0, 15.0)

        ax = plot_anomaly(obs_x, anomaly, xlim=xlim)
        assert ax.get_xlim() == xlim
        plt.close()

    def test_plot_anomaly_gradient_overlay_creates_twin_axis(self) -> None:
        """Supplying gradient adds a secondary y-axis (twinx)."""
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.0, -0.1])

        fig, ax = plt.subplots()
        plot_anomaly(obs_x, anomaly, ax=ax, gradient=gradient)

        # twinx() adds a second axes to the same figure
        assert len(fig.axes) == 2

        # Assert legend contains both labels
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "ΔT (nT)" in legend_texts
        assert "d(ΔT)/dx (nT/m)" in legend_texts
        plt.close()

    def test_plot_anomaly_gradient_overlay_xlim_shared(self) -> None:
        """When xlim and gradient are both supplied, the twin axis shares xlim."""
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.0, -0.1])
        xlim = (0.0, 20.0)

        fig, ax = plt.subplots()
        plot_anomaly(obs_x, anomaly, ax=ax, gradient=gradient, xlim=xlim)

        ax_twin = fig.axes[1]  # twinx axes
        assert ax.get_xlim() == xlim
        assert ax_twin.get_xlim() == xlim
        plt.close()

    def test_plot_anomaly_no_gradient_no_twin_axis(self) -> None:
        """Without gradient, no secondary axis is created."""
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])

        fig, ax = plt.subplots()
        plot_anomaly(obs_x, anomaly, ax=ax)

        assert len(fig.axes) == 1
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

    def test_plot_combined_with_xlim(self, simple_model: ForwardModel) -> None:
        """Test that xlim is applied to both panels."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        xlim = (-50.0, 75.0)

        fig = plot_combined(simple_model, anomaly, xlim=xlim)
        assert fig.axes[0].get_xlim() == xlim
        assert fig.axes[1].get_xlim() == xlim
        plt.close(fig)

    def test_plot_combined_with_zlim(self, simple_model: ForwardModel) -> None:
        """Test that zlim is applied to the cross-section panel."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        zlim = (50.0, 250.0)

        fig = plot_combined(simple_model, anomaly, zlim=zlim)
        assert fig.axes[0].get_ylim() == zlim
        plt.close(fig)

    def test_plot_combined_show_colorbar_false(self) -> None:
        """Test that show_colorbar=False suppresses colorbar in combined view."""
        from forward_model.models import GeologicBody, MagneticField

        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            susceptibility=0.05,
            name="Body1",
        )
        body2 = GeologicBody(
            vertices=[[60.0, 100.0], [110.0, 100.0], [110.0, 200.0], [60.0, 200.0]],
            susceptibility=0.10,
            name="Body2",
        )
        field = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)
        model = ForwardModel(
            bodies=[body1, body2],
            field=field,
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )
        anomaly = np.random.randn(3)

        fig_with = plot_combined(model, anomaly, show_colorbar=True)
        axes_with = len(fig_with.axes)
        plt.close(fig_with)

        fig_without = plot_combined(model, anomaly, show_colorbar=False)
        axes_without = len(fig_without.axes)
        plt.close(fig_without)

        # Suppressing colorbar should result in fewer axes
        assert axes_without < axes_with

    def test_plot_combined_gradient_overlay_adds_twin_axis(
        self, simple_model: ForwardModel
    ) -> None:
        """Supplying gradient to plot_combined adds a secondary axis
        in the anomaly panel."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        gradient = np.random.randn(len(simple_model.observation_x))

        fig = plot_combined(simple_model, anomaly, gradient=gradient)

        # Axes: cross-section + anomaly + gradient twin = at least 3
        assert len(fig.axes) >= 3

        # The anomaly panel is axes[1] (axes[0] is the cross-section)
        anomaly_ax = fig.axes[1]
        legend = anomaly_ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "ΔT (nT)" in legend_texts
        assert "d(ΔT)/dx (nT/m)" in legend_texts
        plt.close()

    def test_plot_combined_passes_label_offsets(
        self, simple_model: ForwardModel
    ) -> None:
        """Test label_offsets and show_label_arrows propagate to cross-section axis."""
        from matplotlib.text import Annotation

        anomaly = np.random.randn(len(simple_model.observation_x))
        label_offsets = {"Rectangle": (0.0, -60.0)}

        fig = plot_combined(
            simple_model,
            anomaly,
            label_offsets=label_offsets,
            show_label_arrows=True,
        )
        cross_section_ax = fig.axes[0]
        annotations = [
            c for c in cross_section_ax.get_children() if isinstance(c, Annotation)
        ]
        assert len(annotations) > 0
        plt.close(fig)

    def test_show_3d_adds_third_panel(self, simple_model: ForwardModel) -> None:
        """show_3d=True adds a third (3D) panel to the combined figure."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        fig = plot_combined(simple_model, anomaly, show_3d=True)
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_show_3d_false_keeps_two_panels(self, simple_model: ForwardModel) -> None:
        """show_3d=False (default) keeps the original two-panel layout."""
        anomaly = np.random.randn(len(simple_model.observation_x))
        fig = plot_combined(simple_model, anomaly, show_3d=False)
        assert len(fig.axes) == 2
        plt.close(fig)


class TestPlotModelBodyVisualProperties:
    """Tests verifying plot_model respects body-level color and hatch."""

    def _make_model(
        self,
        color: str | list[float] | None = None,
        hatch: str | None = None,
    ) -> ForwardModel:
        from forward_model.models import GeologicBody, MagneticField

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            susceptibility=0.05,
            name="Body",
            color=color,
            hatch=hatch,
        )
        field = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)
        return ForwardModel(
            bodies=[body],
            field=field,
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )

    def test_body_color_overrides_colormap(self) -> None:
        """Body with color='#FF0000' renders with facecolor matching that red."""
        import matplotlib.colors as mcolors

        model = self._make_model(color="#FF0000")
        ax = plot_model(model)
        patch_color = ax.patches[0].get_facecolor()
        expected = mcolors.to_rgba("#FF0000")
        # Compare RGB channels (alpha may be modified by the alpha=0.6 kwarg)
        assert abs(patch_color[0] - expected[0]) < 1e-6
        assert abs(patch_color[1] - expected[1]) < 1e-6
        assert abs(patch_color[2] - expected[2]) < 1e-6
        plt.close()

    def test_body_hatch_applied(self) -> None:
        """Body with hatch='///' produces a patch with get_hatch() == '///'."""
        model = self._make_model(hatch="///")
        ax = plot_model(model)
        assert ax.patches[0].get_hatch() == "///"
        plt.close()

    def test_body_no_color_uses_colormap(self) -> None:
        """Body with color=None still gets a colormap-derived (non-None) facecolor."""
        model = self._make_model(color=None)
        ax = plot_model(model)
        face_color = ax.patches[0].get_facecolor()
        assert face_color is not None
        plt.close()


class TestPolygonCentroid:
    """Tests for the _polygon_centroid helper."""

    def test_centroid_triangle(self) -> None:
        """Triangle centroid equals mean of vertices (analytic result)."""
        # Right triangle: (0,0), (6,0), (0,6)
        vertices = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
        cx, cz = _polygon_centroid(vertices)
        assert abs(cx - 2.0) < 1e-9
        assert abs(cz - 2.0) < 1e-9

    def test_centroid_rectangle(self) -> None:
        """Rectangle centroid equals geometric center."""
        vertices = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
        cx, cz = _polygon_centroid(vertices)
        assert abs(cx - 2.0) < 1e-9
        assert abs(cz - 1.0) < 1e-9

    def test_centroid_degenerate(self) -> None:
        """Collinear (zero-area) polygon falls back to vertex mean."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cx, cz = _polygon_centroid(vertices)
        assert abs(cx - 1.0) < 1e-9
        assert abs(cz - 0.0) < 1e-9


class TestPlotModelLabelFeatures:
    """Tests for new label placement features in plot_model."""

    def _make_model(self, label_loc: list[float] | None = None) -> ForwardModel:
        from forward_model.models import GeologicBody, MagneticField

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            susceptibility=0.05,
            name="Body",
            label_loc=label_loc,
        )
        field = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)
        return ForwardModel(
            bodies=[body],
            field=field,
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )

    def test_plot_model_uses_label_loc(self) -> None:
        """Body with label_loc places a plain Text (no annotation) at that location."""
        from matplotlib.text import Annotation, Text

        model = self._make_model(label_loc=[25.0, 150.0])
        ax = plot_model(model)
        texts = [
            c
            for c in ax.get_children()
            if isinstance(c, Text) and not isinstance(c, Annotation)
        ]
        label_texts = [t for t in texts if "Body" in t.get_text()]
        assert len(label_texts) == 1
        assert abs(label_texts[0].get_position()[0] - 25.0) < 1e-9
        assert abs(label_texts[0].get_position()[1] - 150.0) < 1e-9
        plt.close()

    def test_plot_model_label_offsets(self) -> None:
        """label_offsets dict causes an Annotation to be placed, not a plain Text."""
        from matplotlib.text import Annotation

        model = self._make_model()
        ax = plot_model(model, label_offsets={"Body": (0.0, -60.0)})
        annotations = [c for c in ax.get_children() if isinstance(c, Annotation)]
        assert len(annotations) == 1
        plt.close()

    def test_plot_model_show_label_arrows_global(self) -> None:
        """show_label_arrows=True global flag produces an annotation with an arrow."""
        from matplotlib.text import Annotation

        model = self._make_model()
        ax = plot_model(
            model,
            label_offsets={"Body": (0.0, -60.0)},
            show_label_arrows=True,
        )
        annotations = [c for c in ax.get_children() if isinstance(c, Annotation)]
        assert len(annotations) == 1
        assert annotations[0].arrow_patch is not None
        plt.close()

    def test_plot_model_show_label_arrows_per_body(self) -> None:
        """show_label_arrows=False per-body dict produces annotation with no arrow."""
        from matplotlib.text import Annotation

        model = self._make_model()
        ax = plot_model(
            model,
            label_offsets={"Body": (0.0, -60.0)},
            show_label_arrows={"Body": False},
        )
        annotations = [c for c in ax.get_children() if isinstance(c, Annotation)]
        assert len(annotations) == 1
        assert annotations[0].arrow_patch is None
        plt.close()


class TestPlotModel3D:
    """Tests for plot_model_3d function."""

    def test_returns_figure(self, simple_model: ForwardModel) -> None:
        """Smoke test: plot_model_3d returns a Figure without errors."""
        from matplotlib.figure import Figure

        fig = plot_model_3d(simple_model)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_2_5d(self, model_2_5d: ForwardModel) -> None:
        """Smoke test: 2.5D model renders without errors."""
        from matplotlib.figure import Figure

        fig = plot_model_3d(model_2_5d)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_2_75d(self, model_2_75d: ForwardModel) -> None:
        """Smoke test: 2.75D model renders without errors."""
        from matplotlib.figure import Figure

        fig = plot_model_3d(model_2_75d)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_default_strike_fallback(self, simple_model: ForwardModel) -> None:
        """2D body (no strike fields) renders using default_strike without errors."""
        from matplotlib.figure import Figure

        fig = plot_model_3d(simple_model, default_strike=20_000.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_default_strike(self, simple_model: ForwardModel) -> None:
        """Different default_strike values produce a valid Figure."""
        from matplotlib.figure import Figure

        fig = plot_model_3d(simple_model, default_strike=500.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_accepts_existing_ax(self, simple_model: ForwardModel) -> None:
        """Passing an existing Axes3D reuses it and returns its figure."""
        from mpl_toolkits.mplot3d import Axes3D

        fig_pre = plt.figure()
        ax3d = fig_pre.add_subplot(111, projection="3d")
        assert isinstance(ax3d, Axes3D)
        fig_result = plot_model_3d(simple_model, ax=ax3d)
        assert fig_result is fig_pre
        plt.close(fig_pre)

    def test_viewing_angle(self, simple_model: ForwardModel) -> None:
        """Custom elev/azim are applied to the 3D axes."""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plot_model_3d(simple_model, elev=30.0, azim=-45.0)
        ax3d = fig.axes[0]
        assert isinstance(ax3d, Axes3D)
        assert abs(ax3d.elev - 30.0) < 1e-6
        assert abs(ax3d.azim - (-45.0)) < 1e-6
        plt.close(fig)
