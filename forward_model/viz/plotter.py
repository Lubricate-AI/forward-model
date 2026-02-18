"""Matplotlib-based visualization for models and anomalies."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from numpy.typing import NDArray

from forward_model.models.model import ForwardModel


def plot_model(
    model: ForwardModel,
    ax: Axes | None = None,
    color_by: Literal["index", "susceptibility"] = "susceptibility",
    show_observation_lines: bool = True,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_colorbar: bool = True,
    equal_aspect: bool = True,
) -> Axes:
    """Plot geologic cross-section of the forward model.

    Creates a 2D cross-section showing the geologic bodies and
    observation points.

    Args:
        model: The forward model to visualize.
        ax: Matplotlib axes to plot on. If None, creates new axes.
        color_by: How to color bodies. "index" uses different colors for each
                 body, "susceptibility" uses a colormap based on susceptibility.
        show_observation_lines: If True, show vertical dashed lines at obs points.
        xlim: Optional (min, max) x-axis limits in meters.
        zlim: Optional (min, max) depth limits in meters (shallow, deep).
        show_colorbar: If True, show colorbar when color_by="susceptibility".
        equal_aspect: If True, lock x and z axes to equal scale.

    Returns:
        The matplotlib Axes object containing the plot.

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_model(model, ax=ax, color_by="susceptibility")
        >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    # Determine colors based on color_by parameter
    cmap = plt.cm.viridis  # type: ignore
    if color_by == "susceptibility":
        # Use susceptibility-based colormap
        susc_values = [body.susceptibility for body in model.bodies]
        susc_set = set(susc_values)

        if len(susc_set) == 1:
            # All susceptibilities are the same - use single color
            colors = [cmap(0.5)] * len(model.bodies)
            _auto_colorbar = False
        else:
            # Multiple susceptibilities - use colormap
            norm = plt.Normalize(vmin=min(susc_values), vmax=max(susc_values))  # type: ignore
            colors = [cmap(norm(body.susceptibility)) for body in model.bodies]  # type: ignore
            _auto_colorbar = True
    else:
        # Use index-based colors
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(model.bodies), 10)))  # type: ignore
        _auto_colorbar = False

    # Plot each body
    for i, body in enumerate(model.bodies):
        vertices = body.to_numpy()
        color = colors[i % len(colors)] if color_by == "index" else colors[i]

        # Create polygon patch
        poly = Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor="black",
            alpha=0.6,
            linewidth=1.5,
        )
        ax.add_patch(poly)

        # Add label
        centroid_x = np.mean(vertices[:, 0])
        centroid_z = np.mean(vertices[:, 1])
        label = f"{body.name}\n(χ={body.susceptibility:.3f})"
        ax.text(
            centroid_x,
            centroid_z,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    # Add colorbar if using susceptibility coloring with multiple values
    if color_by == "susceptibility" and _auto_colorbar and show_colorbar:
        susc_values = [body.susceptibility for body in model.bodies]
        sm = plt.cm.ScalarMappable(  # type: ignore
            cmap=cmap,
            norm=plt.Normalize(vmin=min(susc_values), vmax=max(susc_values)),  # type: ignore
        )
        sm.set_array([])  # type: ignore
        plt.colorbar(sm, ax=ax, label="Susceptibility (SI)")  # type: ignore

    # Plot observation lines
    if show_observation_lines:
        obs_points = model.get_observation_points()
        for x in obs_points[:, 0]:
            ax.axvline(x, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    # Plot observation points
    obs_points = model.get_observation_points()
    ax.plot(
        obs_points[:, 0],
        obs_points[:, 1],
        "ro",
        markersize=6,
        label="Observation points",
        zorder=10,
    )

    # Configure axes
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title("Geologic Cross-Section", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()

    if zlim is not None:
        ax.set_ylim(*zlim)

    if xlim is not None:
        ax.set_xlim(*xlim)

    return ax


def plot_anomaly(
    observation_x: list[float],
    anomaly: NDArray[np.float64],
    ax: Axes | None = None,
    xlim: tuple[float, float] | None = None,
) -> Axes:
    """Plot magnetic anomaly profile.

    Creates a line plot showing the magnetic anomaly as a function
    of position along the profile.

    Args:
        observation_x: X-coordinates of observation points (meters).
        anomaly: Magnetic anomaly values (nanoTesla).
        ax: Matplotlib axes to plot on. If None, creates new axes.
        xlim: Optional (min, max) x-axis limits in meters.

    Returns:
        The matplotlib Axes object containing the plot.

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_anomaly(model.observation_x, anomaly, ax=ax)
        >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    # Plot anomaly
    ax.plot(observation_x, anomaly, "b-", linewidth=2, label="Magnetic anomaly")

    # Add zero reference line
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Configure axes
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Anomaly (nT)", fontsize=11)
    ax.set_title("Magnetic Anomaly Profile", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if xlim is not None:
        ax.set_xlim(*xlim)

    return ax


def plot_gradient(
    observation_x: list[float],
    anomaly: NDArray[np.float64],
    ax: Axes | None = None,
) -> Axes:
    """Plot the horizontal derivative of the magnetic anomaly.

    Computes the first derivative using np.gradient (central differences,
    one-sided at endpoints) normalized by station spacing.

    Args:
        observation_x: X-coordinates of observation points (meters).
        anomaly: Magnetic anomaly values (nanoTesla).
        ax: Optional matplotlib Axes. Creates new axes if None.

    Returns:
        Matplotlib Axes with gradient plotted.

    Example:
        >>> ax = plot_gradient(model.observation_x, anomaly)
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.asarray(observation_x)
    gradient = np.gradient(anomaly, x)

    ax.plot(x, gradient, color="red", linewidth=2, label="dT/dx (nT/m)")  # type: ignore[union-attr]
    ax.set_ylabel("Gradient (nT/m)", color="red")  # type: ignore[union-attr]
    ax.tick_params(axis="y", labelcolor="red")  # type: ignore[union-attr]

    return ax  # type: ignore[return-value]


def plot_combined(
    model: ForwardModel,
    anomaly: NDArray[np.float64],
    save_path: str | Path | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    color_by: Literal["index", "susceptibility"] = "susceptibility",
    show_observation_lines: bool = True,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_colorbar: bool = True,
    show_gradient: bool = False,
) -> Figure:
    """Create combined plot with cross-section and anomaly profile.

    Creates a two-panel figure with the geologic cross-section on top
    and the magnetic anomaly profile below, with aligned x-axes.

    Args:
        model: The forward model to visualize.
        anomaly: Magnetic anomaly values (nanoTesla).
        save_path: Optional path to save the figure. If None, does not save.
        style: Plot style name ("default", "publication", "presentation").
        figsize: Figure size as (width, height) in inches. If None, uses (12, 8).
        dpi: DPI for saved figure. If None, uses style default.
        color_by: How to color bodies in cross-section.
        show_observation_lines: If True, show vertical lines at observation points.
        xlim: Optional (min, max) x-axis limits in meters.
        zlim: Optional (min, max) depth limits in meters (shallow, deep).
        show_colorbar: If True, show colorbar when color_by="susceptibility".
        show_gradient: If True, overlay horizontal magnetic gradient on anomaly panel.

    Returns:
        The matplotlib Figure object.

    Example:
        >>> fig = plot_combined(model, anomaly, save_path="output.png",
        ...                     style="publication", dpi=300)
        >>> plt.show()
    """
    from forward_model.viz.styles import get_style

    # Get style configuration
    style_config = get_style(style)

    # Use context manager to apply style
    with plt.rc_context(style_config):
        # Use provided figsize or default
        if figsize is None:
            figsize = (12, 8)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.15}
        )

        # Plot cross-section on top
        plot_model(
            model,
            ax=ax1,
            color_by=color_by,
            show_observation_lines=show_observation_lines,
            xlim=xlim,
            zlim=zlim,
            show_colorbar=show_colorbar,
            equal_aspect=False,
        )

        # Plot anomaly below
        plot_anomaly(model.observation_x, anomaly, ax=ax2, xlim=xlim)

        if show_gradient:
            ax_grad = ax2.twinx()
            plot_gradient(model.observation_x, anomaly, ax=ax_grad)
            handles1, labels1 = ax2.get_legend_handles_labels()
            handles2, labels2 = ax_grad.get_legend_handles_labels()
            ax2.legend(handles1 + handles2, labels1 + labels2)
            if ax_grad.get_legend():
                ax_grad.get_legend().remove()
            if xlim is not None:
                ax_grad.set_xlim(xlim)

        if xlim is not None:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)

        # Adjust layout
        fig.tight_layout()

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            # Use provided DPI or style default
            save_dpi = dpi if dpi is not None else style_config.get("savefig.dpi", 150)

            # Detect format from extension
            suffix = save_path.suffix.lower()
            if suffix in [".png", ".jpg", ".jpeg"]:
                # Raster formats
                fig.savefig(
                    save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white"
                )
            elif suffix in [".pdf", ".svg", ".eps"]:
                # Vector formats - DPI less relevant but still used for
                # rasterized elements
                fig.savefig(
                    save_path, dpi=save_dpi, bbox_inches="tight", format=suffix[1:]
                )
            else:
                # Default format
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    return fig
