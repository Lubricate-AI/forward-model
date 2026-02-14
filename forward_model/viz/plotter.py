"""Matplotlib-based visualization for models and anomalies."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from numpy.typing import NDArray

from forward_model.models.model import ForwardModel


def plot_model(model: ForwardModel, ax: Axes | None = None) -> Axes:
    """Plot geologic cross-section of the forward model.

    Creates a 2D cross-section showing the geologic bodies and
    observation points.

    Args:
        model: The forward model to visualize.
        ax: Matplotlib axes to plot on. If None, creates new axes.

    Returns:
        The matplotlib Axes object containing the plot.

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_model(model, ax=ax)
        >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    # Color palette for bodies
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(model.bodies), 10)))  # type: ignore

    # Plot each body
    for i, body in enumerate(model.bodies):
        vertices = body.to_numpy()
        color = colors[i % len(colors)]

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
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()

    return ax


def plot_anomaly(
    observation_x: list[float], anomaly: NDArray[np.float64], ax: Axes | None = None
) -> Axes:
    """Plot magnetic anomaly profile.

    Creates a line plot showing the magnetic anomaly as a function
    of position along the profile.

    Args:
        observation_x: X-coordinates of observation points (meters).
        anomaly: Magnetic anomaly values (nanoTesla).
        ax: Matplotlib axes to plot on. If None, creates new axes.

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

    return ax


def plot_combined(
    model: ForwardModel,
    anomaly: NDArray[np.float64],
    save_path: str | Path | None = None,
) -> Figure:
    """Create combined plot with cross-section and anomaly profile.

    Creates a two-panel figure with the geologic cross-section on top
    and the magnetic anomaly profile below, with aligned x-axes.

    Args:
        model: The forward model to visualize.
        anomaly: Magnetic anomaly values (nanoTesla).
        save_path: Optional path to save the figure. If None, does not save.

    Returns:
        The matplotlib Figure object.

    Example:
        >>> fig = plot_combined(model, anomaly, save_path="output.png")
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.15}
    )

    # Plot cross-section on top
    plot_model(model, ax=ax1)

    # Plot anomaly below
    plot_anomaly(model.observation_x, anomaly, ax=ax2)

    # Adjust layout
    fig.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
