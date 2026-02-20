"""Matplotlib-based visualization for models and anomalies."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false

from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from forward_model.models.body import GeologicBody
from forward_model.models.model import ForwardModel


def _polygon_centroid(vertices: NDArray[np.float64]) -> tuple[float, float]:
    """Compute area-weighted centroid via shoelace formula.

    Falls back to vertex mean for degenerate (zero-area) polygons.
    """
    x, z = vertices[:, 0], vertices[:, 1]
    cross = x[:-1] * z[1:] - x[1:] * z[:-1]
    cross_close = x[-1] * z[0] - x[0] * z[-1]
    area2 = np.sum(cross) + cross_close  # 2 * signed area
    if abs(area2) < 1e-10:
        return float(np.mean(x)), float(np.mean(z))
    cx = (np.sum((x[:-1] + x[1:]) * cross) + (x[-1] + x[0]) * cross_close) / (3 * area2)
    cz = (np.sum((z[:-1] + z[1:]) * cross) + (z[-1] + z[0]) * cross_close) / (3 * area2)
    return float(cx), float(cz)


def _clamp_to_limits(
    x: float,
    z: float,
    xlim: tuple[float, float] | None,
    zlim: tuple[float, float] | None,
) -> tuple[float, float]:
    """Clamp (x, z) to the given axis limits."""
    if xlim is not None:
        x = max(min(xlim), min(max(xlim), x))
    if zlim is not None:
        z = max(min(zlim), min(max(zlim), z))
    return x, z


def _resolve_strike_extents(
    body: GeologicBody, default_strike: float
) -> tuple[float, float]:
    """Return (y_back, y_front) extrusion limits for a body.

    - 2.75D asymmetric: uses strike_backward / strike_forward fields
    - 2.5D symmetric: uses strike_half_length field
    - 2D infinite: splits default_strike symmetrically
    """
    if body.strike_forward is not None and body.strike_backward is not None:
        return -body.strike_backward, body.strike_forward
    if body.strike_half_length is not None:
        return -body.strike_half_length, body.strike_half_length
    return -default_strike / 2.0, default_strike / 2.0


def plot_model(
    model: ForwardModel,
    ax: Axes | None = None,
    color_by: Literal["index", "susceptibility"] = "susceptibility",
    show_observation_lines: bool = True,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_colorbar: bool = True,
    equal_aspect: bool = True,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    show_label_arrows: bool | dict[str, bool] = False,
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
        label_offsets: Optional mapping of body name to (dx, dz) offset from the
                      computed label anchor. The label text is placed at
                      anchor + (dx, dz); optional arrows from the text to the
                      centroid/label_loc are controlled by ``show_label_arrows``.
        show_label_arrows: If True, draw arrows for all offset labels. Can also
                          be a dict mapping body name to bool for per-body control.

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
        face_color = body.color if body.color is not None else color
        poly = Polygon(
            vertices,
            closed=True,
            facecolor=face_color,
            edgecolor="black",
            alpha=0.6,
            linewidth=1.5,
            hatch=body.hatch,
        )
        ax.add_patch(poly)

        # Determine label anchor position
        if body.label_loc is not None:
            lx, lz = body.label_loc[0], body.label_loc[1]
        else:
            lx, lz = _polygon_centroid(vertices)
            lx, lz = _clamp_to_limits(lx, lz, xlim, zlim)

        label = f"{body.name}\n(χ={body.susceptibility:.3f})"

        if label_offsets and body.name in label_offsets:
            dx, dz = label_offsets[body.name]
            if isinstance(show_label_arrows, dict):
                show_arrow = show_label_arrows.get(body.name, False)
            else:
                show_arrow = show_label_arrows
            arrowprops = {"arrowstyle": "->", "color": "black"} if show_arrow else None
            ax.annotate(
                label,
                xy=(lx, lz),
                xytext=(lx + dx, lz + dz),
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
                arrowprops=arrowprops,
            )
        else:
            ax.text(
                lx,
                lz,
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


def plot_model_3d(
    model: ForwardModel,
    default_strike: float = 10_000.0,
    ax: Axes3D | None = None,
    elev: float = 25.0,
    azim: float = -60.0,
    alpha: float = 0.7,
    color_by: Literal["index", "susceptibility"] = "susceptibility",
) -> Figure:
    """Plot a 3D extruded view of the geologic bodies in the model.

    Each body's polygon is extruded along the y-axis (strike direction) using
    the body's own strike fields to determine extent, providing a 3D view of
    the 2D/2.5D/2.75D cross-sections.

    Args:
        model: The forward model to visualize.
        default_strike: Total strike length (m) used when a body has no strike
                       fields set (2D infinite-strike bodies). Split symmetrically
                       as ±default_strike/2. Default: 10 000 m.
        ax: 3D axes to plot on. If None, creates a new figure and 3D axes.
        elev: Elevation angle (degrees) for the 3D viewing angle.
        azim: Azimuth angle (degrees) for the 3D viewing angle.
        alpha: Transparency of polygon faces.
        color_by: How to color bodies. "index" uses different colors for each
                 body, "susceptibility" uses a colormap based on susceptibility.

    Returns:
        The matplotlib Figure object.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = cast(Figure, ax.figure)

    # Determine colors based on color_by parameter
    cmap = plt.cm.viridis  # type: ignore
    if color_by == "susceptibility":
        susc_values = [body.susceptibility for body in model.bodies]
        susc_set = set(susc_values)
        if len(susc_set) == 1:
            colors = [cmap(0.5)] * len(model.bodies)
        else:
            norm = plt.Normalize(vmin=min(susc_values), vmax=max(susc_values))  # type: ignore
            colors = [cmap(norm(body.susceptibility)) for body in model.bodies]  # type: ignore
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(model.bodies), 10)))  # type: ignore

    # Plot each body as an extruded 3D polygon
    for i, body in enumerate(model.bodies):
        verts = body.to_numpy()  # shape (N, 2) — columns are [x, z]
        n = len(verts)
        color = colors[i % len(colors)] if color_by == "index" else colors[i]
        face_color = body.color if body.color is not None else color

        y_back, y_front = _resolve_strike_extents(body, default_strike)

        faces: list[list[tuple[float, float, float]]] = []
        # Front face (y = y_front)
        faces.append([(float(x), y_front, float(z)) for x, z in verts])
        # Back face (y = y_back)
        faces.append([(float(x), y_back, float(z)) for x, z in verts])
        # Side walls — one quad per edge
        for j in range(n):
            j1 = (j + 1) % n
            xi, zi = float(verts[j, 0]), float(verts[j, 1])
            xi1, zi1 = float(verts[j1, 0]), float(verts[j1, 1])
            faces.append(
                [
                    (xi, y_back, zi),
                    (xi1, y_back, zi1),
                    (xi1, y_front, zi1),
                    (xi, y_front, zi),
                ]
            )

        poly = Poly3DCollection(
            faces, alpha=alpha, facecolor=face_color, edgecolor="k", linewidth=0.5
        )
        ax.add_collection3d(poly)

    # Plot observation line in the profile plane (y = 0)
    obs_x = list(model.observation_x)
    obs_z = model.observation_z
    ax.plot3D(
        obs_x,
        [0.0] * len(obs_x),
        [obs_z] * len(obs_x),
        "ro-",
        markersize=4,
        label="Observation points",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y / Strike (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("3D Cross-Section View", fontsize=13, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)
    ax.invert_zaxis()
    ax.legend(loc="upper left")

    return fig


_COMPONENT_LABELS: dict[str, tuple[str, str]] = {
    "bz": ("Bz (nT)", "Vertical Component (Bz)"),
    "bx": ("Bx (nT)", "Horizontal Component (Bx)"),
    "total_field": ("ΔT (nT)", "Total Field Anomaly (ΔT)"),
    "amplitude": ("|ΔB| (nT)", "Anomaly Amplitude (|ΔB|)"),
    "gradient": ("d(ΔT)/dx (nT/m)", "Horizontal Gradient d(ΔT)/dx"),
}


def plot_anomaly(
    observation_x: list[float],
    anomaly: NDArray[np.float64],
    ax: Axes | None = None,
    xlim: tuple[float, float] | None = None,
    component: str = "total_field",
    gradient: NDArray[np.float64] | None = None,
) -> Axes:
    """Plot magnetic anomaly profile.

    Creates a line plot showing the magnetic anomaly as a function
    of position along the profile. When ``gradient`` is supplied, the
    horizontal gradient d(ΔT)/dx is overlaid on a secondary y-axis on
    the right side of the plot.

    Args:
        observation_x: X-coordinates of observation points (meters).
        anomaly: Magnetic anomaly values (nanoTesla).
        ax: Matplotlib axes to plot on. If None, creates new axes.
        xlim: Optional (min, max) x-axis limits in meters.
        component: Which component is being plotted. Controls axis labels.
                   One of ``"bz"``, ``"bx"``, ``"total_field"``, ``"amplitude"``,
                   ``"gradient"``.
        gradient: Optional d(ΔT)/dx values (nT/m). When provided, overlaid
                  on a twin y-axis with an orange dashed line.

    Returns:
        The matplotlib Axes object containing the plot.

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_anomaly(model.observation_x, anomaly, ax=ax)
        >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    ylabel, title = _COMPONENT_LABELS.get(
        component, ("Anomaly (nT)", "Magnetic Anomaly Profile")
    )

    # Plot anomaly
    ax.plot(observation_x, anomaly, "b-", linewidth=2, label=ylabel)

    # Add zero reference line
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Configure primary axes
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(*xlim)

    # Overlay gradient on secondary y-axis
    if gradient is not None:
        ax2 = ax.twinx()
        ax2.plot(
            observation_x,
            gradient,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="d(ΔT)/dx (nT/m)",
        )
        ax2.set_ylabel("d(ΔT)/dx (nT/m)", fontsize=11, color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        # Combined legend from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

    return ax


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
    label_offsets: dict[str, tuple[float, float]] | None = None,
    show_label_arrows: bool | dict[str, bool] = False,
    component: str = "total_field",
    gradient: NDArray[np.float64] | None = None,
    show_3d: bool = False,
    default_strike: float = 10_000.0,
) -> Figure:
    """Create combined plot with cross-section and anomaly profile.

    Creates a two-panel figure with the geologic cross-section on top
    and the magnetic anomaly profile below, with aligned x-axes. When
    ``gradient`` is supplied, d(ΔT)/dx is overlaid on a secondary y-axis
    in the anomaly panel. When ``show_3d`` is True, a third panel with a
    3D extruded view is added below the anomaly panel.

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
        label_offsets: Optional mapping of body name to (dx, dz) offset from the
                      computed label anchor. Passed through to ``plot_model``.
        show_label_arrows: If True or per-body dict, draw arrows from text to centroid.
        component: Which anomaly component is being plotted. Controls axis labels.
                   One of ``"bz"``, ``"bx"``, ``"total_field"``, ``"amplitude"``.
        gradient: Optional d(ΔT)/dx values (nT/m). When provided, overlaid on a
                  secondary y-axis in the anomaly panel.
        show_3d: If True, add a third panel with a 3D extruded view.
        default_strike: Total strike length (m) used for 2D (infinite-strike) bodies
                       when ``show_3d=True``. Passed through to ``plot_model_3d``.

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

        ax3d: Axes3D | None = None
        if show_3d:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 1, hspace=0.15)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3d = fig.add_subplot(gs[2], projection="3d")
        else:
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
            label_offsets=label_offsets,
            show_label_arrows=show_label_arrows,
        )

        # Plot anomaly below
        plot_anomaly(
            model.observation_x,
            anomaly,
            ax=ax2,
            xlim=xlim,
            component=component,
            gradient=gradient,
        )

        # Plot 3D view if requested
        if show_3d:
            plot_model_3d(model, default_strike=default_strike, ax=ax3d)

        if xlim is not None:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)

        if show_3d and ax3d is not None:
            if xlim is not None:
                ax3d.set_xlim(xlim)
            if zlim is not None:
                ax3d.set_zlim(zlim)

        # Adjust layout (tight_layout is incompatible with 3D axes)
        if not show_3d:
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
