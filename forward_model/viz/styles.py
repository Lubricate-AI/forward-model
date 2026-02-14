"""Plot style configurations for visualizations."""

from typing import Any

# Plot style configurations
PLOT_STYLES: dict[str, dict[str, Any]] = {
    "default": {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
    },
    "publication": {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    },
    "presentation": {
        "font.size": 14,
        "font.family": "sans-serif",
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 3,
        "axes.linewidth": 2,
    },
}


def get_style(style_name: str = "default") -> dict[str, Any]:
    """Get plot style configuration by name.

    Args:
        style_name: Name of the style ("default", "publication", "presentation").

    Returns:
        Dictionary of matplotlib rcParams for the specified style.

    Raises:
        ValueError: If style_name is not recognized.

    Example:
        >>> style = get_style("publication")
        >>> with plt.rc_context(style):
        ...     plt.plot([1, 2, 3])
    """
    if style_name not in PLOT_STYLES:
        raise ValueError(
            f"Unknown style: {style_name}. "
            f"Available styles: {', '.join(PLOT_STYLES.keys())}"
        )
    return PLOT_STYLES[style_name]
