"""2D Talwani-style algorithm for heat flow anomalies."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.geometry import (
    edge_geometry_2_5d,
    edge_geometry_2_75d,
    edge_geometry_2d,
)
from forward_model.models.heatflow_model import HeatFlowModel


@dataclass
class HeatFlowComponents:
    """Heat flow anomaly components at each observation point.

    Attributes:
        heat_flow: Vertical heat flow perturbation (mW/m²).
        heat_flow_x: Horizontal heat flow perturbation (mW/m²).
        heat_flow_gradient: Horizontal gradient of vertical heat flow (mW/m³).
    """

    heat_flow: NDArray[np.float64]
    heat_flow_x: NDArray[np.float64]
    heat_flow_gradient: NDArray[np.float64]


def _apply_heatflow_kernel(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    conductivity_contrast: float,
    background_heat_flow: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute vertical heat flow perturbation from a 2D polygon.

    Implements the 2D Talwani-style algorithm for heat flow, directly analogous
    to the Talwani (1959) gravity algorithm. The governing equation is the 2D
    steady-state heat conduction (Laplace) equation.

    The perturbation to vertical heat flow at the surface from a 2D body
    with conductivity contrast Δk in a background heat flow q₀ is:

        δqz(x) = (q₀ / π) * Δk * Σ_edges [-tx·dθ - tz·ln(r₂/r₁)]

    The 1/π factor comes from the half-space Green's function for the 2D
    Laplace equation (Benfield 1949 analog of Talwani 1959).

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        conductivity_contrast: Thermal conductivity contrast Δk (W/m·K).
            Treated directly as the contrast value, analogous to density_contrast
            in the gravity case.
        background_heat_flow: Background vertical heat flow q₀ (mW/m²).
        min_distance: Minimum distance threshold to avoid singularities (m).

    Returns:
        Vertical heat flow perturbation δqz in mW/m² at each observation point.
    """
    n_obs = len(observation_points)
    qz = np.zeros(n_obs, dtype=np.float64)

    # Scale factor: (q₀ / π) gives mW/m² when multiplied by Δk [W/(m·K)]
    # and the dimensionless edge-geometry sum. The 1/π normalisation comes from
    # the 2D half-space Green's function, directly analogous to the 2G factor
    # in the Talwani gravity algorithm.
    scale_factor = background_heat_flow / np.pi

    for tx, tz, dtheta, log_term, valid in edge_geometry_2d(
        vertices, observation_points, min_distance
    ):
        qz += np.where(valid, (-tx * dtheta - log_term * tz), 0.0)

    qz *= scale_factor * conductivity_contrast
    return qz


def _apply_heatflow_kernel_x(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    conductivity_contrast: float,
    background_heat_flow: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute horizontal heat flow perturbation from a 2D polygon.

    Computes the horizontal (x) component using the complementary kernel terms
    from the same edge geometry, exactly as magnetic bx relates to bz:

        δqx(x) = (q₀ / π) * Δk * Σ_edges [tz·dθ - tx·ln(r₂/r₁)]

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        conductivity_contrast: Thermal conductivity contrast Δk (W/m·K).
        background_heat_flow: Background vertical heat flow q₀ (mW/m²).
        min_distance: Minimum distance threshold to avoid singularities (m).

    Returns:
        Horizontal heat flow perturbation δqx in mW/m² at each observation point.
    """
    n_obs = len(observation_points)
    qx = np.zeros(n_obs, dtype=np.float64)

    scale_factor = background_heat_flow / np.pi

    for tx, tz, dtheta, log_term, valid in edge_geometry_2d(
        vertices, observation_points, min_distance
    ):
        qx += np.where(valid, (tz * dtheta - tx * log_term), 0.0)

    qx *= scale_factor * conductivity_contrast
    return qx


def _apply_heatflow_kernel_2_5d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    conductivity_contrast: float,
    background_heat_flow: float,
    strike_half_length: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """2.5D heat flow with finite symmetric strike extent (Won & Bevis 1987).

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        conductivity_contrast: Thermal conductivity contrast Δk (W/m·K).
        background_heat_flow: Background vertical heat flow q₀ (mW/m²).
        strike_half_length: Half-length of the body in the strike direction (m).
        min_distance: Minimum distance threshold to avoid singularities (m).

    Returns:
        Vertical heat flow perturbation in mW/m², attenuated for finite strike.
    """
    n_obs = len(observation_points)
    qz = np.zeros(n_obs, dtype=np.float64)

    scale_factor = background_heat_flow / np.pi

    for tx, tz, dtheta, dlambda, valid in edge_geometry_2_5d(
        vertices, observation_points, strike_half_length, min_distance
    ):
        qz += np.where(valid, (-tx * dtheta - dlambda * tz), 0.0)

    qz *= scale_factor * conductivity_contrast
    return qz


def _apply_heatflow_kernel_2_75d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    conductivity_contrast: float,
    background_heat_flow: float,
    strike_forward: float,
    strike_backward: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """2.75D heat flow: finite asymmetric strike extent (Won & Bevis 1987).

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        conductivity_contrast: Thermal conductivity contrast Δk (W/m·K).
        background_heat_flow: Background vertical heat flow q₀ (mW/m²).
        strike_forward: Forward (+y) half-extent in meters.
        strike_backward: Backward (−y) half-extent in meters.
        min_distance: Minimum distance threshold to avoid singularities (m).

    Returns:
        Vertical heat flow perturbation in mW/m² at each observation point.
    """
    n_obs = len(observation_points)
    qz = np.zeros(n_obs, dtype=np.float64)

    scale_factor = background_heat_flow / np.pi

    for tx, tz, dtheta, dlambda, valid in edge_geometry_2_75d(
        vertices, observation_points, strike_forward, strike_backward, min_distance
    ):
        qz += np.where(valid, (-tx * dtheta - dlambda * tz), 0.0)

    qz *= scale_factor * conductivity_contrast
    return qz
