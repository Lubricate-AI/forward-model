"""Talwani (1959) algorithm for 2D gravity anomalies."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.geometry import (
    edge_geometry_2_5d,
    edge_geometry_2_75d,
    edge_geometry_2d,
)
from forward_model.models.gravity_model import GravityModel


@dataclass
class GravityComponents:
    """Gravity anomaly components at each observation point.

    Attributes:
        gz: Vertical gravity anomaly (mGal).
        gz_gradient: Horizontal gradient of the vertical anomaly (mGal/m).
    """

    gz: NDArray[np.float64]
    gz_gradient: NDArray[np.float64]


class PolygonGravityComponents:
    """Per-polygon gravity contribution before superposition."""

    def __init__(self, gz: NDArray[np.float64]):
        self.gz = gz


# Gravitational constant in SI units: m³/(kg·s²)
_G = 6.674e-11


def _apply_gravity_kernel(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    density_contrast: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute vertical gravity anomaly from a 2D polygon using Talwani algorithm.

    Implements the Talwani (1959) algorithm for computing the vertical (gz)
    component of the gravity anomaly from a 2D polygonal body.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        density_contrast: Density contrast in kg/m³.
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        Vertical gravity anomaly in mGal at each observation point.
    """
    n_obs = len(observation_points)
    gz = np.zeros(n_obs, dtype=np.float64)

    # Conversion factor: 2 * G (SI) * 1e5 (converts m/s² to mGal)
    # 2 * G * 1e5 = 2 * 6.674e-11 * 1e5 = 1.3348e-5
    scale_factor = 2.0 * _G * 1e5

    for tx, tz, dtheta, log_term, valid in edge_geometry_2d(
        vertices, observation_points, min_distance
    ):
        # gz contribution: -tx * dtheta - log_term * tz
        # (following the vertical component form from gravity Talwani algorithm)
        gz += np.where(
            valid,
            (-tx * dtheta - log_term * tz),
            0.0,
        )

    gz *= scale_factor * density_contrast
    return gz


def _apply_gravity_kernel_2_5d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    density_contrast: float,
    strike_half_length: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """2.5D gravity anomaly with finite symmetric strike extent (Won & Bevis 1987).

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        density_contrast: Density contrast in kg/m³.
        strike_half_length: Half-length of the body in the strike direction (m).
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        Vertical gravity anomaly in mGal at each observation point, attenuated
        for finite strike extent.
    """
    n_obs = len(observation_points)
    gz = np.zeros(n_obs, dtype=np.float64)

    scale_factor = 2.0 * _G * 1e5

    for tx, tz, dtheta, dlambda, valid in edge_geometry_2_5d(
        vertices, observation_points, strike_half_length, min_distance
    ):
        gz += np.where(
            valid,
            (-tx * dtheta - dlambda * tz),
            0.0,
        )

    gz *= scale_factor * density_contrast
    return gz


def _apply_gravity_kernel_2_75d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    density_contrast: float,
    strike_forward: float,
    strike_backward: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """2.75D gravity anomaly: finite asymmetric strike extent (Won & Bevis 1987).

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        density_contrast: Density contrast in kg/m³.
        strike_forward: Forward (+y) half-extent in meters.
        strike_backward: Backward (−y) half-extent in meters.
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        Vertical gravity anomaly in mGal at each observation point.
    """
    n_obs = len(observation_points)
    gz = np.zeros(n_obs, dtype=np.float64)

    scale_factor = 2.0 * _G * 1e5

    for tx, tz, dtheta, dlambda, valid in edge_geometry_2_75d(
        vertices, observation_points, strike_forward, strike_backward, min_distance
    ):
        gz += np.where(
            valid,
            (-tx * dtheta - dlambda * tz),
            0.0,
        )

    gz *= scale_factor * density_contrast
    return gz


_worker_obs_points: NDArray[np.float64] | None = None


def _init_worker(obs_points: NDArray[np.float64]) -> None:
    """Populate worker-process global with the shared observation grid."""
    global _worker_obs_points
    _worker_obs_points = obs_points


def _compute_single_body(
    args: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float | None,
        float | None,
        float | None,
    ],
) -> NDArray[np.float64]:
    """Compute gravity anomaly for a single body. Module-level for pickling."""
    vertices, observation_points, density_contrast, shl, sf, sb = args
    if sf is not None:
        if sb is None:  # pragma: no cover
            raise ValueError("strike_backward must be set when strike_forward is set")
        return _apply_gravity_kernel_2_75d(
            vertices,
            observation_points,
            density_contrast,
            sf,
            sb,
        )
    if shl is not None:
        return _apply_gravity_kernel_2_5d(
            vertices, observation_points, density_contrast, shl
        )
    return _apply_gravity_kernel(vertices, observation_points, density_contrast)


def _compute_body_parallel(
    args: tuple[
        NDArray[np.float64],
        float,
        float | None,
        float | None,
        float | None,
    ],
) -> NDArray[np.float64]:
    """Compute gravity anomaly for one body using the observation points."""
    if _worker_obs_points is None:  # pragma: no cover
        raise RuntimeError("Worker process not initialized with observation points.")
    vertices, density_contrast, shl, sf, sb = args
    if sf is not None:
        if sb is None:  # pragma: no cover
            raise ValueError("strike_backward must be set when strike_forward is set")
        return _apply_gravity_kernel_2_75d(
            vertices,
            _worker_obs_points,
            density_contrast,
            sf,
            sb,
        )
    if shl is not None:
        return _apply_gravity_kernel_2_5d(
            vertices, _worker_obs_points, density_contrast, shl
        )
    return _apply_gravity_kernel(vertices, _worker_obs_points, density_contrast)


def calculate_gravity(model: GravityModel, parallel: bool = False) -> GravityComponents:
    """Calculate total gravity anomaly for a gravity model.

    Computes the gravity anomaly using the Talwani (1959) algorithm,
    summing contributions from all geologic bodies via superposition.

    Args:
        model: Complete gravity model specification including bodies and
               observation points.
        parallel: If True, compute each body's anomaly in a separate process
                  using ProcessPoolExecutor. Useful when the model has many
                  bodies and observation grids are large.

    Returns:
        GravityComponents containing:
        - gz: Vertical gravity anomaly in mGal at each observation point
        - gz_gradient: Horizontal gradient of gz in mGal/m

    Raises:
        ValueError: If any body lacks gravity properties.
    """
    observation_points = model.get_observation_points()

    per_body: list[
        tuple[
            NDArray[np.float64],
            float,
            float | None,
            float | None,
            float | None,
        ]
    ] = []
    for body in model.bodies:
        if body.gravity is None:
            raise ValueError(
                f"Body '{body.name}' has no gravity properties; "
                "gravity calculation requires gravity to be set"
            )
        per_body.append(
            (
                body.to_numpy(),
                body.gravity.density_contrast,
                body.strike_half_length,
                body.strike_forward,
                body.strike_backward,
            )
        )

    if parallel:
        with ProcessPoolExecutor(
            initializer=_init_worker, initargs=(observation_points,)
        ) as executor:
            body_components = list(executor.map(_compute_body_parallel, per_body))
    else:
        body_components = [
            _compute_single_body((v, observation_points, dc, shl, sf, sb))
            for v, dc, shl, sf, sb in per_body
        ]

    # Sum gz via superposition
    total_gz: NDArray[np.float64] = np.sum(body_components, axis=0)

    # Compute horizontal gradient
    obs_x: NDArray[np.float64] = observation_points[:, 0]
    if len(obs_x) > 1:
        gz_gradient: NDArray[np.float64] = np.gradient(total_gz, obs_x)
    else:
        # For single observation point, gradient is undefined; return zero
        gz_gradient = np.zeros_like(total_gz)

    return GravityComponents(gz=total_gz, gz_gradient=gz_gradient)
