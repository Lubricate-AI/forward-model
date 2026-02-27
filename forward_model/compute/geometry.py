"""Private edge-geometry kernels shared by all Talwani variants."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def edge_geometry_2d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    min_distance: float = 1e-10,
) -> Iterator[
    tuple[
        float,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]
]:
    """Per-edge geometry for the 2D Talwani algorithm.

    For each non-degenerate polygon edge, yields the edge tangent unit vector
    and per-observation-point angular and log-ratio terms.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        min_distance: Minimum distance threshold for singularity avoidance.

    Yields:
        (tx, tz, dtheta, log_ratio, valid) for each non-degenerate edge:
        - tx, tz: Edge tangent unit vector components (scalars).
        - dtheta: arctan2(z2,x2) − arctan2(z1,x1), wrapped to [−π, π], shape (M,).
        - log_ratio: log(r2/r1) masked at invalid points, shape (M,).
        - valid: Boolean mask (r1 >= min_distance) & (r2 >= min_distance), shape (M,).
    """
    n_vertices = len(vertices)
    obs_x = observation_points[:, 0]
    obs_z = observation_points[:, 1]

    for j in range(n_vertices):
        j_next = (j + 1) % n_vertices

        dx = vertices[j_next, 0] - vertices[j, 0]
        dz = vertices[j_next, 1] - vertices[j, 1]

        if np.abs(dx) < min_distance and np.abs(dz) < min_distance:
            continue

        edge_length = np.sqrt(dx**2 + dz**2)
        tx = float(dx / edge_length)
        tz = float(dz / edge_length)

        x1 = vertices[j, 0] - obs_x
        z1 = vertices[j, 1] - obs_z
        x2 = vertices[j_next, 0] - obs_x
        z2 = vertices[j_next, 1] - obs_z

        r1 = np.sqrt(x1**2 + z1**2)
        r2 = np.sqrt(x2**2 + z2**2)
        valid = (r1 >= min_distance) & (r2 >= min_distance)

        theta1: NDArray[np.float64] = np.arctan2(z1, x1)
        theta2: NDArray[np.float64] = np.arctan2(z2, x2)
        dtheta: NDArray[np.float64] = theta2 - theta1
        dtheta = np.where(dtheta > np.pi, dtheta - 2 * np.pi, dtheta)
        dtheta = np.where(dtheta < -np.pi, dtheta + 2 * np.pi, dtheta)

        r_ratio = np.empty_like(r1)
        np.divide(r2, r1, out=r_ratio, where=valid)
        log_ratio = np.zeros_like(r1)
        np.log(r_ratio, out=log_ratio, where=valid)

        yield tx, tz, dtheta, log_ratio, valid


def edge_geometry_2_5d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    strike_half_length: float,
    min_distance: float = 1e-10,
) -> Iterator[
    tuple[
        float,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]
]:
    """Per-edge geometry for the 2.5D Talwani algorithm (Won & Bevis 1987).

    Replaces the standard 2D vertex functions with Won & Bevis (1987) modified
    functions that account for finite strike extent:

    - Modified angle:  Θk = arctan2(zk·y0, xk·sqrt(rk²+y0²))
    - Modified log:    Λk = log(rk / (sqrt(rk²+y0²) + y0))

    Both reduce to their 2D counterparts as y0 → ∞.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        strike_half_length: Half-length of the body in the strike direction (m).
        min_distance: Minimum distance threshold for singularity avoidance.

    Yields:
        (tx, tz, dtheta, dlambda, valid) for each non-degenerate edge:
        - tx, tz: Edge tangent unit vector components (scalars).
        - dtheta: Θ2 − Θ1 wrapped to [−π, π], shape (M,).
        - dlambda: Λ2 − Λ1 masked at invalid points, shape (M,).
        - valid: Boolean mask (r1 >= min_distance) & (r2 >= min_distance), shape (M,).
    """
    n_vertices = len(vertices)
    obs_x = observation_points[:, 0]
    obs_z = observation_points[:, 1]
    y0 = strike_half_length

    for j in range(n_vertices):
        j_next = (j + 1) % n_vertices

        dx = vertices[j_next, 0] - vertices[j, 0]
        dz = vertices[j_next, 1] - vertices[j, 1]

        if np.abs(dx) < min_distance and np.abs(dz) < min_distance:
            continue

        edge_length = np.sqrt(dx**2 + dz**2)
        tx = float(dx / edge_length)
        tz = float(dz / edge_length)

        x1 = vertices[j, 0] - obs_x
        z1 = vertices[j, 1] - obs_z
        x2 = vertices[j_next, 0] - obs_x
        z2 = vertices[j_next, 1] - obs_z

        r1 = np.sqrt(x1**2 + z1**2)
        r2 = np.sqrt(x2**2 + z2**2)
        valid = (r1 >= min_distance) & (r2 >= min_distance)

        sr1 = np.sqrt(r1**2 + y0**2)
        sr2 = np.sqrt(r2**2 + y0**2)

        theta1: NDArray[np.float64] = np.arctan2(z1 * y0, x1 * sr1)
        theta2: NDArray[np.float64] = np.arctan2(z2 * y0, x2 * sr2)
        dtheta: NDArray[np.float64] = theta2 - theta1
        dtheta = np.where(dtheta > np.pi, dtheta - 2 * np.pi, dtheta)
        dtheta = np.where(dtheta < -np.pi, dtheta + 2 * np.pi, dtheta)

        ratio1 = np.zeros_like(r1)
        np.divide(r1, sr1 + y0, out=ratio1, where=valid)
        lambda1 = np.zeros_like(r1)
        np.log(ratio1, out=lambda1, where=valid)

        ratio2 = np.zeros_like(r2)
        np.divide(r2, sr2 + y0, out=ratio2, where=valid)
        lambda2 = np.zeros_like(r2)
        np.log(ratio2, out=lambda2, where=valid)
        dlambda: NDArray[np.float64] = lambda2 - lambda1

        yield tx, tz, dtheta, dlambda, valid


def edge_geometry_2_75d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    strike_forward: float,
    strike_backward: float,
    min_distance: float = 1e-10,
) -> Iterator[
    tuple[
        float,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]
]:
    """Per-edge geometry for the 2.75D Talwani algorithm (Won & Bevis 1987).

    Superposition of two symmetric 2.5D geometries (forward and backward
    strike extents), yielding the per-edge average of their angular and
    log-ratio terms. Physics linearity guarantees this is equivalent to
    averaging the anomaly contributions.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        strike_forward: Forward (+y) half-extent in meters.
        strike_backward: Backward (−y) half-extent in meters.
        min_distance: Minimum distance threshold for singularity avoidance.

    Yields:
        (tx, tz, dtheta, dlambda, valid) for each non-degenerate edge, where
        dtheta and dlambda are the per-edge averages of the forward and
        backward 2.5D geometry terms.
    """
    for (tx_f, tz_f, dt_f, dl_f, v_f), (_, _, dt_b, dl_b, _) in zip(
        edge_geometry_2_5d(vertices, observation_points, strike_forward, min_distance),
        edge_geometry_2_5d(vertices, observation_points, strike_backward, min_distance),
        strict=True,
    ):
        yield tx_f, tz_f, (dt_f + dt_b) / 2.0, (dl_f + dl_b) / 2.0, v_f
