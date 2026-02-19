"""Talwani (1965) algorithm for 2D magnetic anomalies."""

import numpy as np
from numpy.typing import NDArray


def field_to_magnetization(
    susceptibility: float,
    field_intensity: float,
    field_inclination: float,
    field_declination: float,
    remanent_intensity: float = 0.0,
    remanent_inclination: float = 0.0,
    remanent_declination: float = 0.0,
) -> NDArray[np.float64]:
    """Convert magnetic field parameters to 2D magnetization vector.

    Computes the total magnetization as the sum of the induced component
    (from susceptibility and the ambient field) and the remanent component
    (a permanent vector fixed at rock formation time).

    Args:
        susceptibility: Magnetic susceptibility (SI units, dimensionless).
        field_intensity: Inducing field intensity in nanoTesla (nT).
        field_inclination: Field inclination in degrees (-90 to 90).
        field_declination: Field declination in degrees (-180 to 180).
        remanent_intensity: Remanent magnetization intensity in A/m. Default 0.0.
        remanent_inclination: Inclination of remanent vector in degrees. Default 0.0.
        remanent_declination: Declination of remanent vector in degrees. Default 0.0.

    Returns:
        2D magnetization vector [Mx, Mz] in A/m representing total magnetization
        (induced + remanent) projected into the vertical profile plane.
    """
    # Convert nT to Tesla
    field_T = field_intensity * 1e-9

    # Convert to radians
    inc_rad = np.deg2rad(field_inclination)
    dec_rad = np.deg2rad(field_declination)

    # Magnetic field vector components (in Tesla)
    # For 2D modeling, we use the component in the profile plane
    # Assuming profile is oriented N-S (declination rotates field in horizontal)
    Bx = field_T * np.cos(inc_rad) * np.cos(dec_rad)
    Bz = -field_T * np.sin(inc_rad)  # Negative because z is depth (positive down)

    # Induced magnetization M = χ * H = χ * B/μ₀
    # μ₀ = 4π × 10⁻⁷ H/m
    mu_0 = 4.0 * np.pi * 1e-7
    Mx = susceptibility * Bx / mu_0
    Mz = susceptibility * Bz / mu_0

    # Remanent component — already in A/m, project into 2D profile plane
    inc_rem = np.deg2rad(remanent_inclination)
    dec_rem = np.deg2rad(remanent_declination)
    Mx_rem = remanent_intensity * np.cos(inc_rem) * np.cos(dec_rem)
    Mz_rem = -remanent_intensity * np.sin(inc_rem)  # negative: z positive-down

    return np.array([Mx + Mx_rem, Mz + Mz_rem], dtype=np.float64)


def compute_polygon_anomaly(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    magnetization: NDArray[np.float64],
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute magnetic anomaly from a 2D polygon using Talwani algorithm.

    Implements the Talwani (1965) algorithm for computing the vertical
    component of the magnetic anomaly from a 2D polygonal body.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        magnetization: 2D magnetization vector [Mx, Mz] in A/m.
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        Array of length M containing the vertical magnetic anomaly in nT
        at each observation point.

    References:
        Talwani, M., and Heirtzler, J. R. (1965). Computation of magnetic
        anomalies caused by two-dimensional structures of arbitrary shape.
    """
    n_vertices = len(vertices)
    n_obs = len(observation_points)
    anomaly = np.zeros(n_obs, dtype=np.float64)

    # Magnetic constant: μ₀/(4π) in SI units
    # Convert to nT: multiply by 10⁹
    mu_0_4pi = 1e-7  # T·m/A in SI
    mu_0_4pi_nT = mu_0_4pi * 1e9  # nT·m/A

    Mx, Mz = magnetization

    obs_x = observation_points[:, 0]  # (M,)
    obs_z = observation_points[:, 1]  # (M,)

    # Loop over polygon edges (N iterations, each vectorized over M obs points)
    for j in range(n_vertices):
        j_next = (j + 1) % n_vertices

        # Edge geometry — scalars, same for all observation points
        dx = vertices[j_next, 0] - vertices[j, 0]
        dz = vertices[j_next, 1] - vertices[j, 1]
        edge_length = np.sqrt(dx**2 + dz**2)

        # Restore original L∞ predicate for backward-compatible degenerate-edge skipping
        if np.abs(dx) < min_distance and np.abs(dz) < min_distance:
            continue

        tx = dx / edge_length
        tz = dz / edge_length

        # Vectors from each observation point to each edge vertex — (M,) arrays
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

        # Guard against division-by-zero and log(0) using masked ufuncs so
        # NumPy never evaluates r2/r1 or log(...) for invalid points.
        r_ratio = np.empty_like(r1)
        np.divide(r2, r1, out=r_ratio, where=valid)
        log_term = np.zeros_like(r1)
        np.log(r_ratio, out=log_term, where=valid)

        contrib = np.where(
            valid,
            Mx * (dtheta * tz - log_term * tx) + Mz * (-dtheta * tx - log_term * tz),
            0.0,
        )
        anomaly += contrib

    anomaly *= mu_0_4pi_nT
    return anomaly
