"""Talwani (1965) algorithm for 2D magnetic anomalies."""

import numpy as np
from numpy.typing import NDArray


def field_to_magnetization(
    susceptibility: float,
    field_intensity: float,
    field_inclination: float,
    field_declination: float,
) -> NDArray[np.float64]:
    """Convert magnetic field parameters to 2D magnetization vector.

    Args:
        susceptibility: Magnetic susceptibility (SI units, dimensionless).
        field_intensity: Inducing field intensity in nanoTesla (nT).
        field_inclination: Field inclination in degrees (-90 to 90).
        field_declination: Field declination in degrees (-180 to 180).

    Returns:
        2D magnetization vector [Mx, Mz] in A/m.
        For 2D models, we project the field into the vertical plane.
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

    # Magnetization M = χ * H = χ * B/μ₀
    # μ₀ = 4π × 10⁻⁷ H/m
    mu_0 = 4.0 * np.pi * 1e-7
    Mx = susceptibility * Bx / mu_0
    Mz = susceptibility * Bz / mu_0

    return np.array([Mx, Mz], dtype=np.float64)


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

    # Loop over observation points
    for i, obs in enumerate(observation_points):
        contribution = 0.0

        # Loop over polygon edges
        for j in range(n_vertices):
            j_next = (j + 1) % n_vertices

            # Edge vertices
            x1, z1 = vertices[j] - obs
            x2, z2 = vertices[j_next] - obs

            # Skip degenerate edges
            if np.abs(x2 - x1) < min_distance and np.abs(z2 - z1) < min_distance:
                continue

            # Distances from observation point
            r1 = np.sqrt(x1**2 + z1**2)
            r2 = np.sqrt(x2**2 + z2**2)

            # Skip if too close (singularity)
            if r1 < min_distance or r2 < min_distance:
                continue

            # Angles
            theta1 = np.arctan2(z1, x1)
            theta2 = np.arctan2(z2, x2)

            # Handle angle wrapping for proper integration
            dtheta = theta2 - theta1
            if dtheta > np.pi:
                dtheta -= 2 * np.pi
            elif dtheta < -np.pi:
                dtheta += 2 * np.pi

            # Edge direction vector
            dx = x2 - x1
            dz = z2 - z1
            edge_length = np.sqrt(dx**2 + dz**2)

            if edge_length < min_distance:
                continue

            # Unit tangent along edge
            tx = dx / edge_length
            tz = dz / edge_length

            # Logarithmic term
            if r2 > min_distance and r1 > min_distance:
                log_term = np.log(r2 / r1)
            else:
                log_term = 0.0

            # Talwani formula contributions
            # Vertical component from horizontal magnetization
            contrib_x = Mx * (dtheta * tz - log_term * tx)

            # Vertical component from vertical magnetization
            contrib_z = Mz * (-dtheta * tx - log_term * tz)

            contribution += contrib_x + contrib_z

        anomaly[i] = mu_0_4pi_nT * contribution

    return anomaly
