"""Talwani (1965) algorithm for 2D magnetic anomalies."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class AnomalyComponents:
    """All magnetic anomaly components at each observation point.

    Attributes:
        bz: Vertical component of the magnetic anomaly (nT).
        bx: Horizontal component of the magnetic anomaly (nT).
        total_field: Total field anomaly ΔT (nT), the projection of the anomaly
                     vector onto the inducing field direction.
        amplitude: Vector amplitude |ΔB| = sqrt(Bx² + Bz²) in nT.
        gradient: Horizontal gradient of the total field anomaly d(ΔT)/dx
                  (nT/m). Forward model of what a total-field gradiometer
                  measures along the profile.
    """

    bz: NDArray[np.float64]
    bx: NDArray[np.float64]
    total_field: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    gradient: NDArray[np.float64]


class PolygonComponents(NamedTuple):
    """Per-polygon Bz and Bx contributions before superposition."""

    bz: NDArray[np.float64]
    bx: NDArray[np.float64]


def field_to_magnetization(
    susceptibility: float,
    field_intensity: float,
    field_inclination: float,
    field_declination: float,
    remanent_intensity: float = 0.0,
    remanent_inclination: float = 0.0,
    remanent_declination: float = 0.0,
    demagnetization_factor: float = 0.0,
) -> NDArray[np.float64]:
    """Convert magnetic field parameters to 2D magnetization vector.

    Computes the total magnetization as the sum of the induced component
    (from susceptibility and the ambient field) and the remanent component
    (a permanent vector fixed at rock formation time).

    When ``demagnetization_factor`` (N_d) is non-zero, the induced susceptibility
    is corrected for the internal demagnetizing field:

        χ_eff = χ / (1 + N_d · χ)

    This correction is only applied to the induced component; the remanent
    component is unaffected.

    Args:
        susceptibility: Magnetic susceptibility (SI units, dimensionless).
        field_intensity: Inducing field intensity in nanoTesla (nT).
        field_inclination: Field inclination in degrees (-90 to 90).
        field_declination: Field declination in degrees (-180 to 180).
        remanent_intensity: Remanent magnetization intensity in A/m. Default 0.0.
        remanent_inclination: Inclination of remanent vector in degrees. Default 0.0.
        remanent_declination: Declination of remanent vector in degrees. Default 0.0.
        demagnetization_factor: Demagnetization factor N_d in [0.0, 1.0].
                                Default 0.0 (no correction).

    Returns:
        2D magnetization vector [Mx, Mz] in A/m representing total magnetization
        (induced + remanent) projected into the vertical profile plane.
    """
    # Apply demagnetization correction to effective susceptibility
    chi_eff = susceptibility / (1.0 + demagnetization_factor * susceptibility)

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

    # Induced magnetization M = χ_eff * H = χ_eff * B/μ₀
    # μ₀ = 4π × 10⁻⁷ H/m
    mu_0 = 4.0 * np.pi * 1e-7
    Mx = chi_eff * Bx / mu_0
    Mz = chi_eff * Bz / mu_0

    # Remanent component — already in A/m, project into 2D profile plane
    inc_rem = np.deg2rad(remanent_inclination)
    dec_rem = np.deg2rad(remanent_declination)
    Mx_rem = remanent_intensity * np.cos(inc_rem) * np.cos(dec_rem)
    Mz_rem = -remanent_intensity * np.sin(inc_rem)  # negative: z positive-down

    return np.array([Mx + Mx_rem, Mz + Mz_rem], dtype=np.float64)


def compute_demagnetization_factor(vertices: NDArray[np.float64]) -> float:
    """Estimate 2D demagnetization factor from polygon geometry.

    Uses the equivalent-ellipse approximation of the polygon's axis-aligned
    bounding box. For an ellipse with horizontal semi-axis *a* and vertical
    semi-axis *b*, the 2D demagnetization factor (field applied along *b*) is:

        N_d = b / (a + b)

    The result is clamped to [0.0, 0.5], which is the physically valid range
    for 2D infinite-strike bodies.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.

    Returns:
        Estimated demagnetization factor N_d in [0.0, 0.5].
    """
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    z_range = vertices[:, 1].max() - vertices[:, 1].min()
    a = x_range / 2.0  # horizontal semi-axis
    b = z_range / 2.0  # vertical semi-axis
    denom = a + b
    if denom == 0.0:
        return 0.0
    n_d = b / denom
    return float(np.clip(n_d, 0.0, 0.5))


def compute_polygon_anomaly(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    magnetization: NDArray[np.float64],
    min_distance: float = 1e-10,
) -> PolygonComponents:
    """Compute magnetic anomaly components from a 2D polygon using Talwani algorithm.

    Implements the Talwani (1965) algorithm for computing the vertical (Bz) and
    horizontal (Bx) components of the magnetic anomaly from a 2D polygonal body.
    Both components are computed in the same edge loop, reusing all intermediate
    variables (r1, r2, dθ, log term, tx, tz) for efficiency.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        magnetization: 2D magnetization vector [Mx, Mz] in A/m.
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        PolygonComponents(bz, bx): Vertical and horizontal anomaly components
        in nT at each observation point.

    References:
        Talwani, M., and Heirtzler, J. R. (1965). Computation of magnetic
        anomalies caused by two-dimensional structures of arbitrary shape.
    """
    n_vertices = len(vertices)
    n_obs = len(observation_points)
    bz = np.zeros(n_obs, dtype=np.float64)
    bx = np.zeros(n_obs, dtype=np.float64)

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

        # Bz (vertical): Talwani (1965)
        bz_contrib = np.where(
            valid,
            Mx * (dtheta * tz - log_term * tx) + Mz * (-dtheta * tx - log_term * tz),
            0.0,
        )
        # Bx (horizontal): Talwani (1965)
        bx_contrib = np.where(
            valid,
            Mx * (dtheta * tx + log_term * tz) + Mz * (-dtheta * tz + log_term * tx),
            0.0,
        )
        bz += bz_contrib
        bx += bx_contrib

    bz *= mu_0_4pi_nT
    bx *= mu_0_4pi_nT
    return PolygonComponents(bz=bz, bx=bx)


def compute_polygon_anomaly_2_5d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    magnetization: NDArray[np.float64],
    strike_half_length: float,
    min_distance: float = 1e-10,
) -> PolygonComponents:
    """2.5D Talwani computation with finite symmetric strike extent (Won & Bevis 1987).

    Replaces the standard 2D Talwani vertex functions with Won & Bevis (1987)
    modified functions that account for finite strike extent:

    - Modified angle:  Θk = arctan2(zk·y0,  xk·sqrt(rk²+y0²))
    - Modified log:    Λk = log(rk / (sqrt(rk²+y0²) + y0))

    Both functions reduce to the 2D Talwani values (arctan2(z,x) and log(r)) as
    y0 → ∞, so the formula is exactly backwards-compatible in the limiting case.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        magnetization: 2D magnetization vector [Mx, Mz] in A/m.
        strike_half_length: Half-length of the body in the strike direction (m).
                           Must be strictly positive and finite.
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        PolygonComponents(bz, bx): Vertical and horizontal anomaly components
        in nT at each observation point, attenuated for finite strike extent.

    References:
        Won, I. J., and Bevis, M. (1987). Computing the gravitational and magnetic
        anomalies due to a polygon: Algorithms and Fortran subroutines.
        Geophysics, 52(2), 232–238.
    """
    n_vertices = len(vertices)
    n_obs = len(observation_points)
    bz = np.zeros(n_obs, dtype=np.float64)
    bx = np.zeros(n_obs, dtype=np.float64)

    mu_0_4pi = 1e-7  # T·m/A in SI
    mu_0_4pi_nT = mu_0_4pi * 1e9  # nT·m/A

    Mx, Mz = magnetization

    obs_x = observation_points[:, 0]  # (M,)
    obs_z = observation_points[:, 1]  # (M,)

    y0 = strike_half_length

    for j in range(n_vertices):
        j_next = (j + 1) % n_vertices

        dx = vertices[j_next, 0] - vertices[j, 0]
        dz = vertices[j_next, 1] - vertices[j, 1]
        edge_length = np.sqrt(dx**2 + dz**2)

        if np.abs(dx) < min_distance and np.abs(dz) < min_distance:
            continue

        tx = dx / edge_length
        tz = dz / edge_length

        x1 = vertices[j, 0] - obs_x
        z1 = vertices[j, 1] - obs_z
        x2 = vertices[j_next, 0] - obs_x
        z2 = vertices[j_next, 1] - obs_z

        r1 = np.sqrt(x1**2 + z1**2)
        r2 = np.sqrt(x2**2 + z2**2)
        valid = (r1 >= min_distance) & (r2 >= min_distance)

        # Won & Bevis (1987) modified angle function:
        #   Θk = arctan2(zk·y0, xk·sqrt(rk²+y0²))
        # Reduces to arctan2(zk, xk) = θk as y0 → ∞.
        sr1 = np.sqrt(r1**2 + y0**2)  # sqrt(r1²+y0²), shape (M,)
        sr2 = np.sqrt(r2**2 + y0**2)
        theta1: NDArray[np.float64] = np.arctan2(z1 * y0, x1 * sr1)
        theta2: NDArray[np.float64] = np.arctan2(z2 * y0, x2 * sr2)
        dtheta: NDArray[np.float64] = theta2 - theta1
        dtheta = np.where(dtheta > np.pi, dtheta - 2 * np.pi, dtheta)
        dtheta = np.where(dtheta < -np.pi, dtheta + 2 * np.pi, dtheta)

        # Won & Bevis (1987) modified log function:
        #   Λk = log(rk / (sqrt(rk²+y0²) + y0))
        # Difference Λ2-Λ1 reduces to log(r2/r1) as y0 → ∞.
        lambda1 = np.zeros_like(r1)
        np.log(r1 / (sr1 + y0), out=lambda1, where=valid)
        lambda2 = np.zeros_like(r2)
        np.log(r2 / (sr2 + y0), out=lambda2, where=valid)
        dlambda: NDArray[np.float64] = lambda2 - lambda1

        bz_contrib = np.where(
            valid,
            Mx * (dtheta * tz - dlambda * tx) + Mz * (-dtheta * tx - dlambda * tz),
            0.0,
        )
        bx_contrib = np.where(
            valid,
            Mx * (dtheta * tx + dlambda * tz) + Mz * (-dtheta * tz + dlambda * tx),
            0.0,
        )
        bz += bz_contrib
        bx += bx_contrib

    bz *= mu_0_4pi_nT
    bx *= mu_0_4pi_nT
    return PolygonComponents(bz=bz, bx=bx)
