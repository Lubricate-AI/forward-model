"""Talwani (1965) algorithm for 2D magnetic anomalies."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.geometry import (
    edge_geometry_2_5d,
    edge_geometry_2_75d,
    edge_geometry_2d,
)


@dataclass
class MagneticComponents:
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


# Backward-compatible alias — external code using AnomalyComponents keeps working
AnomalyComponents = MagneticComponents


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
    n_obs = len(observation_points)
    bz = np.zeros(n_obs, dtype=np.float64)
    bx = np.zeros(n_obs, dtype=np.float64)

    # Magnetic constant: μ₀/(4π) in SI units
    # Convert to nT: multiply by 10⁹
    mu_0_4pi = 1e-7  # T·m/A in SI
    mu_0_4pi_nT = mu_0_4pi * 1e9  # nT·m/A

    Mx, Mz = magnetization

    # Loop over polygon edges (N iterations, each vectorized over M obs points)
    for tx, tz, dtheta, log_term, valid in edge_geometry_2d(
        vertices, observation_points, min_distance
    ):
        # Bz (vertical): Talwani (1965)
        bz += np.where(
            valid,
            Mx * (dtheta * tz - log_term * tx) + Mz * (-dtheta * tx - log_term * tz),
            0.0,
        )
        # Bx (horizontal): Talwani (1965)
        bx += np.where(
            valid,
            Mx * (dtheta * tx + log_term * tz) + Mz * (-dtheta * tz + log_term * tx),
            0.0,
        )

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
    n_obs = len(observation_points)
    bz = np.zeros(n_obs, dtype=np.float64)
    bx = np.zeros(n_obs, dtype=np.float64)

    mu_0_4pi = 1e-7  # T·m/A in SI
    mu_0_4pi_nT = mu_0_4pi * 1e9  # nT·m/A

    Mx, Mz = magnetization

    for tx, tz, dtheta, dlambda, valid in edge_geometry_2_5d(
        vertices, observation_points, strike_half_length, min_distance
    ):
        bz += np.where(
            valid,
            Mx * (dtheta * tz - dlambda * tx) + Mz * (-dtheta * tx - dlambda * tz),
            0.0,
        )
        bx += np.where(
            valid,
            Mx * (dtheta * tx + dlambda * tz) + Mz * (-dtheta * tz + dlambda * tx),
            0.0,
        )

    bz *= mu_0_4pi_nT
    bx *= mu_0_4pi_nT
    return PolygonComponents(bz=bz, bx=bx)


def compute_polygon_anomaly_2_75d(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    magnetization: NDArray[np.float64],
    strike_forward: float,
    strike_backward: float,
    min_distance: float = 1e-10,
) -> PolygonComponents:
    """2.75D Talwani: finite asymmetric strike extent (Won & Bevis 1987).

    Body extends from y = −strike_backward to y = +strike_forward along strike.
    Implements superposition of two symmetric half-space contributions:

        ΔB_2.75D = (ΔB_2.5D(y₁) + ΔB_2.5D(y₂)) / 2

    where y₁ = strike_forward and y₂ = strike_backward. This formula follows
    from Won & Bevis (1987): the 2.5D kernel is even in y, so each half-space
    integral equals half the symmetric 2.5D result.

    Limit cases:
    - strike_forward == strike_backward == y₀ → ΔB_2.5D(y₀) exactly
    - both → ∞ → ΔB_2D exactly

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        magnetization: 2D magnetization vector [Mx, Mz] in A/m.
        strike_forward: Forward (+y) half-extent in meters. Must be > 0.
        strike_backward: Backward (−y) half-extent in meters. Must be > 0.
        min_distance: Minimum distance threshold to avoid singularities (meters).

    Returns:
        PolygonComponents(bz, bx): Vertical and horizontal anomaly components
        in nT at each observation point.

    References:
        Won, I. J., and Bevis, M. (1987). Computing the gravitational and magnetic
        anomalies due to a polygon: Algorithms and Fortran subroutines.
        Geophysics, 52(2), 232–238.
    """
    n_obs = len(observation_points)
    bz = np.zeros(n_obs, dtype=np.float64)
    bx = np.zeros(n_obs, dtype=np.float64)

    mu_0_4pi = 1e-7  # T·m/A in SI
    mu_0_4pi_nT = mu_0_4pi * 1e9  # nT·m/A

    Mx, Mz = magnetization

    for tx, tz, dtheta, dlambda, valid in edge_geometry_2_75d(
        vertices, observation_points, strike_forward, strike_backward, min_distance
    ):
        bz += np.where(
            valid,
            Mx * (dtheta * tz - dlambda * tx) + Mz * (-dtheta * tx - dlambda * tz),
            0.0,
        )
        bx += np.where(
            valid,
            Mx * (dtheta * tx + dlambda * tz) + Mz * (-dtheta * tz + dlambda * tx),
            0.0,
        )

    bz *= mu_0_4pi_nT
    bx *= mu_0_4pi_nT
    return PolygonComponents(bz=bz, bx=bx)
