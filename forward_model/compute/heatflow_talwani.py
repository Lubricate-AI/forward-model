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


def _apply_radiogenic_kernel(
    vertices: NDArray[np.float64],
    observation_points: NDArray[np.float64],
    heat_generation: float,
    min_distance: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute radiogenic heat flow contribution from a 2D polygon.

    For a 2D body with uniform volumetric heat generation A (µW/m³), the
    contribution to surface vertical heat flow at point (x, 0) is:

        q_rad(x) = (A × 1e-6 / (2π)) × ∫∫_body z_s / r² dA' × 1e3 [mW/m²]

    where r = distance from observation point to source point (x_s, z_s).
    Via Green's theorem this area integral converts to a boundary line integral:

        ∫∫_body z_s/r² dA = ∮_∂body arctan(x_s/z_s) dz_s

    The per-edge analytic form (derived by integration by parts along each
    straight edge parameterised by arc length) is:

        I_edge = z₂·φ₂ − z₁·φ₁ + a·(tz·log(r₂/r₁) − tx·dθ)

    where φₖ = arctan2(xₖ, zₖ) = arctan(xₖ/zₖ), dθ = arctan2(z₂,x₂) −
    arctan2(z₁,x₁) (wrapped to [−π,π]), log = log(r₂/r₁), and
    a = x₁·tz − z₁·tx (signed perpendicular distance from the observation
    point to the edge line). The sum Σ I_edge is always ≥ 0 for a body
    located entirely below the observation surface.

    Args:
        vertices: Nx2 array of polygon vertices [x, z] in meters.
        observation_points: Mx2 array of observation points [x, z] in meters.
        heat_generation: Volumetric heat generation A in µW/m³.
        min_distance: Minimum distance threshold to avoid singularities (m).

    Returns:
        Radiogenic heat flow contribution in mW/m² at each observation point.
    """
    n_obs = len(observation_points)
    q_rad = np.zeros(n_obs, dtype=np.float64)

    # Convert µW/m³ → W/m³ (×1e-6), and W/m² → mW/m² (×1e3): net factor 1e-3
    # Divide by 2π for the 2D half-space normalization.
    scale_factor = heat_generation * 1e-3 / (2.0 * np.pi)

    obs_x = observation_points[:, 0]
    obs_z = observation_points[:, 1]
    n_vertices = len(vertices)

    for j in range(n_vertices):
        j_next = (j + 1) % n_vertices

        dx = vertices[j_next, 0] - vertices[j, 0]
        dz = vertices[j_next, 1] - vertices[j, 1]
        edge_length = np.sqrt(dx**2 + dz**2)

        if edge_length < min_distance:
            continue

        tx = float(dx / edge_length)
        tz = float(dz / edge_length)

        x1 = vertices[j, 0] - obs_x
        z1 = vertices[j, 1] - obs_z
        x2 = vertices[j_next, 0] - obs_x
        z2 = vertices[j_next, 1] - obs_z

        r1 = np.sqrt(x1**2 + z1**2)
        r2 = np.sqrt(x2**2 + z2**2)
        valid = (r1 >= min_distance) & (r2 >= min_distance)

        # phi = arctan(x/z) = arctan2(x, z)  [swapped args vs standard atan2]
        phi1: NDArray[np.float64] = np.arctan2(x1, z1)
        phi2: NDArray[np.float64] = np.arctan2(x2, z2)

        # Standard dtheta = arctan2(z2,x2) - arctan2(z1,x1), wrapped to [-π,π]
        theta1: NDArray[np.float64] = np.arctan2(z1, x1)
        theta2: NDArray[np.float64] = np.arctan2(z2, x2)
        dtheta: NDArray[np.float64] = theta2 - theta1
        dtheta = np.where(dtheta > np.pi, dtheta - 2 * np.pi, dtheta)
        dtheta = np.where(dtheta < -np.pi, dtheta + 2 * np.pi, dtheta)

        r_ratio = np.where(valid, r2 / r1, 1.0)
        log_ratio: NDArray[np.float64] = np.where(valid, np.log(r_ratio), 0.0)

        # Signed perpendicular distance from observation point to edge line
        a: NDArray[np.float64] = x1 * tz - z1 * tx

        contrib = np.where(
            valid,
            z2 * phi2 - z1 * phi1 + a * (tz * log_ratio - tx * dtheta),
            0.0,
        )
        q_rad += contrib

    q_rad *= scale_factor
    return q_rad


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
        float,
        float,
        float | None,
        float | None,
        float | None,
    ],
) -> NDArray[np.float64]:
    """Compute heat flow for a single body. Module-level for pickling."""
    vertices, obs, conductivity, heat_gen, q0, shl, sf, sb = args
    if sf is not None:
        if sb is None:  # pragma: no cover
            raise ValueError("strike_backward must be set when strike_forward is set")
        qz = _apply_heatflow_kernel_2_75d(vertices, obs, conductivity, q0, sf, sb)
    elif shl is not None:
        qz = _apply_heatflow_kernel_2_5d(vertices, obs, conductivity, q0, shl)
    else:
        qz = _apply_heatflow_kernel(vertices, obs, conductivity, q0)
    if heat_gen > 0.0:
        qz = qz + _apply_radiogenic_kernel(vertices, obs, heat_gen)
    return qz


def _compute_body_parallel(
    args: tuple[
        NDArray[np.float64],
        float,
        float,
        float,
        float | None,
        float | None,
        float | None,
    ],
) -> NDArray[np.float64]:
    """Compute heat flow for one body using the worker-local observation points."""
    if _worker_obs_points is None:  # pragma: no cover
        raise RuntimeError("Worker process not initialized with observation points.")
    vertices, conductivity, heat_gen, q0, shl, sf, sb = args
    if sf is not None:
        if sb is None:  # pragma: no cover
            raise ValueError("strike_backward must be set when strike_forward is set")
        qz = _apply_heatflow_kernel_2_75d(
            vertices, _worker_obs_points, conductivity, q0, sf, sb
        )
    elif shl is not None:
        qz = _apply_heatflow_kernel_2_5d(
            vertices, _worker_obs_points, conductivity, q0, shl
        )
    else:
        qz = _apply_heatflow_kernel(vertices, _worker_obs_points, conductivity, q0)
    if heat_gen > 0.0:
        qz = qz + _apply_radiogenic_kernel(vertices, _worker_obs_points, heat_gen)
    return qz


def calculate_heat_flow(
    model: HeatFlowModel, parallel: bool = False
) -> HeatFlowComponents:
    """Calculate total heat flow anomaly for a heat flow model.

    Computes surface heat flow perturbations using the 2D Talwani-style
    algorithm, summing contributions from all geologic bodies via superposition.
    Each body contributes:
    - A conductive perturbation (from thermal conductivity contrast)
    - An optional radiogenic contribution (from volumetric heat generation)

    Args:
        model: Complete heat flow model including bodies and observation points.
        parallel: If True, compute each body's contribution in a separate process
                  using ProcessPoolExecutor.

    Returns:
        HeatFlowComponents containing:
        - heat_flow: Vertical heat flow perturbation in mW/m²
        - heat_flow_x: Horizontal heat flow perturbation in mW/m²
        - heat_flow_gradient: Horizontal gradient of heat_flow in mW/m³

    Raises:
        ValueError: If any body lacks thermal properties.
    """
    observation_points = model.get_observation_points()
    q0 = model.background_heat_flow

    per_body: list[
        tuple[
            NDArray[np.float64],
            float,
            float,
            float,
            float | None,
            float | None,
            float | None,
        ]
    ] = []
    for body in model.bodies:
        if body.thermal is None:
            raise ValueError(
                f"Body '{body.name}' has no thermal properties; "
                "heat flow calculation requires thermal to be set"
            )
        per_body.append(
            (
                body.to_numpy(),
                body.thermal.conductivity,
                body.thermal.heat_generation,
                q0,
                body.strike_half_length,
                body.strike_forward,
                body.strike_backward,
            )
        )

    if parallel:
        with ProcessPoolExecutor(
            initializer=_init_worker, initargs=(observation_points,)
        ) as executor:
            body_qz = list(executor.map(_compute_body_parallel, per_body))
    else:
        body_qz = [
            _compute_single_body((v, observation_points, c, hg, q0, shl, sf, sb))
            for v, c, hg, q0, shl, sf, sb in per_body
        ]

    total_qz: NDArray[np.float64] = np.sum(body_qz, axis=0)

    # Horizontal gradient of vertical heat flow
    obs_x: NDArray[np.float64] = observation_points[:, 0]
    if len(obs_x) > 1:
        qz_gradient: NDArray[np.float64] = np.gradient(total_qz, obs_x)
    else:
        qz_gradient = np.zeros_like(total_qz)

    # Horizontal heat flow component (qx) via complementary kernel
    body_qx: list[NDArray[np.float64]] = []
    for body in model.bodies:
        assert body.thermal is not None  # already validated above
        qx = _apply_heatflow_kernel_x(
            body.to_numpy(), observation_points, body.thermal.conductivity, q0
        )
        body_qx.append(qx)
    total_qx: NDArray[np.float64] = np.sum(body_qx, axis=0)

    return HeatFlowComponents(
        heat_flow=total_qz,
        heat_flow_x=total_qx,
        heat_flow_gradient=qz_gradient,
    )
