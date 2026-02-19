"""High-level interface for magnetic anomaly calculations."""

from concurrent.futures import ProcessPoolExecutor
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.talwani import (
    AnomalyComponents,
    PolygonComponents,
    compute_polygon_anomaly,
    compute_polygon_anomaly_2_5d,
    compute_polygon_anomaly_2_75d,
    field_to_magnetization,
)
from forward_model.models.model import ForwardModel

_worker_obs_points: NDArray[np.float64] | None = None


def _init_worker(obs_points: NDArray[np.float64]) -> None:
    """Populate worker-process global with the shared observation grid."""
    global _worker_obs_points
    _worker_obs_points = obs_points


def _compute_single_body(
    args: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float | None,
        float | None,
        float | None,
    ],
) -> PolygonComponents:
    """Compute anomaly components for a single body. Module-level for pickling."""
    vertices, observation_points, magnetization, shl, sf, sb = args
    if sf is not None:
        return compute_polygon_anomaly_2_75d(
            vertices,
            observation_points,
            magnetization,
            sf,
            sb,  # type: ignore[arg-type]
        )
    if shl is not None:
        return compute_polygon_anomaly_2_5d(
            vertices, observation_points, magnetization, shl
        )
    return compute_polygon_anomaly(vertices, observation_points, magnetization)


def _compute_body_parallel(
    args: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        float | None,
        float | None,
        float | None,
    ],
) -> PolygonComponents:
    """Compute anomaly for one body using the worker-local observation points."""
    if _worker_obs_points is None:  # pragma: no cover
        raise RuntimeError("Worker process not initialized with observation points.")
    vertices, magnetization, shl, sf, sb = args
    if sf is not None:
        return compute_polygon_anomaly_2_75d(
            vertices,
            _worker_obs_points,
            magnetization,
            sf,
            sb,  # type: ignore[arg-type]
        )
    if shl is not None:
        return compute_polygon_anomaly_2_5d(
            vertices, _worker_obs_points, magnetization, shl
        )
    return compute_polygon_anomaly(vertices, _worker_obs_points, magnetization)


@overload
def calculate_anomaly(
    model: ForwardModel,
    parallel: bool = ...,
    component: Literal["bz", "bx", "total_field", "amplitude", "gradient"] = ...,
) -> NDArray[np.float64]: ...


@overload
def calculate_anomaly(
    model: ForwardModel,
    parallel: bool = ...,
    *,
    component: Literal["all"],
) -> AnomalyComponents: ...


def calculate_anomaly(
    model: ForwardModel,
    parallel: bool = False,
    component: Literal[
        "bz", "bx", "total_field", "amplitude", "gradient", "all"
    ] = "bz",
) -> NDArray[np.float64] | AnomalyComponents:
    """Calculate total magnetic anomaly for a forward model.

    Computes the magnetic anomaly using the Talwani (1965) algorithm,
    summing contributions from all geologic bodies via superposition.

    Args:
        model: Complete forward model specification including bodies,
               field parameters, and observation points.
        parallel: If True, compute each body's anomaly in a separate process
                  using ProcessPoolExecutor. Useful when the model has many
                  bodies and observation grids are large.
        component: Which anomaly component to return. One of:
                   - ``"bz"`` (default): Vertical component in nT. Returns
                     ``NDArray[np.float64]``. Backward-compatible.
                   - ``"bx"``: Horizontal component in nT.
                   - ``"total_field"``: Total field anomaly ΔT in nT,
                     computed as Bx·cos(I₀)·cos(D₀) + Bz·sin(I₀).
                   - ``"amplitude"``: Vector amplitude |ΔB| = sqrt(Bx²+Bz²) in nT.
                   - ``"gradient"``: Horizontal gradient d(ΔT)/dx in nT/m.
                     Forward model of the measurement from a total-field
                     gradiometer oriented along the profile.
                   - ``"all"``: Returns an ``AnomalyComponents`` dataclass
                     containing all five fields.

    Returns:
        ``NDArray[np.float64]`` for single-component requests, or
        ``AnomalyComponents`` when ``component="all"``.

    Example:
        >>> model = load_model("model.json")
        >>> anomaly = calculate_anomaly(model)
        >>> print(f"Max anomaly: {anomaly.max():.1f} nT")
        >>> components = calculate_anomaly(model, component="all")
        >>> print(f"Max ΔT: {components.total_field.max():.1f} nT")
    """
    observation_points = model.get_observation_points()

    per_body: list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            float | None,
            float | None,
            float | None,
        ]
    ] = []
    for body in model.bodies:
        magnetization = field_to_magnetization(
            susceptibility=body.susceptibility,
            field_intensity=model.field.intensity,
            field_inclination=model.field.inclination,
            field_declination=model.field.declination,
            remanent_intensity=body.remanent_intensity,
            remanent_inclination=body.remanent_inclination,
            remanent_declination=body.remanent_declination,
            demagnetization_factor=body.demagnetization_factor,
        )
        per_body.append(
            (
                body.to_numpy(),
                magnetization,
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
            _compute_single_body((v, observation_points, m, shl, sf, sb))
            for v, m, shl, sf, sb in per_body
        ]

    # Sum Bz and Bx separately via superposition
    total_bz: NDArray[np.float64] = np.sum([c.bz for c in body_components], axis=0)
    total_bx: NDArray[np.float64] = np.sum([c.bx for c in body_components], axis=0)

    if component == "bz":
        return total_bz

    if component == "bx":
        return total_bx

    # Derive model-level components requiring field geometry
    inc_rad = np.deg2rad(model.field.inclination)
    dec_rad = np.deg2rad(model.field.declination)
    total_field: NDArray[np.float64] = total_bx * np.cos(inc_rad) * np.cos(
        dec_rad
    ) + total_bz * np.sin(inc_rad)
    amplitude: NDArray[np.float64] = np.sqrt(total_bx**2 + total_bz**2)

    if component == "total_field":
        return total_field

    if component == "amplitude":
        return amplitude

    # d(ΔT)/dx — forward model of the horizontal gradient measured by a
    # total-field gradiometer along the profile
    obs_x: NDArray[np.float64] = observation_points[:, 0]
    gradient: NDArray[np.float64] = np.gradient(total_field, obs_x)

    if component == "gradient":
        return gradient

    if component == "all":
        return AnomalyComponents(
            bz=total_bz,
            bx=total_bx,
            total_field=total_field,
            amplitude=amplitude,
            gradient=gradient,
        )

    raise ValueError(f"Unknown component: {component!r}")  # pragma: no cover
