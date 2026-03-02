"""High-level interface for forward model anomaly calculations.

Dispatches to the appropriate algorithm based on model type:
magnetic (Talwani 1965), gravity (Talwani 1959), and heat flow (Talwani-style 2D).
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.gravity import GravityComponents, calculate_gravity
from forward_model.compute.heatflow_talwani import (
    HeatFlowComponents,
    calculate_heat_flow,
)
from forward_model.compute.talwani import (
    MagneticComponents,
    PolygonComponents,
    compute_polygon_anomaly,
    compute_polygon_anomaly_2_5d,
    compute_polygon_anomaly_2_75d,
    field_to_magnetization,
)
from forward_model.models.gravity_model import GravityModel
from forward_model.models.heatflow_model import HeatFlowModel
from forward_model.models.magnetic_model import MagneticModel

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
        if sb is None:  # pragma: no cover
            raise ValueError("strike_backward must be set when strike_forward is set")
        return compute_polygon_anomaly_2_75d(
            vertices,
            observation_points,
            magnetization,
            sf,
            sb,
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
        if sb is None:  # pragma: no cover
            raise ValueError("strike_backward must be set when strike_forward is set")
        return compute_polygon_anomaly_2_75d(
            vertices,
            _worker_obs_points,
            magnetization,
            sf,
            sb,
        )
    if shl is not None:
        return compute_polygon_anomaly_2_5d(
            vertices, _worker_obs_points, magnetization, shl
        )
    return compute_polygon_anomaly(vertices, _worker_obs_points, magnetization)


@overload
def calculate_anomaly(
    model: MagneticModel,
    parallel: bool = ...,
    component: Literal["bz", "bx", "total_field", "amplitude", "gradient"] = ...,
) -> NDArray[np.float64]: ...


@overload
def calculate_anomaly(
    model: MagneticModel,
    parallel: bool = ...,
    *,
    component: Literal["all"],
) -> MagneticComponents: ...


@overload
def calculate_anomaly(
    model: GravityModel,
    parallel: bool = ...,
    component: Literal["bz", "bx", "total_field", "amplitude", "gradient", "all"] = ...,
) -> GravityComponents: ...


@overload
def calculate_anomaly(
    model: HeatFlowModel,
    parallel: bool = ...,
    component: Literal["bz", "bx", "total_field", "amplitude", "gradient", "all"] = ...,
) -> HeatFlowComponents: ...


def calculate_anomaly(
    model: MagneticModel | GravityModel | HeatFlowModel,
    parallel: bool = False,
    component: Literal[
        "bz", "bx", "total_field", "amplitude", "gradient", "all"
    ] = "bz",
) -> NDArray[np.float64] | MagneticComponents | GravityComponents | HeatFlowComponents:
    """Calculate anomaly for a forward model, dispatching on model type.

    Computes the anomaly using the appropriate algorithm for the model type:
    - MagneticModel (magnetic): Talwani (1965) algorithm, returns NDArray or
      MagneticComponents
    - GravityModel: Talwani (1959) algorithm, returns GravityComponents (gz in mGal)
    - HeatFlowModel: 2D Talwani-style heat flow, returns HeatFlowComponents (mW/m²)

    Args:
        model: A MagneticModel, GravityModel, or HeatFlowModel instance.
        parallel: If True, compute each body's contribution in a separate process.
        component: For MagneticModel only — which magnetic component to return.
                   Ignored for GravityModel and HeatFlowModel. One of:
                   ``"bz"`` (default), ``"bx"``, ``"total_field"``,
                   ``"amplitude"``, ``"gradient"``, ``"all"``.

    Returns:
        - MagneticModel: ``NDArray[np.float64]`` or ``MagneticComponents``
          (when ``component="all"``)
        - GravityModel: ``GravityComponents`` with gz (mGal) and gz_gradient (mGal/m)
        - HeatFlowModel: ``HeatFlowComponents`` with heat_flow and heat_flow_x
          (mW/m²) and heat_flow_gradient (mW/m³)

    Example:
        >>> mag_model = load_model("magnetic.json")
        >>> anomaly = calculate_anomaly(mag_model)
        >>> grav_model = load_model("gravity.json")
        >>> components = calculate_anomaly(grav_model)
        >>> print(f"Max gz: {components.gz.max():.3f} mGal")
    """
    if isinstance(model, GravityModel):
        return calculate_gravity(model, parallel=parallel)

    if isinstance(model, HeatFlowModel):
        return calculate_heat_flow(model, parallel=parallel)

    # MagneticModel path — all existing logic unchanged below this point
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
        if body.magnetic is None:
            raise ValueError(
                f"Body '{body.name}' has no magnetic properties; "
                "magnetic calculation requires magnetic to be set"
            )
        magnetization = field_to_magnetization(
            susceptibility=body.magnetic.susceptibility,
            field_intensity=model.field.intensity,
            field_inclination=model.field.inclination,
            field_declination=model.field.declination,
            remanent_intensity=body.magnetic.remanent_intensity,
            remanent_inclination=body.magnetic.remanent_inclination,
            remanent_declination=body.magnetic.remanent_declination,
            demagnetization_factor=body.magnetic.demagnetization_factor,
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
        return MagneticComponents(
            bz=total_bz,
            bx=total_bx,
            total_field=total_field,
            amplitude=amplitude,
            gradient=gradient,
        )

    raise ValueError(f"Unknown component: {component!r}")  # pragma: no cover
