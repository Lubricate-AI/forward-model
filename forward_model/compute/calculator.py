"""High-level interface for magnetic anomaly calculations."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.talwani import (
    compute_polygon_anomaly,
    field_to_magnetization,
)
from forward_model.models.model import ForwardModel


def _compute_single_body(
    args: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute anomaly for a single body. Module-level for pickling."""
    vertices, observation_points, magnetization = args
    return compute_polygon_anomaly(vertices, observation_points, magnetization)


def calculate_anomaly(
    model: ForwardModel,
    parallel: bool = False,
) -> NDArray[np.float64]:
    """Calculate total magnetic anomaly for a forward model.

    Computes the magnetic anomaly using the Talwani (1965) algorithm,
    summing contributions from all geologic bodies via superposition.

    Args:
        model: Complete forward model specification including bodies,
               field parameters, and observation points.
        parallel: If True, compute each body's anomaly in a separate process
                  using ProcessPoolExecutor. Useful when the model has many
                  bodies and observation grids are large.

    Returns:
        Array of magnetic anomaly values in nanoTesla (nT) at each
        observation point. Length matches model.observation_x.

    Example:
        >>> model = load_model("model.json")
        >>> anomaly = calculate_anomaly(model)
        >>> print(f"Max anomaly: {anomaly.max():.1f} nT")
    """
    observation_points = model.get_observation_points()

    body_args: list[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
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
        )
        vertices = body.to_numpy()
        body_args.append((vertices, observation_points, magnetization))

    if parallel:
        with ProcessPoolExecutor() as executor:
            body_anomalies = list(executor.map(_compute_single_body, body_args))
    else:
        body_anomalies = [_compute_single_body(args) for args in body_args]

    return np.sum(body_anomalies, axis=0)
