"""High-level interface for magnetic anomaly calculations."""

import numpy as np
from numpy.typing import NDArray

from forward_model.compute.talwani import (
    compute_polygon_anomaly,
    field_to_magnetization,
)
from forward_model.models.model import ForwardModel


def calculate_anomaly(model: ForwardModel) -> NDArray[np.float64]:
    """Calculate total magnetic anomaly for a forward model.

    Computes the magnetic anomaly using the Talwani (1965) algorithm,
    summing contributions from all geologic bodies via superposition.

    Args:
        model: Complete forward model specification including bodies,
               field parameters, and observation points.

    Returns:
        Array of magnetic anomaly values in nanoTesla (nT) at each
        observation point. Length matches model.observation_x.

    Example:
        >>> model = load_model("model.json")
        >>> anomaly = calculate_anomaly(model)
        >>> print(f"Max anomaly: {anomaly.max():.1f} nT")
    """
    observation_points = model.get_observation_points()
    n_obs = len(observation_points)
    total_anomaly = np.zeros(n_obs, dtype=np.float64)

    # Sum contributions from all bodies (superposition principle)
    for body in model.bodies:
        # Convert field to magnetization for this body
        magnetization = field_to_magnetization(
            susceptibility=body.susceptibility,
            field_intensity=model.field.intensity,
            field_inclination=model.field.inclination,
            field_declination=model.field.declination,
        )

        # Get body vertices as NumPy array
        vertices = body.to_numpy()

        # Compute anomaly from this body
        body_anomaly = compute_polygon_anomaly(
            vertices=vertices,
            observation_points=observation_points,
            magnetization=magnetization,
        )

        # Add to total
        total_anomaly += body_anomaly

    return total_anomaly
