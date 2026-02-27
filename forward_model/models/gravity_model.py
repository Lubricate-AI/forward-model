"""Gravity forward model container."""

from typing import Literal

from forward_model.models.base import ObservationModel


class GravityModel(ObservationModel, frozen=True):
    """Complete forward gravity model specification.

    Unlike ``ForwardModel``, this model does not require an inducing
    ``MagneticField`` — gravity bodies have no ambient field dependency.

    Attributes:
        model_type: Literal discriminator field. Always "gravity".
        bodies: List of geologic bodies to include in the model.
                Must contain at least one body.
        observation_x: List of x-coordinates for observation points (meters).
        observation_z: Fixed z-coordinate for all observation points (meters).
                      Typically 0 for surface observations.
    """

    model_type: Literal["gravity"] = "gravity"
