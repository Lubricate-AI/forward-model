"""Forward model container."""

from typing import Literal

from forward_model.models.base import ObservationModel
from forward_model.models.field import MagneticField


class MagneticModel(ObservationModel, frozen=True):
    """Complete forward magnetic model specification.

    Attributes:
        model_type: Literal discriminator field. Always "magnetic".
        bodies: List of geologic bodies to include in the model.
                Must contain at least one body.
        field: Earth's magnetic field parameters.
        observation_x: List of x-coordinates for observation points (meters).
        observation_z: Fixed z-coordinate for all observation points (meters).
                      Typically 0 for surface observations.
    """

    model_type: Literal["magnetic"] = "magnetic"
    field: MagneticField
