"""Forward model container."""

from forward_model.models.base import ObservationModel
from forward_model.models.field import MagneticField


class ForwardModel(ObservationModel, frozen=True):
    """Complete forward magnetic model specification.

    Attributes:
        bodies: List of geologic bodies to include in the model.
                Must contain at least one body.
        field: Earth's magnetic field parameters.
        observation_x: List of x-coordinates for observation points (meters).
        observation_z: Fixed z-coordinate for all observation points (meters).
                      Typically 0 for surface observations.
    """

    field: MagneticField
