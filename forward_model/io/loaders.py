"""Model loading from JSON and CSV files."""

import csv
import json
from pathlib import Path

from pydantic import ValidationError

from forward_model.models.model import ForwardModel


def load_model(filepath: str | Path) -> ForwardModel:
    """Load a forward model from a JSON file.

    Args:
        filepath: Path to JSON file containing model definition.
                 Can be a string or Path object.

    Returns:
        Validated ForwardModel instance.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains invalid JSON or fails model validation.

    Example:
        >>> model = load_model("model.json")
        >>> print(f"Loaded {len(model.bodies)} bodies")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}") from e

    try:
        model = ForwardModel.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Model validation failed for {filepath}: {e}") from e

    return model


def load_model_from_csv(filepath: str | Path) -> ForwardModel:
    """Load a forward model from a CSV file.

    CSV Format (3 sections separated by blank lines):

    1. Field parameters (4 values on one line):
       intensity,inclination,declination,observation_z

    2. Bodies section (one body per line):
       # Bodies
       name,susceptibility,x1,z1,x2,z2,...

    3. Observations section (one line of x-coordinates):
       # Observations
       x1,x2,x3,...

    Lines starting with # are comments/headers.

    Args:
        filepath: Path to CSV file containing model definition.
                 Can be a string or Path object.

    Returns:
        Validated ForwardModel instance.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is malformed or fails model validation.

    Example:
        >>> model = load_model_from_csv("model.csv")
        >>> print(f"Loaded {len(model.bodies)} bodies")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        with open(filepath) as f:
            reader = csv.reader(f)
            lines = [line for line in reader if line and not line[0].startswith("#")]

        if len(lines) < 3:
            raise ValueError(
                "CSV must contain at least 3 lines: field params, body, observations"
            )

        # Parse field parameters (line 0)
        field_params = lines[0]
        if len(field_params) != 4:
            raise ValueError(
                f"Field parameters must have 4 values, got {len(field_params)}"
            )

        field_data = {
            "intensity": float(field_params[0]),
            "inclination": float(field_params[1]),
            "declination": float(field_params[2]),
        }
        observation_z = float(field_params[3])

        # Parse observation x-coordinates (last line)
        observation_x = [float(x) for x in lines[-1]]

        # Parse bodies (all lines between field params and observations)
        bodies_data = []
        for body_line in lines[1:-1]:
            if len(body_line) < 3:
                raise ValueError(
                    "Body line must have at least 3 values (name, susc, vertices)"
                )

            name = body_line[0]
            susceptibility = float(body_line[1])

            # Parse vertices (remaining values as x,z pairs)
            coords = [float(v) for v in body_line[2:]]
            if len(coords) % 2 != 0:
                raise ValueError(
                    f"Body '{name}' has odd number of coordinates: {len(coords)}"
                )

            vertices = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]

            bodies_data.append(
                {"name": name, "susceptibility": susceptibility, "vertices": vertices}
            )

        # Construct model dict
        model_data = {
            "bodies": bodies_data,
            "field": field_data,
            "observation_x": observation_x,
            "observation_z": observation_z,
        }

        # Validate and return
        model = ForwardModel.model_validate(model_data)
        return model

    except ValidationError as e:
        raise ValueError(f"Model validation failed for {filepath}: {e}") from e
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse CSV file {filepath}: {e}") from e
