"""Model loading from JSON files."""

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
