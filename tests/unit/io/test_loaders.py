"""Tests for model loading."""

import json
from pathlib import Path

import pytest

from forward_model.io import load_model
from forward_model.models import ForwardModel


class TestLoadModel:
    """Tests for loading models from JSON."""

    def test_load_valid_model(self, tmp_path: Path, simple_model: ForwardModel) -> None:
        """Test loading a valid model from JSON."""
        # Write model to temporary file
        model_file = tmp_path / "model.json"
        with open(model_file, "w") as f:
            json.dump(simple_model.model_dump(), f)

        # Load and verify
        loaded = load_model(model_file)
        assert len(loaded.bodies) == len(simple_model.bodies)
        assert loaded.field.intensity == simple_model.field.intensity
        assert loaded.observation_x == simple_model.observation_x

    def test_load_with_string_path(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test that string paths work."""
        model_file = tmp_path / "model.json"
        with open(model_file, "w") as f:
            json.dump(simple_model.model_dump(), f)

        # Load using string path
        loaded = load_model(str(model_file))
        assert len(loaded.bodies) == 1

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test that missing file raises FileNotFoundError."""
        model_file = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(model_file)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test that malformed JSON raises ValueError."""
        model_file = tmp_path / "invalid.json"
        model_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_model(model_file)

    def test_load_invalid_model(self, tmp_path: Path) -> None:
        """Test that model validation errors are caught."""
        model_file = tmp_path / "invalid_model.json"
        # Missing required field
        with open(model_file, "w") as f:
            json.dump(
                {
                    "bodies": [],  # Empty bodies list (invalid)
                    "field": {
                        "intensity": 50000.0,
                        "inclination": 60.0,
                        "declination": 0.0,
                    },
                    "observation_x": [0.0, 10.0],
                    "observation_z": 0.0,
                },
                f,
            )

        with pytest.raises(ValueError, match="validation failed"):
            load_model(model_file)
