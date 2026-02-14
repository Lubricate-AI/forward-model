"""Tests for CSV model loading."""

from pathlib import Path

import pytest

from forward_model.io.loaders import load_model_from_csv
from forward_model.models import ForwardModel


class TestLoadModelFromCSV:
    """Tests for load_model_from_csv function."""

    def test_load_simple_csv(self, tmp_path: Path) -> None:
        """Test loading a simple CSV model."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
-100.0,-50.0,0.0,25.0,50.0,100.0,150.0
"""
        csv_file.write_text(csv_content)

        model = load_model_from_csv(csv_file)

        assert isinstance(model, ForwardModel)
        assert len(model.bodies) == 1
        assert model.bodies[0].name == "Rectangle"
        assert model.bodies[0].susceptibility == 0.05
        assert len(model.bodies[0].vertices) == 4
        assert model.field.intensity == 50000.0
        assert model.field.inclination == 60.0
        assert model.field.declination == 0.0
        assert model.observation_z == 0.0
        assert len(model.observation_x) == 7

    def test_load_csv_with_comments(self, tmp_path: Path) -> None:
        """Test loading CSV with comment lines."""
        csv_file = tmp_path / "model.csv"
        # Field parameters: intensity,inclination,declination,observation_z
        csv_content = """50000.0,60.0,0.0,0.0
# Bodies: name,susceptibility,x1,z1,x2,z2,...
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
# Observations: x-coordinates
-100.0,-50.0,0.0,25.0,50.0,100.0,150.0
"""
        csv_file.write_text(csv_content)

        model = load_model_from_csv(csv_file)

        assert len(model.bodies) == 1
        assert len(model.observation_x) == 7

    def test_load_csv_multiple_bodies(self, tmp_path: Path) -> None:
        """Test loading CSV with multiple bodies."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0
Body1,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
Body2,0.08,100.0,150.0,150.0,150.0,150.0,250.0,100.0,250.0
-100.0,0.0,100.0
"""
        csv_file.write_text(csv_content)

        model = load_model_from_csv(csv_file)

        assert len(model.bodies) == 2
        assert model.bodies[0].name == "Body1"
        assert model.bodies[1].name == "Body2"
        assert model.bodies[0].susceptibility == 0.05
        assert model.bodies[1].susceptibility == 0.08

    def test_load_csv_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model_from_csv(nonexistent)

    def test_load_csv_too_few_lines(self, tmp_path: Path) -> None:
        """Test CSV with too few lines raises ValueError."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="at least 3 lines"):
            load_model_from_csv(csv_file)

    def test_load_csv_invalid_field_params(self, tmp_path: Path) -> None:
        """Test CSV with invalid field parameters raises ValueError."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
-100.0,0.0,100.0
"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="Field parameters must have 4 values"):
            load_model_from_csv(csv_file)

    def test_load_csv_odd_coordinates(self, tmp_path: Path) -> None:
        """Test CSV with odd number of coordinates raises ValueError."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0
Rectangle,0.05,0.0,100.0,50.0
-100.0,0.0,100.0
"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="odd number of coordinates"):
            load_model_from_csv(csv_file)

    def test_load_csv_invalid_number_format(self, tmp_path: Path) -> None:
        """Test CSV with invalid number format raises ValueError."""
        csv_file = tmp_path / "model.csv"
        csv_content = """not_a_number,60.0,0.0,0.0
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
-100.0,0.0,100.0
"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="Failed to parse CSV"):
            load_model_from_csv(csv_file)

    def test_load_csv_empty_lines_ignored(self, tmp_path: Path) -> None:
        """Test that empty lines in CSV are ignored."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0

Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0

-100.0,-50.0,0.0,25.0,50.0,100.0,150.0
"""
        csv_file.write_text(csv_content)

        model = load_model_from_csv(csv_file)

        assert len(model.bodies) == 1
        assert len(model.observation_x) == 7
