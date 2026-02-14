"""Tests for result writers."""

import json
from pathlib import Path

import numpy as np

from forward_model.io import write_csv, write_json
from forward_model.models import ForwardModel


class TestWriteCSV:
    """Tests for writing CSV files."""

    def test_write_csv(self, tmp_path: Path) -> None:
        """Test writing anomaly results to CSV."""
        csv_file = tmp_path / "results.csv"
        obs_x = [0.0, 10.0, 20.0]
        anomaly = np.array([1.0, 2.0, 3.0])

        write_csv(csv_file, obs_x, anomaly)

        # Verify file exists
        assert csv_file.exists()

        # Read and verify contents
        lines = csv_file.read_text().strip().split("\n")
        assert len(lines) == 4  # Header + 3 data rows
        assert lines[0] == "x_m,anomaly_nT"
        assert lines[1] == "0.0,1.0"
        assert lines[2] == "10.0,2.0"
        assert lines[3] == "20.0,3.0"

    def test_write_csv_with_string_path(self, tmp_path: Path) -> None:
        """Test that string paths work for CSV writing."""
        csv_file = tmp_path / "results.csv"
        obs_x = [0.0, 10.0]
        anomaly = np.array([1.0, 2.0])

        write_csv(str(csv_file), obs_x, anomaly)
        assert csv_file.exists()


class TestWriteJSON:
    """Tests for writing JSON files."""

    def test_write_json(self, tmp_path: Path, simple_model: ForwardModel) -> None:
        """Test writing model and results to JSON."""
        json_file = tmp_path / "results.json"
        anomaly = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        write_json(json_file, simple_model, anomaly)

        # Verify file exists
        assert json_file.exists()

        # Read and verify contents
        with open(json_file) as f:
            data = json.load(f)

        assert "model" in data
        assert "results" in data
        assert data["results"]["observation_x"] == simple_model.observation_x
        assert data["results"]["anomaly_nT"] == anomaly.tolist()

    def test_write_json_with_string_path(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test that string paths work for JSON writing."""
        json_file = tmp_path / "results.json"
        anomaly = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        write_json(str(json_file), simple_model, anomaly)
        assert json_file.exists()

    def test_json_roundtrip(self, tmp_path: Path, simple_model: ForwardModel) -> None:
        """Test that we can write and reload a model."""
        json_file = tmp_path / "roundtrip.json"
        anomaly = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        # Write
        write_json(json_file, simple_model, anomaly)

        # Read back
        with open(json_file) as f:
            data = json.load(f)

        # Verify we can reconstruct the model
        reloaded = ForwardModel.model_validate(data["model"])
        assert len(reloaded.bodies) == len(simple_model.bodies)
        assert reloaded.field.intensity == simple_model.field.intensity
        assert reloaded.observation_x == simple_model.observation_x
