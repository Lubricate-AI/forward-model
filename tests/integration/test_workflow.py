"""End-to-end integration tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from forward_model.compute import calculate_anomaly
from forward_model.io import load_model, write_csv, write_json
from forward_model.models import ForwardModel, GeologicBody, MagneticField
from forward_model.viz import plot_combined


class TestCompleteWorkflow:
    """Test complete workflow from model creation to visualization."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete end-to-end workflow."""
        # Step 1: Create model programmatically
        body = GeologicBody(
            vertices=[
                [0.0, 100.0],
                [100.0, 100.0],
                [100.0, 200.0],
                [0.0, 200.0],
            ],
            susceptibility=0.05,
            name="Test Body",
        )

        field = MagneticField(
            intensity=50000.0,
            inclination=60.0,
            declination=0.0,
        )

        model = ForwardModel(
            bodies=[body],
            field=field,
            observation_x=[-100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0],
            observation_z=0.0,
        )

        # Step 2: Compute anomaly
        anomaly = calculate_anomaly(model)

        # Verify computation results
        assert anomaly.shape == (7,)
        assert np.all(np.isfinite(anomaly))

        # Step 3: Export to JSON and CSV
        json_file = tmp_path / "model_results.json"
        csv_file = tmp_path / "results.csv"

        write_json(json_file, model, anomaly)
        write_csv(csv_file, model.observation_x, anomaly)

        # Verify files created
        assert json_file.exists()
        assert csv_file.exists()

        # Step 4: Reload model from JSON
        with open(json_file) as f:
            data = json.load(f)

        reloaded_model = ForwardModel.model_validate(data["model"])

        # Verify roundtrip equality
        assert len(reloaded_model.bodies) == len(model.bodies)
        assert reloaded_model.bodies[0].name == model.bodies[0].name
        assert reloaded_model.bodies[0].susceptibility == model.bodies[0].susceptibility
        assert reloaded_model.field.intensity == model.field.intensity
        assert reloaded_model.observation_x == model.observation_x

        # Step 5: Generate visualization
        plot_file = tmp_path / "output.png"
        fig = plot_combined(model, anomaly, save_path=plot_file)

        # Verify visualization created
        assert plot_file.exists()
        assert len(fig.axes) == 2

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_bodies_workflow(self, tmp_path: Path) -> None:
        """Test workflow with multiple geologic bodies."""
        # Create two bodies
        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            susceptibility=0.05,
            name="Body 1",
        )

        body2 = GeologicBody(
            vertices=[[100.0, 50.0], [150.0, 50.0], [150.0, 150.0], [100.0, 150.0]],
            susceptibility=0.1,
            name="Body 2",
        )

        field = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)

        model = ForwardModel(
            bodies=[body1, body2],
            field=field,
            observation_x=np.linspace(-100, 250, 50).tolist(),
            observation_z=0.0,
        )

        # Compute and verify
        anomaly = calculate_anomaly(model)
        assert anomaly.shape == (50,)
        assert np.all(np.isfinite(anomaly))

        # Export
        json_file = tmp_path / "multi_body.json"
        write_json(json_file, model, anomaly)

        # Verify
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert len(data["model"]["bodies"]) == 2


class TestJSONRoundtrip:
    """Test JSON serialization/deserialization."""

    def test_minimal_model_roundtrip(self, tmp_path: Path) -> None:
        """Test roundtrip with minimal model."""
        # Create minimal model
        model = ForwardModel(
            bodies=[
                GeologicBody(
                    vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    susceptibility=0.01,
                    name="Triangle",
                )
            ],
            field=MagneticField(intensity=50000.0, inclination=0.0, declination=0.0),
            observation_x=[0.0],
            observation_z=0.0,
        )

        # Save to JSON
        json_file = tmp_path / "minimal.json"
        with open(json_file, "w") as f:
            json.dump(model.model_dump(), f)

        # Load and verify
        loaded = load_model(json_file)
        assert len(loaded.bodies) == 1
        assert loaded.bodies[0].name == "Triangle"
        assert loaded.field.intensity == 50000.0

    def test_complex_model_roundtrip(
        self, tmp_path: Path, simple_model: ForwardModel
    ) -> None:
        """Test roundtrip with complex model from fixtures."""
        # Compute anomaly
        anomaly = calculate_anomaly(simple_model)

        # Write to JSON
        json_file = tmp_path / "complex.json"
        write_json(json_file, simple_model, anomaly)

        # Read back
        with open(json_file) as f:
            data = json.load(f)

        # Verify model can be reconstructed
        reloaded = ForwardModel.model_validate(data["model"])
        assert reloaded.model_dump() == simple_model.model_dump()

        # Verify results match
        loaded_anomaly = np.array(data["results"]["anomaly_nT"])
        assert np.allclose(loaded_anomaly, anomaly)


class TestErrorHandling:
    """Test error handling in integrated workflow."""

    def test_invalid_model_file(self, tmp_path: Path) -> None:
        """Test handling of invalid model files."""
        # Create invalid JSON
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_model(bad_file)

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test handling of missing files."""
        missing = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_model(missing)
