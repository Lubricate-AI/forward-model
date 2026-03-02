"""End-to-end integration tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from forward_model.compute import calculate_anomaly
from forward_model.io import load_model, write_csv, write_json
from forward_model.models import (
    MagneticModel,
    GeologicBody,
    HeatFlowModel,
    MagneticField,
    MagneticProperties,
    ThermalProperties,
)
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
            magnetic=MagneticProperties(susceptibility=0.05),
            name="Test Body",
        )

        field = MagneticField(
            intensity=50000.0,
            inclination=60.0,
            declination=0.0,
        )

        model = MagneticModel(
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

        reloaded_model = MagneticModel.model_validate(data["model"])

        # Verify roundtrip equality
        assert len(reloaded_model.bodies) == len(model.bodies)
        assert reloaded_model.bodies[0].name == model.bodies[0].name
        assert reloaded_model.bodies[0].magnetic is not None
        assert model.bodies[0].magnetic is not None
        assert (
            reloaded_model.bodies[0].magnetic.susceptibility
            == model.bodies[0].magnetic.susceptibility
        )
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
            magnetic=MagneticProperties(susceptibility=0.05),
            name="Body 1",
        )

        body2 = GeologicBody(
            vertices=[[100.0, 50.0], [150.0, 50.0], [150.0, 150.0], [100.0, 150.0]],
            magnetic=MagneticProperties(susceptibility=0.1),
            name="Body 2",
        )

        field = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)

        model = MagneticModel(
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
        model = MagneticModel(
            bodies=[
                GeologicBody(
                    vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    magnetic=MagneticProperties(susceptibility=0.01),
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
        assert isinstance(loaded, MagneticModel)
        assert len(loaded.bodies) == 1
        assert loaded.bodies[0].name == "Triangle"
        assert loaded.field.intensity == 50000.0

    def test_complex_model_roundtrip(
        self, tmp_path: Path, simple_model: MagneticModel
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
        reloaded = MagneticModel.model_validate(data["model"])
        assert reloaded.model_dump() == simple_model.model_dump()

        # Verify results match
        loaded_anomaly = np.array(data["results"]["anomaly_nT"])
        assert np.allclose(loaded_anomaly, anomaly)


class TestHeatFlowWorkflow:
    """Integration test: load → compute → verify for heat flow."""

    def test_heat_flow_compute_full_path(self) -> None:
        """Full path: HeatFlowModel created, anomaly computed, result correct."""
        from forward_model import HeatFlowComponents, calculate_anomaly

        body = GeologicBody(
            vertices=[[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            thermal=ThermalProperties(conductivity=1.0, heat_generation=2.5),
            name="GraniteBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=list(np.linspace(-100.0, 100.0, 21)),
            observation_z=0.0,
            background_heat_flow=65.0,
        )

        result = calculate_anomaly(model)

        assert isinstance(result, HeatFlowComponents)
        assert result.heat_flow.shape == (21,)
        assert np.all(np.isfinite(result.heat_flow))
        assert np.all(np.isfinite(result.heat_flow_gradient))
        # Symmetric body → symmetric heat flow
        np.testing.assert_allclose(result.heat_flow[0], result.heat_flow[20], rtol=1e-6)
        np.testing.assert_allclose(result.heat_flow[5], result.heat_flow[15], rtol=1e-6)
        # Perturbation only — should be much less than background
        assert np.all(np.abs(result.heat_flow) < 100.0)

    def test_heat_flow_json_roundtrip(self, tmp_path: Path) -> None:
        """HeatFlowModel serialises to JSON and can be reloaded."""
        from forward_model import calculate_anomaly

        model = HeatFlowModel(
            bodies=[
                GeologicBody(
                    vertices=[[0.0, 50.0], [100.0, 50.0], [100.0, 150.0], [0.0, 150.0]],
                    thermal=ThermalProperties(conductivity=2.0),
                    name="Slab",
                )
            ],
            observation_x=[0.0, 50.0, 100.0],
            observation_z=0.0,
        )
        result = calculate_anomaly(model)

        json_file = tmp_path / "heat_flow_model.json"
        write_json(json_file, model, result.heat_flow)
        assert json_file.exists()

        import json as _json

        with open(json_file) as f:
            data = _json.load(f)

        loaded = HeatFlowModel.model_validate(data["model"])
        assert loaded.model_type == "heat_flow"
        assert len(loaded.bodies) == 1
        assert loaded.bodies[0].thermal is not None
        assert loaded.bodies[0].thermal.conductivity == 2.0


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
