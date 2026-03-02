"""Tests for CLI commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forward_model.cli.commands import app
from forward_model.models import MagneticModel, GravityModel, HeatFlowModel

runner = CliRunner()


@pytest.fixture
def model_json_file(tmp_path: Path, simple_model: MagneticModel) -> Path:
    """Create a temporary JSON model file."""
    model_file = tmp_path / "test_model.json"
    with open(model_file, "w") as f:
        json.dump(simple_model.model_dump(), f)
    return model_file


@pytest.fixture
def gravity_model_json_file(tmp_path: Path, gravity_model: GravityModel) -> Path:
    """Create a temporary gravity model JSON file."""
    model_file = tmp_path / "test_gravity_model.json"
    with open(model_file, "w") as f:
        json.dump(gravity_model.model_dump(), f)
    return model_file


@pytest.fixture
def heat_flow_model_json_file(tmp_path: Path, heat_flow_model: HeatFlowModel) -> Path:
    """Create a temporary heat flow model JSON file."""
    model_file = tmp_path / "test_heat_flow_model.json"
    with open(model_file, "w") as f:
        json.dump(heat_flow_model.model_dump(), f)
    return model_file


@pytest.fixture
def results_json_file(
    tmp_path: Path, simple_model: MagneticModel, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Create a temporary results JSON file."""
    # Import here to avoid circular dependency
    from forward_model import calculate_anomaly

    # Calculate anomaly
    anomaly = calculate_anomaly(simple_model)

    # Create results file
    results_file = tmp_path / "test_results.json"
    output = {
        "model": simple_model.model_dump(),
        "results": {
            "observation_x": simple_model.observation_x,
            "anomaly_nT": anomaly.tolist(),
        },
    }

    with open(results_file, "w") as f:
        json.dump(output, f)

    return results_file


class TestRunCommand:
    """Tests for the 'run' command."""

    def test_run_basic(
        self, model_json_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test basic run command without outputs."""
        # Mock plt.show to avoid displaying plot
        import matplotlib.pyplot as plt

        show_called: list[bool] = []

        def mock_show() -> None:
            show_called.append(True)  # type: ignore[reportUnknownMemberType]

        monkeypatch.setattr(plt, "show", mock_show)

        result = runner.invoke(app, ["run", str(model_json_file), "--no-plot"])

        assert result.exit_code == 0
        assert "Calculation complete" in result.stdout

    def test_run_with_csv_output(
        self, model_json_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test run command with CSV output."""
        output_csv = tmp_path / "output.csv"

        result = runner.invoke(
            app,
            [
                "run",
                str(model_json_file),
                "--output-csv",
                str(output_csv),
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert output_csv.exists()

        # Verify CSV contents
        with open(output_csv) as f:
            lines = f.readlines()
            assert lines[0].strip() == "x_m,anomaly_nT"
            assert len(lines) > 1  # Header + data

    def test_run_with_json_output(self, model_json_file: Path, tmp_path: Path) -> None:
        """Test run command with JSON output."""
        output_json = tmp_path / "output.json"

        result = runner.invoke(
            app,
            [
                "run",
                str(model_json_file),
                "--output-json",
                str(output_json),
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert output_json.exists()

        # Verify JSON structure
        with open(output_json) as f:
            data = json.load(f)
            assert "model" in data
            assert "results" in data
            assert "anomaly_nT" in data["results"]

    def test_run_with_plot_output(self, model_json_file: Path, tmp_path: Path) -> None:
        """Test run command with plot output."""
        output_plot = tmp_path / "output.png"

        result = runner.invoke(
            app, ["run", str(model_json_file), "--plot", str(output_plot)]
        )

        assert result.exit_code == 0
        assert output_plot.exists()

    def test_run_with_npy_output(self, model_json_file: Path, tmp_path: Path) -> None:
        """Test run command with NumPy output."""
        import numpy as np

        output_npy = tmp_path / "output.npy"

        result = runner.invoke(
            app,
            [
                "run",
                str(model_json_file),
                "--output-npy",
                str(output_npy),
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert output_npy.exists()

        # Verify NumPy array structure
        data = np.load(output_npy)
        assert data.shape[1] == 2  # [x, anomaly] columns
        assert len(data) > 0

    def test_run_with_csv_input(
        self, tmp_path: Path, simple_model: MagneticModel
    ) -> None:
        """Test run command with CSV input file."""
        # Create CSV model file
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
-100.0,-50.0,0.0,25.0,50.0,100.0,150.0
"""
        csv_file.write_text(csv_content)

        result = runner.invoke(app, ["run", str(csv_file), "--no-plot"])

        assert result.exit_code == 0
        assert "Calculation complete" in result.stdout

    def test_run_verbose(self, model_json_file: Path) -> None:
        """Test run command with verbose output."""
        result = runner.invoke(
            app, ["run", str(model_json_file), "--verbose", "--no-plot"]
        )

        assert result.exit_code == 0
        assert "Loading model" in result.stdout
        assert "Calculating magnetic anomaly" in result.stdout
        assert "bodies" in result.stdout

    def test_run_nonexistent_file(self, tmp_path: Path) -> None:
        """Test run command with nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.json"

        result = runner.invoke(app, ["run", str(nonexistent)])

        assert result.exit_code != 0

    def test_run_invalid_json(self, tmp_path: Path) -> None:
        """Test run command with invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json{")

        result = runner.invoke(app, ["run", str(invalid_file), "--no-plot"])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestValidateCommand:
    """Tests for the 'validate' command."""

    def test_validate_valid_model(self, model_json_file: Path) -> None:
        """Test validate command with valid model."""
        result = runner.invoke(app, ["validate", str(model_json_file)])

        assert result.exit_code == 0
        assert "Model is valid" in result.stdout

    def test_validate_csv_model(self, tmp_path: Path) -> None:
        """Test validate command with CSV model file."""
        csv_file = tmp_path / "model.csv"
        csv_content = """50000.0,60.0,0.0,0.0
Rectangle,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
-100.0,-50.0,0.0,50.0,100.0
"""
        csv_file.write_text(csv_content)

        result = runner.invoke(app, ["validate", str(csv_file)])

        assert result.exit_code == 0
        assert "Model is valid" in result.stdout

    def test_validate_verbose(self, model_json_file: Path) -> None:
        """Test validate command with verbose output."""
        result = runner.invoke(app, ["validate", str(model_json_file), "--verbose"])

        assert result.exit_code == 0
        assert "bodies defined" in result.stdout
        assert "observation points" in result.stdout
        assert "Field intensity" in result.stdout

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test validate command with nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.json"

        result = runner.invoke(app, ["validate", str(nonexistent)])

        assert result.exit_code != 0

    def test_validate_invalid_json(self, tmp_path: Path) -> None:
        """Test validate command with invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json{")

        result = runner.invoke(app, ["validate", str(invalid_file)])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_validate_invalid_model(self, tmp_path: Path) -> None:
        """Test validate command with invalid model schema."""
        invalid_model = tmp_path / "invalid_model.json"
        with open(invalid_model, "w") as f:
            json.dump({"bodies": [], "field": {}}, f)  # Missing required fields

        result = runner.invoke(app, ["validate", str(invalid_model)])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_validate_gravity_model(self, gravity_model_json_file: Path) -> None:
        """Test validate command with gravity model."""
        result = runner.invoke(app, ["validate", str(gravity_model_json_file)])

        assert result.exit_code == 0
        assert "Model is valid" in result.stdout

    def test_validate_gravity_model_verbose(
        self, gravity_model_json_file: Path
    ) -> None:
        """Test validate gravity model with verbose output."""
        result = runner.invoke(
            app, ["validate", str(gravity_model_json_file), "--verbose"]
        )

        assert result.exit_code == 0
        assert "bodies defined" in result.stdout
        assert "observation points" in result.stdout
        assert "Gravity" in result.stdout

    def test_validate_heat_flow_model(self, heat_flow_model_json_file: Path) -> None:
        """Test validate command with heat flow model."""
        result = runner.invoke(app, ["validate", str(heat_flow_model_json_file)])

        assert result.exit_code == 0
        assert "Model is valid" in result.stdout

    def test_validate_heat_flow_model_verbose(
        self, heat_flow_model_json_file: Path
    ) -> None:
        """Test validate heat flow model with verbose output."""
        result = runner.invoke(
            app, ["validate", str(heat_flow_model_json_file), "--verbose"]
        )

        assert result.exit_code == 0
        assert "bodies defined" in result.stdout
        assert "observation points" in result.stdout
        assert "Heat Flow" in result.stdout
        assert "Background heat flow" in result.stdout


class TestRunNonMagneticModels:
    """Tests for run command with non-magnetic models."""

    def test_run_gravity_model_basic(self, gravity_model_json_file: Path) -> None:
        """Test that gravity model runs successfully."""
        result = runner.invoke(app, ["run", str(gravity_model_json_file), "--no-plot"])

        assert result.exit_code == 0
        assert "Calculation complete" in result.stdout

    def test_run_gravity_model_with_csv_output(
        self, gravity_model_json_file: Path, tmp_path: Path
    ) -> None:
        """Test gravity model run with CSV output."""
        output_csv = tmp_path / "gravity_output.csv"
        result = runner.invoke(
            app,
            [
                "run",
                str(gravity_model_json_file),
                "--output-csv",
                str(output_csv),
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert output_csv.exists()
        with open(output_csv) as f:
            lines = f.readlines()
            assert lines[0].strip() == "x_m,anomaly_mGal"
            assert len(lines) > 1

    def test_run_gravity_model_with_json_output(
        self, gravity_model_json_file: Path, tmp_path: Path
    ) -> None:
        """Test gravity model run with JSON output uses mGal label."""
        output_json = tmp_path / "gravity_output.json"
        result = runner.invoke(
            app,
            [
                "run",
                str(gravity_model_json_file),
                "--output-json",
                str(output_json),
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert output_json.exists()
        with open(output_json) as f:
            data = json.load(f)
        assert "anomaly_mGal" in data["results"]
        assert "anomaly_nT" not in data["results"]
        assert len(data["results"]["anomaly_mGal"]) == len(
            data["results"]["observation_x"]
        )

    def test_run_gravity_model_gz_gradient_component(
        self, gravity_model_json_file: Path
    ) -> None:
        """Test gravity model run selecting gz_gradient component."""
        result = runner.invoke(
            app,
            [
                "run",
                str(gravity_model_json_file),
                "--component",
                "gz_gradient",
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert "Calculation complete" in result.stdout

    def test_run_gravity_model_invalid_component(
        self, gravity_model_json_file: Path
    ) -> None:
        """Test that invalid magnetic component for gravity model raises error."""
        result = runner.invoke(
            app,
            ["run", str(gravity_model_json_file), "--component", "bz", "--no-plot"],
        )

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_run_gravity_model_with_plot(
        self, gravity_model_json_file: Path, tmp_path: Path
    ) -> None:
        """Test gravity model run with plot output."""
        output_plot = tmp_path / "gravity_plot.png"
        result = runner.invoke(
            app, ["run", str(gravity_model_json_file), "--plot", str(output_plot)]
        )

        assert result.exit_code == 0
        assert output_plot.exists()

    def test_run_heat_flow_model_basic(self, heat_flow_model_json_file: Path) -> None:
        """Test that heat flow model runs successfully."""
        result = runner.invoke(
            app, ["run", str(heat_flow_model_json_file), "--no-plot"]
        )

        assert result.exit_code == 0
        assert "Calculation complete" in result.output

    def test_run_heat_flow_model_output_csv(
        self,
        heat_flow_model_json_file: Path,
        tmp_path: Path,
        heat_flow_model: HeatFlowModel,
    ) -> None:
        """Test heat flow model run with CSV output."""
        output_csv = tmp_path / "heatflow.csv"
        result = runner.invoke(
            app,
            [
                "run",
                str(heat_flow_model_json_file),
                "--no-plot",
                "--output-csv",
                str(output_csv),
            ],
        )

        assert result.exit_code == 0
        assert output_csv.exists()

        # Verify CSV header and content
        with open(output_csv) as f:
            lines = f.readlines()
            assert lines[0].strip() == "x_m,anomaly_mW_m2"
            # Row count = header + data rows; should match observation points
            expected_rows = len(heat_flow_model.observation_x) + 1
            assert len(lines) == expected_rows

    def test_run_heat_flow_model_output_json(
        self,
        heat_flow_model_json_file: Path,
        tmp_path: Path,
        heat_flow_model: HeatFlowModel,
    ) -> None:
        """Test heat flow model run with JSON output."""
        output_json = tmp_path / "heatflow_output.json"
        result = runner.invoke(
            app,
            [
                "run",
                str(heat_flow_model_json_file),
                "--output-json",
                str(output_json),
                "--no-plot",
            ],
        )

        assert result.exit_code == 0
        assert output_json.exists()

        # Verify JSON structure and content
        with open(output_json) as f:
            data = json.load(f)

        assert "anomaly_mW_m2" in data["results"]
        assert len(data["results"]["anomaly_mW_m2"]) == len(
            data["results"]["observation_x"]
        )
        assert len(data["results"]["observation_x"]) == len(
            heat_flow_model.observation_x
        )

    def test_run_heat_flow_model_default_component(
        self, heat_flow_model_json_file: Path
    ) -> None:
        """Test that default component for heat flow is heat_flow."""
        result = runner.invoke(
            app,
            ["run", str(heat_flow_model_json_file), "--no-plot", "--verbose"],
        )

        assert result.exit_code == 0
        assert "heat_flow" in result.output

    def test_run_heat_flow_model_invalid_component(
        self, heat_flow_model_json_file: Path
    ) -> None:
        """Test invalid component for heat flow model raises error."""
        result = runner.invoke(
            app,
            [
                "run",
                str(heat_flow_model_json_file),
                "--no-plot",
                "--component",
                "bz",
            ],
        )

        assert result.exit_code != 0
        assert "not valid" in result.output.lower()


class TestVisualizeCommand:
    """Tests for the 'visualize' command."""

    def test_visualize_basic(
        self, results_json_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test basic visualize command."""
        # Mock plt.show to avoid displaying plot
        import matplotlib.pyplot as plt

        show_called: list[bool] = []

        def mock_show() -> None:
            show_called.append(True)  # type: ignore[reportUnknownMemberType]

        monkeypatch.setattr(plt, "show", mock_show)

        result = runner.invoke(app, ["visualize", str(results_json_file)])

        assert result.exit_code == 0
        assert "Visualization complete" in result.stdout

    def test_visualize_with_output(
        self, results_json_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test visualize command with output file."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        output_plot = tmp_path / "plot.pdf"

        result = runner.invoke(
            app, ["visualize", str(results_json_file), "--output", str(output_plot)]
        )

        assert result.exit_code == 0
        assert output_plot.exists()

    def test_visualize_with_custom_dpi(
        self, results_json_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test visualize command with custom DPI."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

        output_plot = tmp_path / "plot_hires.png"

        result = runner.invoke(
            app,
            [
                "visualize",
                str(results_json_file),
                "--output",
                str(output_plot),
                "--dpi",
                "300",
            ],
        )

        assert result.exit_code == 0
        assert output_plot.exists()

    def test_visualize_verbose(self, results_json_file: Path) -> None:
        """Test visualize command with verbose output."""
        result = runner.invoke(
            app, ["visualize", str(results_json_file), "--verbose", "--no-show"]
        )

        assert result.exit_code == 0
        assert "Loading results" in result.stdout
        assert "bodies" in result.stdout
        assert "Generating plot" in result.stdout

    def test_visualize_nonexistent_file(self, tmp_path: Path) -> None:
        """Test visualize command with nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.json"

        result = runner.invoke(app, ["visualize", str(nonexistent)])

        assert result.exit_code != 0

    def test_visualize_invalid_results(self, tmp_path: Path) -> None:
        """Test visualize command with invalid results file."""
        invalid_results = tmp_path / "invalid_results.json"
        with open(invalid_results, "w") as f:
            json.dump({"wrong": "format"}, f)

        result = runner.invoke(app, ["visualize", str(invalid_results), "--no-show"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_visualize_with_style(self, results_json_file: Path) -> None:
        """Test visualize command with custom style."""
        result = runner.invoke(
            app,
            [
                "visualize",
                str(results_json_file),
                "--style",
                "publication",
                "--no-show",
            ],
        )

        assert result.exit_code == 0
        assert "Visualization complete" in result.stdout


class TestConfigCommand:
    """Tests for the 'config' sub-commands."""

    def test_config_init_creates_project_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config init writes .forward-model.toml in the CWD."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        assert (tmp_path / ".forward-model.toml").exists()

    def test_config_init_user_creates_user_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config init --user writes to ~/.forward-model/config.toml."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        result = runner.invoke(app, ["config", "init", "--user"])
        assert result.exit_code == 0
        assert (fake_home / ".forward-model" / "config.toml").exists()

    def test_config_show_no_files_exits_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config show exits 0 and prints section headers even with no config files."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        # Clear any env vars that could interfere
        for var in ["FORWARD_MODEL_PLOT_STYLE", "FORWARD_MODEL_PLOT_DPI"]:
            monkeypatch.delenv(var, raising=False)
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "[plot]" in result.stdout
        assert "[field]" in result.stdout

    def test_config_show_displays_project_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config show reports values from a project config file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("FORWARD_MODEL_PLOT_STYLE", raising=False)
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "publication"\n')
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "publication" in result.stdout
        assert "project" in result.stdout

    def test_run_plot_style_from_config(
        self, model_json_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run command picks up plot style from config when no CLI flag is given."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("FORWARD_MODEL_PLOT_STYLE", raising=False)
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "publication"\n')
        output_plot = tmp_path / "out.png"
        result = runner.invoke(
            app, ["run", str(model_json_file), "--plot", str(output_plot)]
        )
        assert result.exit_code == 0
        assert output_plot.exists()

    def test_run_plot_style_cli_overrides_config(
        self, model_json_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit --plot-style CLI flag overrides the config file value."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("FORWARD_MODEL_PLOT_STYLE", raising=False)
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "publication"\n')
        output_plot = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "run",
                str(model_json_file),
                "--plot",
                str(output_plot),
                "--plot-style",
                "default",
            ],
        )
        assert result.exit_code == 0
        assert output_plot.exists()
