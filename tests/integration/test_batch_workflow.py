"""Integration tests for batch workflow."""

import json
from pathlib import Path

from typer.testing import CliRunner

from forward_model.cli.commands import app
from forward_model.compute.batch import batch_calculate

runner = CliRunner()


def _make_model_dict(
    observation_x: list[float],
    susceptibility: float = 0.05,
) -> dict[str, object]:
    """Create a minimal model JSON dict."""
    return {
        "bodies": [
            {
                "name": "Test Body",
                "susceptibility": susceptibility,
                "vertices": [
                    [0.0, 100.0],
                    [100.0, 100.0],
                    [100.0, 200.0],
                    [0.0, 200.0],
                ],
            }
        ],
        "field": {
            "intensity": 50000.0,
            "inclination": 60.0,
            "declination": 0.0,
        },
        "observation_x": observation_x,
        "observation_z": 0.0,
    }


class TestBatchWorkflow:
    """End-to-end tests for batch processing."""

    def test_full_batch_workflow(self, tmp_path: Path) -> None:
        """Three models → three CSVs + batch_summary.csv."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        obs_x = [-100.0, -50.0, 0.0, 50.0, 100.0]

        susceptibilities = [0.03, 0.05, 0.08]
        model_paths: list[Path] = [models_dir / f"model_{i}.json" for i in range(1, 4)]
        for path, chi in zip(model_paths, susceptibilities, strict=True):
            path.write_text(json.dumps(_make_model_dict(obs_x, chi)))

        results_dir = tmp_path / "results"
        result = batch_calculate(
            model_paths=model_paths,
            output_dir=results_dir,
            fmt="csv",
            write_summary=True,
        )

        for i in range(1, 4):
            assert (results_dir / f"model_{i}.csv").exists()

        summary_csv = results_dir / "batch_summary.csv"
        assert summary_csv.exists()
        lines = summary_csv.read_text().splitlines()
        assert lines[0] == "x_m,mean_nT,min_nT,max_nT,std_nT"
        assert len(lines) == len(obs_x) + 1

        assert len(result.succeeded) == 3
        assert len(result.failed) == 0

    def test_batch_cli_basic(self, tmp_path: Path) -> None:
        """Batch subcommand processes models and writes output files."""
        obs_x = [-100.0, 0.0, 100.0]
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        m1.write_text(json.dumps(_make_model_dict(obs_x, 0.05)))
        m2.write_text(json.dumps(_make_model_dict(obs_x, 0.08)))
        out = tmp_path / "results"

        result = runner.invoke(
            app,
            [
                "batch",
                str(m1),
                str(m2),
                "--output-dir",
                str(out),
                "--format",
                "csv",
                "--summary",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out / "m1.csv").exists()
        assert (out / "m2.csv").exists()
        assert (out / "batch_summary.csv").exists()
        assert "succeeded" in result.stdout

    def test_batch_cli_invalid_format(self, tmp_path: Path) -> None:
        """CLI exits 2 with error when format is invalid (Typer rejects non-Literal value)."""
        m = tmp_path / "m.json"
        m.write_text(json.dumps(_make_model_dict([-100.0, 0.0, 100.0])))

        result = runner.invoke(
            app,
            [
                "batch",
                str(m),
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "xml",
            ],
        )

        assert result.exit_code == 2
        assert "Error" in result.output

    def test_batch_cli_missing_file(self, tmp_path: Path) -> None:
        """CLI exits 1 when a model path does not exist."""
        missing = tmp_path / "does_not_exist.json"

        result = runner.invoke(
            app,
            [
                "batch",
                str(missing),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0

    def test_batch_cli_verbose(self, tmp_path: Path) -> None:
        """Verbose mode prints per-model status lines."""
        obs_x = [-100.0, 0.0, 100.0]
        m1 = tmp_path / "m1.json"
        m1.write_text(json.dumps(_make_model_dict(obs_x)))
        out = tmp_path / "results"

        result = runner.invoke(
            app,
            [
                "batch",
                str(m1),
                "--output-dir",
                str(out),
                "--verbose",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Processing" in result.stdout
