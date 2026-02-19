"""Unit tests for batch processing."""

import json
from pathlib import Path

import numpy as np
import pytest

from forward_model.compute.batch import BatchResult, batch_calculate


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


def _write_model(
    path: Path,
    observation_x: list[float],
    susceptibility: float = 0.05,
) -> Path:
    """Write a minimal model JSON file and return the path."""
    path.write_text(json.dumps(_make_model_dict(observation_x, susceptibility)))
    return path


class TestBatchCalculate:
    """Tests for batch_calculate()."""

    def test_successful_batch_writes_files(self, tmp_path: Path) -> None:
        """Two valid model files are processed; BatchResult reflects both succeed."""
        obs_x = [-100.0, 0.0, 100.0]
        m1 = _write_model(tmp_path / "model_a.json", obs_x)
        m2 = _write_model(tmp_path / "model_b.json", obs_x)
        out = tmp_path / "results"

        result = batch_calculate([m1, m2], output_dir=out)

        assert (out / "model_a.csv").exists()
        assert (out / "model_b.csv").exists()
        assert len(result.succeeded) == 2
        assert len(result.failed) == 0

    def test_failed_model_recorded_in_failed(self, tmp_path: Path) -> None:
        """Invalid model is recorded in failed dict with non-empty error message."""
        obs_x = [-100.0, 0.0, 100.0]
        m_good = _write_model(tmp_path / "good.json", obs_x)
        m_bad = tmp_path / "bad.json"
        m_bad.write_text("not valid json{")
        out = tmp_path / "results"

        result = batch_calculate(
            [m_good, m_bad], output_dir=out, continue_on_error=True
        )

        assert len(result.succeeded) == 1
        assert str(m_bad) in result.failed
        assert len(result.failed[str(m_bad)]) > 0

    def test_continue_on_error_false_raises(self, tmp_path: Path) -> None:
        """Invalid model with continue_on_error=False raises an exception."""
        m_bad = tmp_path / "bad.json"
        m_bad.write_text("not valid json{")
        out = tmp_path / "results"

        with pytest.raises(ValueError):
            batch_calculate([m_bad], output_dir=out, continue_on_error=False)

    def test_summary_statistics_correct(self, tmp_path: Path) -> None:
        """Summary stats are mathematically correct when grids match."""
        obs_x = [-100.0, 0.0, 100.0]
        m1 = _write_model(tmp_path / "m1.json", obs_x, susceptibility=0.05)
        m2 = _write_model(tmp_path / "m2.json", obs_x, susceptibility=0.10)
        out = tmp_path / "results"

        result = batch_calculate([m1, m2], output_dir=out, write_summary=True)

        assert result.summary is not None
        summary_csv = out / "batch_summary.csv"
        assert summary_csv.exists()

        lines = summary_csv.read_text().splitlines()
        assert lines[0] == "x_m,mean_nT,min_nT,max_nT,std_nT"
        assert len(lines) == len(obs_x) + 1  # header + data rows

        from forward_model.compute import calculate_anomaly
        from forward_model.io import load_model

        a1 = calculate_anomaly(load_model(m1))
        a2 = calculate_anomaly(load_model(m2))
        expected_mean = (a1 + a2) / 2

        means = result.summary[:, 1]
        np.testing.assert_allclose(means, expected_mean, rtol=1e-6)

    def test_summary_skipped_when_grids_differ(self, tmp_path: Path) -> None:
        """Summary is None when models have different observation_x."""
        m1 = _write_model(tmp_path / "m1.json", [-100.0, 0.0, 100.0])
        m2 = _write_model(tmp_path / "m2.json", [-50.0, 0.0, 50.0])
        out = tmp_path / "results"

        result = batch_calculate([m1, m2], output_dir=out, write_summary=True)

        assert result.summary is None
        assert not (out / "batch_summary.csv").exists()

    def test_parallel_produces_same_results(self, tmp_path: Path) -> None:
        """Parallel and sequential runs produce identical output files."""
        obs_x = [-100.0, 0.0, 100.0]
        m1 = _write_model(tmp_path / "m1.json", obs_x, susceptibility=0.05)
        m2 = _write_model(tmp_path / "m2.json", obs_x, susceptibility=0.08)

        out_seq = tmp_path / "sequential"
        out_par = tmp_path / "parallel"

        batch_calculate([m1, m2], output_dir=out_seq, parallel=False)
        batch_calculate([m1, m2], output_dir=out_par, parallel=True)

        for stem in ["m1", "m2"]:
            seq_content = (out_seq / f"{stem}.csv").read_text()
            par_content = (out_par / f"{stem}.csv").read_text()
            assert seq_content == par_content

    def test_output_stems_match_input_stems(self, tmp_path: Path) -> None:
        """Output file stem matches the input file stem."""
        m = _write_model(tmp_path / "dyke.json", [-100.0, 0.0, 100.0])
        out = tmp_path / "results"

        result = batch_calculate([m], output_dir=out, fmt="csv")

        assert (out / "dyke.csv").exists()
        assert str(m) in result.succeeded

    def test_output_dir_created_if_absent(self, tmp_path: Path) -> None:
        """Output directory is created automatically if it does not exist."""
        m = _write_model(tmp_path / "model.json", [-100.0, 0.0, 100.0])
        out = tmp_path / "nested" / "output"

        assert not out.exists()
        batch_calculate([m], output_dir=out)
        assert out.exists()

    def test_batch_result_is_dataclass(self) -> None:
        """BatchResult can be instantiated with defaults."""
        r = BatchResult()
        assert r.succeeded == []
        assert r.failed == {}
        assert r.summary is None
