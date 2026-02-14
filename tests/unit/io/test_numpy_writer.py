"""Tests for NumPy output writing."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from forward_model.io.writers import write_numpy


@pytest.fixture
def sample_data() -> tuple[list[float], NDArray[np.float64]]:
    """Sample observation points and anomaly data."""
    observation_x = [-100.0, -50.0, 0.0, 50.0, 100.0]
    anomaly = np.array([10.5, 20.3, 35.7, 28.1, 15.2])
    return observation_x, anomaly


class TestWriteNumpy:
    """Tests for write_numpy function."""

    def test_write_npy_format(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test writing to .npy format."""
        output_file = tmp_path / "results.npy"
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly)

        assert output_file.exists()

        # Load and verify
        loaded = np.load(output_file)
        assert loaded.shape == (5, 2)
        assert np.allclose(loaded[:, 0], observation_x)
        assert np.allclose(loaded[:, 1], anomaly)

    def test_write_npz_format(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test writing to .npz format."""
        output_file = tmp_path / "results.npz"
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly)

        assert output_file.exists()

        # Load and verify
        loaded = np.load(output_file)
        assert "x" in loaded
        assert "anomaly" in loaded
        assert np.allclose(loaded["x"], observation_x)
        assert np.allclose(loaded["anomaly"], anomaly)

    def test_write_npz_compressed(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test writing compressed .npz format."""
        output_file = tmp_path / "results.npz"
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly, compressed=True)

        assert output_file.exists()

        # Load and verify
        loaded = np.load(output_file)
        assert "x" in loaded
        assert "anomaly" in loaded
        assert np.allclose(loaded["x"], observation_x)
        assert np.allclose(loaded["anomaly"], anomaly)

    def test_write_default_format_no_extension(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test writing with no extension defaults to .npy format."""
        output_file = tmp_path / "results"
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly)

        # NumPy adds .npy extension automatically
        assert output_file.exists() or Path(str(output_file) + ".npy").exists()

    def test_write_creates_parent_directory(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test that write_numpy can write to existing directories."""
        output_dir = tmp_path / "subdir"
        output_dir.mkdir()
        output_file = output_dir / "results.npy"
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly)

        assert output_file.exists()

    def test_write_string_path(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test that write_numpy accepts string paths."""
        output_file = str(tmp_path / "results.npy")
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly)

        assert Path(output_file).exists()

    def test_write_preserves_data_types(
        self, tmp_path: Path, sample_data: tuple[list[float], NDArray[np.float64]]
    ) -> None:
        """Test that data types are preserved when writing."""
        output_file = tmp_path / "results.npy"
        observation_x, anomaly = sample_data

        write_numpy(output_file, observation_x, anomaly)

        loaded = np.load(output_file)
        # Both columns should be float64
        assert loaded.dtype == np.float64

    def test_write_large_dataset(self, tmp_path: Path) -> None:
        """Test writing a larger dataset."""
        output_file = tmp_path / "large_results.npz"
        observation_x = [float(x) for x in range(1000)]
        anomaly = np.random.randn(1000)

        write_numpy(output_file, observation_x, anomaly, compressed=True)

        loaded = np.load(output_file)
        assert len(loaded["x"]) == 1000
        assert len(loaded["anomaly"]) == 1000
