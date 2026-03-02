"""Test that version strings stay synchronized."""

import tomllib
from pathlib import Path

import forward_model


def test_version_matches_pyproject():
    """Verify __version__ matches the version in pyproject.toml."""
    # Read version from pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    pyproject_version = pyproject["project"]["version"]

    # Compare with __version__
    assert forward_model.__version__ == pyproject_version, (
        f"Version mismatch: forward_model.__version__ = {forward_model.__version__}, "
        f"but pyproject.toml version = {pyproject_version}"
    )
