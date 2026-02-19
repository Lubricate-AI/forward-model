"""Unit tests for forward_model.config."""

from pathlib import Path

import pytest

from forward_model.config import (
    Config,
    FieldConfig,
    ObservationConfig,
    PlotConfig,
    config_from_dict,
    env_config,
    load_config,
    merge_configs,
    parse_toml_file,
    user_config_path,
)

# ---------------------------------------------------------------------------
# TestParseTomlFile
# ---------------------------------------------------------------------------


class TestParseTomlFile:
    def test_valid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[plot]\nstyle = "publication"\ndpi = 300\n')
        data = parse_toml_file(toml_file)
        assert data == {"plot": {"style": "publication", "dpi": 300}}

    def test_malformed_toml_raises_value_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("this is not = [valid toml\n")
        with pytest.raises(ValueError, match="Malformed TOML"):
            parse_toml_file(toml_file)

    def test_empty_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "empty.toml"
        toml_file.write_text("")
        data = parse_toml_file(toml_file)
        assert data == {}

    def test_partial_sections(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "partial.toml"
        toml_file.write_text("[field]\nintensity = 50000.0\n")
        data = parse_toml_file(toml_file)
        assert data == {"field": {"intensity": 50000.0}}


# ---------------------------------------------------------------------------
# TestConfigFromDict
# ---------------------------------------------------------------------------


class TestConfigFromDict:
    def test_empty_dict_yields_all_none(self) -> None:
        cfg = config_from_dict({})
        assert cfg.field.intensity is None
        assert cfg.plot.style is None
        assert cfg.output.directory is None

    def test_unknown_keys_are_ignored(self) -> None:
        cfg = config_from_dict({"unknown_section": {"foo": "bar"}})
        assert cfg == Config()

    def test_valid_field_section(self) -> None:
        cfg = config_from_dict({"field": {"intensity": 50000.0, "inclination": 60.0}})
        assert cfg.field.intensity == 50000.0
        assert cfg.field.inclination == 60.0
        assert cfg.field.declination is None

    def test_invalid_format_literal_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            config_from_dict({"output": {"format": "xlsx"}})

    def test_unknown_keys_within_section_ignored(self) -> None:
        cfg = config_from_dict({"plot": {"style": "default", "nonexistent": True}})
        assert cfg.plot.style == "default"


# ---------------------------------------------------------------------------
# TestMergeConfigs
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_later_config_wins(self) -> None:
        low = Config(plot=PlotConfig(style="default", dpi=100))
        high = Config(plot=PlotConfig(style="publication"))
        merged = merge_configs(low, high)
        assert merged.plot.style == "publication"
        # dpi from low survives because high didn't set it
        assert merged.plot.dpi == 100

    def test_none_does_not_override(self) -> None:
        base = Config(field=FieldConfig(intensity=50000.0))
        override = Config()  # all None
        merged = merge_configs(base, override)
        assert merged.field.intensity == 50000.0

    def test_three_layer_merge(self) -> None:
        user = Config(plot=PlotConfig(style="default", dpi=72))
        project = Config(plot=PlotConfig(style="publication"))
        env = Config(plot=PlotConfig(dpi=300))
        merged = merge_configs(user, project, env)
        assert merged.plot.style == "publication"  # project wins over user
        assert merged.plot.dpi == 300  # env wins over user

    def test_zero_value_survives(self) -> None:
        base = Config(observation=ObservationConfig(z=0.0))
        merged = merge_configs(Config(), base)
        assert merged.observation.z == 0.0


# ---------------------------------------------------------------------------
# TestEnvConfig
# ---------------------------------------------------------------------------


class TestEnvConfig:
    def test_float_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORWARD_MODEL_FIELD_INTENSITY", "32500.0")
        cfg = env_config()
        assert cfg.field.intensity == 32500.0

    def test_int_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORWARD_MODEL_PLOT_DPI", "300")
        cfg = env_config()
        assert cfg.plot.dpi == 300

    def test_path_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORWARD_MODEL_OUTPUT_DIRECTORY", "/tmp/results")
        cfg = env_config()
        assert cfg.output.directory == Path("/tmp/results")

    def test_invalid_value_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORWARD_MODEL_PLOT_DPI", "not-an-int")
        with pytest.raises(ValueError, match="FORWARD_MODEL_PLOT_DPI"):
            env_config()

    def test_no_vars_yields_all_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any FORWARD_MODEL_* vars that may exist in the environment
        env_vars = [
            "FORWARD_MODEL_FIELD_INTENSITY",
            "FORWARD_MODEL_FIELD_INCLINATION",
            "FORWARD_MODEL_FIELD_DECLINATION",
            "FORWARD_MODEL_OBSERVATION_Z",
            "FORWARD_MODEL_OUTPUT_DIRECTORY",
            "FORWARD_MODEL_OUTPUT_FORMAT",
            "FORWARD_MODEL_PLOT_STYLE",
            "FORWARD_MODEL_PLOT_DPI",
            "FORWARD_MODEL_PLOT_COLOR_BY",
            "FORWARD_MODEL_APP_PORT",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)
        cfg = env_config()
        assert cfg == Config()


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_no_files_yields_all_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure no env vars interfere
        for var in ["FORWARD_MODEL_PLOT_STYLE", "FORWARD_MODEL_PLOT_DPI"]:
            monkeypatch.delenv(var, raising=False)
        # Point user config to a non-existent path
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        cfg = load_config(cwd=tmp_path)
        assert cfg.plot.style is None
        assert cfg.field.intensity is None

    def test_project_config_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("FORWARD_MODEL_PLOT_STYLE", raising=False)
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "publication"\n')
        cfg = load_config(cwd=tmp_path)
        assert cfg.plot.style == "publication"

    def test_env_overrides_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "default"\n')
        monkeypatch.setenv("FORWARD_MODEL_PLOT_STYLE", "publication")
        cfg = load_config(cwd=tmp_path)
        assert cfg.plot.style == "publication"

    def test_malformed_toml_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / ".forward-model.toml").write_text("bad [ toml")
        with pytest.raises(ValueError, match="Malformed TOML"):
            load_config(cwd=tmp_path)

    def test_partial_config_leaves_unset_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("FORWARD_MODEL_PLOT_DPI", raising=False)
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "default"\n')
        cfg = load_config(cwd=tmp_path)
        assert cfg.plot.style == "default"
        assert cfg.plot.dpi is None

    def test_user_config_lower_priority_than_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        user_cfg_dir = home / ".forward-model"
        user_cfg_dir.mkdir(parents=True)
        (user_cfg_dir / "config.toml").write_text(
            '[plot]\nstyle = "default"\ndpi = 72\n'
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("FORWARD_MODEL_PLOT_STYLE", raising=False)
        monkeypatch.delenv("FORWARD_MODEL_PLOT_DPI", raising=False)
        (tmp_path / ".forward-model.toml").write_text('[plot]\nstyle = "publication"\n')
        cfg = load_config(cwd=tmp_path)
        # project wins over user for style
        assert cfg.plot.style == "publication"
        # dpi comes from user (project didn't set it)
        assert cfg.plot.dpi == 72


# ---------------------------------------------------------------------------
# TestUserConfigPath
# ---------------------------------------------------------------------------


class TestUserConfigPath:
    def test_returns_expected_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", "/fake/home")
        path = user_config_path()
        assert path == Path("/fake/home/.forward-model/config.toml")
