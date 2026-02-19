"""Configuration file support for forward-model.

Supports a layered configuration system (highest priority wins):
1. Explicit CLI flags
2. Environment variables (FORWARD_MODEL_*)
3. Project-level config: .forward-model.toml in CWD
4. User-level config: ~/.forward-model/config.toml
5. Built-in defaults (all None)
"""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FieldConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intensity: float | None = None
    inclination: float | None = None
    declination: float | None = None


class ObservationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    z: float | None = None


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    directory: Path | None = None
    format: Literal["csv", "json", "npy"] | None = None


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    style: str | None = None
    dpi: int | None = None
    color_by: Literal["index", "susceptibility"] | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    port: int | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: FieldConfig = FieldConfig()
    observation: ObservationConfig = ObservationConfig()
    output: OutputConfig = OutputConfig()
    plot: PlotConfig = PlotConfig()
    app: AppConfig = AppConfig()


TEMPLATE_TOML: str = """\
# forward-model configuration file
# Place this file as .forward-model.toml in your project directory
# or as ~/.forward-model/config.toml for user-level defaults.
#
# All settings are optional. Uncomment and set values as needed.

# [field]
# Earth's magnetic field parameters
# intensity = 50000.0       # Field intensity in nanoTeslas (nT)
# inclination = 60.0        # Field inclination in degrees
# declination = 0.0         # Field declination in degrees

# [observation]
# Observation surface parameters
# z = 0.0                   # Observation height/depth in metres

# [output]
# Default output settings for batch processing
# directory = "results"     # Output directory path
# format = "csv"            # Output format: csv, json, or npy

# [plot]
# Visualization defaults
# style = "default"         # Plot style: default, publication, or presentation
# dpi = 150                 # Resolution for saved figures (dots per inch)
# color_by = "index"        # Body colour scheme: index or susceptibility

# [app]
# Application server settings
# port = 8080               # Port for the web app server
"""


def user_config_path() -> Path:
    """Return path to user-level config file."""
    return Path.home() / ".forward-model" / "config.toml"


def parse_toml_file(path: Path) -> dict[str, object]:
    """Parse a TOML file, raising ValueError on malformed content."""
    with open(path, "rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Malformed TOML in {path}: {e}") from e


def config_from_dict(data: dict[str, object]) -> Config:
    """Build a Config from a raw dict (e.g., parsed TOML)."""
    return Config.model_validate(data)


def merge_configs(*configs: Config) -> Config:
    """Merge configs in order; later configs take priority over earlier ones.

    None values never override previously-set values.
    """
    merged_field: dict[str, object] = {}
    merged_observation: dict[str, object] = {}
    merged_output: dict[str, object] = {}
    merged_plot: dict[str, object] = {}
    merged_app: dict[str, object] = {}

    for cfg in configs:
        merged_field.update(cfg.field.model_dump(exclude_none=True))
        merged_observation.update(cfg.observation.model_dump(exclude_none=True))
        merged_output.update(cfg.output.model_dump(exclude_none=True))
        merged_plot.update(cfg.plot.model_dump(exclude_none=True))
        merged_app.update(cfg.app.model_dump(exclude_none=True))

    return Config(
        field=FieldConfig.model_validate(merged_field),
        observation=ObservationConfig.model_validate(merged_observation),
        output=OutputConfig.model_validate(merged_output),
        plot=PlotConfig.model_validate(merged_plot),
        app=AppConfig.model_validate(merged_app),
    )


# Map of env var name → (section, field, cast_function)
_ENV_VAR_MAP: dict[str, tuple[str, str, type]] = {
    "FORWARD_MODEL_FIELD_INTENSITY": ("field", "intensity", float),
    "FORWARD_MODEL_FIELD_INCLINATION": ("field", "inclination", float),
    "FORWARD_MODEL_FIELD_DECLINATION": ("field", "declination", float),
    "FORWARD_MODEL_OBSERVATION_Z": ("observation", "z", float),
    "FORWARD_MODEL_OUTPUT_DIRECTORY": ("output", "directory", Path),
    "FORWARD_MODEL_OUTPUT_FORMAT": ("output", "format", str),
    "FORWARD_MODEL_PLOT_STYLE": ("plot", "style", str),
    "FORWARD_MODEL_PLOT_DPI": ("plot", "dpi", int),
    "FORWARD_MODEL_PLOT_COLOR_BY": ("plot", "color_by", str),
    "FORWARD_MODEL_APP_PORT": ("app", "port", int),
}


def env_config() -> Config:
    """Build a Config from FORWARD_MODEL_* environment variables."""
    sections: dict[str, dict[str, object]] = {}

    for env_var, (section, field, cast) in _ENV_VAR_MAP.items():
        raw = os.environ.get(env_var)
        if raw is None:
            continue
        try:
            value: object = cast(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for {env_var}={raw!r}: {e}") from e
        sections.setdefault(section, {})[field] = value

    # Build Config section by section; wrap ValidationError so the env var is named
    from pydantic import ValidationError

    try:
        return Config(
            field=FieldConfig.model_validate(sections.get("field", {})),
            observation=ObservationConfig.model_validate(
                sections.get("observation", {})
            ),
            output=OutputConfig.model_validate(sections.get("output", {})),
            plot=PlotConfig.model_validate(sections.get("plot", {})),
            app=AppConfig.model_validate(sections.get("app", {})),
        )
    except ValidationError as e:
        # Surface which env var(s) caused the validation failure
        bad = [
            f"{var}={os.environ[var]!r}" for var in _ENV_VAR_MAP if var in os.environ
        ]
        raise ValueError(
            f"Invalid environment variable value(s): {', '.join(bad)}\n{e}"
        ) from e


def load_config(cwd: Path | None = None) -> Config:
    """Load and merge configuration from all sources.

    Priority (highest to lowest):
    1. Environment variables
    2. Project-level .forward-model.toml
    3. User-level ~/.forward-model/config.toml
    """
    effective_cwd = cwd if cwd is not None else Path.cwd()
    project_config_path = effective_cwd / ".forward-model.toml"
    user_cfg_path = user_config_path()

    layers: list[Config] = [Config()]  # base: all None

    if user_cfg_path.exists():
        data = parse_toml_file(user_cfg_path)
        layers.append(config_from_dict(data))

    if project_config_path.exists():
        data = parse_toml_file(project_config_path)
        layers.append(config_from_dict(data))

    layers.append(env_config())

    return merge_configs(*layers)


def load_config_with_sources(
    cwd: Path | None = None,
) -> dict[str, dict[str, tuple[object, str]]]:
    """Load config and track which source provided each value.

    Returns a dict keyed by section name, each mapping field name to
    (value, source_label). Fields not set in any source are omitted.
    """
    effective_cwd = cwd if cwd is not None else Path.cwd()
    project_config_path = effective_cwd / ".forward-model.toml"
    user_cfg_path = user_config_path()

    # (config, source_label) pairs, lowest priority first
    candidates: list[tuple[Config, str]] = []

    if user_cfg_path.exists():
        data = parse_toml_file(user_cfg_path)
        candidates.append((config_from_dict(data), "user"))

    if project_config_path.exists():
        data = parse_toml_file(project_config_path)
        candidates.append((config_from_dict(data), "project"))

    candidates.append((env_config(), "env"))

    sections = ["field", "observation", "output", "plot", "app"]
    result: dict[str, dict[str, tuple[object, str]]] = {s: {} for s in sections}

    for cfg, source in candidates:
        for section in sections:
            section_model = getattr(cfg, section)
            for field, value in section_model.model_dump(exclude_none=True).items():
                result[section][field] = (value, source)

    return result
