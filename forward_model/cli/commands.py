"""CLI commands for forward-model."""

# ruff: noqa: B008
# Typer requires function calls in defaults - this is the correct pattern

import json
import sys
from pathlib import Path
from typing import Literal

import typer
from pydantic import ValidationError

from forward_model import (
    calculate_anomaly,
    load_model,
    load_model_from_csv,
    plot_combined,
    write_csv,
    write_json,
    write_numpy,
)
from forward_model.compute.batch import batch_calculate
from forward_model.config import (
    TEMPLATE_TOML,
    load_config,
    load_config_with_sources,
    user_config_path,
)

app = typer.Typer(
    name="forward-model",
    help="2D forward magnetic modeling using Talwani algorithm",
    add_completion=False,
)

config_app = typer.Typer(name="config", help="Manage configuration settings")
app.add_typer(config_app)


@config_app.command("show")
def config_show() -> None:
    """Show current effective configuration and where each value comes from."""
    sources = load_config_with_sources()
    sections = ["field", "observation", "output", "plot", "app"]
    section_fields: dict[str, list[str]] = {
        "field": ["intensity", "inclination", "declination"],
        "observation": ["z"],
        "output": ["directory", "format"],
        "plot": ["style", "dpi", "color_by"],
        "app": ["port"],
    }

    for section in sections:
        typer.echo(f"[{section}]")
        section_data = sources.get(section, {})
        for field in section_fields[section]:
            if field in section_data:
                value, source = section_data[field]
                typer.echo(f"  {field} = {value}  <- {source}")
            else:
                typer.echo(f"  {field} = (not set)")
        typer.echo("")


@config_app.command("init")
def config_init(
    user: bool = typer.Option(
        False,
        "--user",
        help="Write to user-level config (~/.forward-model/config.toml)",
    ),
) -> None:
    """Create a commented configuration file with all available settings."""
    if user:
        target = user_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = Path.cwd() / ".forward-model.toml"

    if target.exists():
        overwrite = typer.confirm(f"{target} already exists. Overwrite?", default=False)
        if not overwrite:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    target.write_text(TEMPLATE_TOML)
    typer.echo(typer.style(f"✓ Created {target}", fg=typer.colors.GREEN))


@app.command("run")
def run(
    input_file: Path = typer.Argument(
        ..., help="Path to input JSON model file", exists=True, dir_okay=False
    ),
    output_csv: Path | None = typer.Option(
        None, "--output-csv", help="Save results to CSV file"
    ),
    output_json: Path | None = typer.Option(
        None, "--output-json", help="Save results to JSON file"
    ),
    output_npy: Path | None = typer.Option(
        None, "--output-npy", help="Save results to NumPy file (.npy or .npz)"
    ),
    plot: Path | None = typer.Option(
        None, "--plot", help="Save plot to file (e.g., output.png)"
    ),
    no_plot: bool = typer.Option(
        False, "--no-plot", help="Skip plot generation entirely"
    ),
    component: Literal[
        "bz", "bx", "total_field", "amplitude", "gradient"
    ] = typer.Option(
        "bz",
        "--component",
        help="Anomaly component: bz (default), bx, total_field, amplitude, gradient",
    ),
    plot_style: str | None = typer.Option(
        None, "--plot-style", help="Plot style (default, publication)"
    ),
    plot_dpi: int | None = typer.Option(
        None, "--plot-dpi", help="DPI for saved plot figure"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Run forward model calculation.

    Loads a model from JSON, computes the magnetic anomaly, and optionally
    exports results and generates visualizations.
    """
    try:
        cfg = load_config()
        effective_style = plot_style or cfg.plot.style or "default"
        effective_dpi = plot_dpi if plot_dpi is not None else cfg.plot.dpi

        # Load model (auto-detect format from extension)
        if verbose:
            typer.echo(f"Loading model from {input_file}...")

        if input_file.suffix.lower() == ".csv":
            model = load_model_from_csv(input_file)
        else:
            model = load_model(input_file)
        if verbose:
            typer.echo(f"  Loaded {len(model.bodies)} bodies")
            typer.echo(f"  {len(model.observation_x)} observation points")

        # Calculate all anomaly components in one pass
        if verbose:
            typer.echo("Calculating magnetic anomaly...")
        all_components = calculate_anomaly(model, component="all")
        anomaly = {
            "bz": all_components.bz,
            "bx": all_components.bx,
            "total_field": all_components.total_field,
            "amplitude": all_components.amplitude,
            "gradient": all_components.gradient,
        }[component]
        if verbose:
            min_val = float(anomaly.min())
            max_val = float(anomaly.max())
            typer.echo(f"  {component} range: {min_val:.2f} to {max_val:.2f}")

        # Export results (uses the selected component)
        if output_csv:
            if verbose:
                typer.echo(f"Writing CSV to {output_csv}...")
            write_csv(output_csv, model.observation_x, anomaly)

        if output_json:
            if verbose:
                typer.echo(f"Writing JSON to {output_json}...")
            write_json(output_json, model, anomaly)

        if output_npy:
            if verbose:
                typer.echo(f"Writing NumPy to {output_npy}...")
            write_numpy(output_npy, model.observation_x, anomaly)

        # Generate plot — always shows TMI on primary axis with gradient overlay
        if not no_plot:
            if verbose:
                typer.echo("Generating plot...")
            plot_combined(
                model,
                all_components.total_field,
                save_path=plot,
                component="total_field",
                gradient=all_components.gradient,
                style=effective_style,
                dpi=effective_dpi,
            )
            if plot:
                if verbose:
                    typer.echo(f"  Plot saved to {plot}")
            else:
                if verbose:
                    typer.echo("  Displaying plot...")
                import matplotlib.pyplot as plt

                plt.show()  # type: ignore[reportUnknownMemberType]

        typer.echo(typer.style("✓ Calculation complete", fg=typer.colors.GREEN))

    except FileNotFoundError as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)
    except ValueError as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(typer.style(f"Unexpected error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)


@app.command("validate")
def validate(
    input_file: Path = typer.Argument(
        ..., help="Path to input JSON model file", exists=True, dir_okay=False
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Validate a model file without running calculations.

    Checks that the JSON file is valid and conforms to the model schema.
    Returns exit code 0 if valid, 1 if invalid.
    """
    try:
        if verbose:
            typer.echo(f"Validating {input_file}...")

        # Auto-detect format from extension
        if input_file.suffix.lower() == ".csv":
            model = load_model_from_csv(input_file)
        else:
            model = load_model(input_file)

        typer.echo(typer.style("✓ Model is valid", fg=typer.colors.GREEN))
        if verbose:
            typer.echo(f"  {len(model.bodies)} bodies defined")
            typer.echo(f"  {len(model.observation_x)} observation points")
            typer.echo(f"  Field intensity: {model.field.intensity:.1f} nT")
            typer.echo(f"  Field inclination: {model.field.inclination:.1f}°")
            typer.echo(f"  Field declination: {model.field.declination:.1f}°")

        sys.exit(0)

    except FileNotFoundError as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)
    except ValueError as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(typer.style(f"Unexpected error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)


@app.command("visualize")
def visualize(
    results_file: Path = typer.Argument(
        ..., help="Path to results JSON file", exists=True, dir_okay=False
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save plot to file (e.g., output.pdf)"
    ),
    style: str | None = typer.Option(
        None, "--style", help="Plot style (default, publication)"
    ),
    dpi: int | None = typer.Option(None, "--dpi", help="DPI for saved figure"),
    no_show: bool = typer.Option(
        False, "--no-show", help="Don't display plot (only save)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Visualize results from a previously computed model.

    Loads results from a JSON file and generates a combined plot with
    cross-section and anomaly profile.
    """
    try:
        cfg = load_config()
        effective_style = style or cfg.plot.style or "default"
        effective_dpi = dpi if dpi is not None else cfg.plot.dpi

        # Load results JSON
        if verbose:
            typer.echo(f"Loading results from {results_file}...")

        with open(results_file) as f:
            data = json.load(f)

        # Extract model and results
        if "model" not in data or "results" not in data:
            raise ValueError(
                "Invalid results file: must contain 'model' and 'results' sections"
            )

        # Reconstruct model and anomaly
        import numpy as np

        from forward_model.models import ForwardModel

        model = ForwardModel.model_validate(data["model"])
        anomaly = np.array(data["results"]["anomaly_nT"])

        if verbose:
            typer.echo(f"  Loaded {len(model.bodies)} bodies")
            min_val = anomaly.min()
            max_val = anomaly.max()
            typer.echo(f"  Anomaly range: {min_val:.2f} to {max_val:.2f} nT")

        # Generate plot
        if verbose:
            typer.echo(f"Generating plot with style '{effective_style}'...")

        # Create plot with style and DPI parameters
        plot_combined(
            model, anomaly, save_path=output, style=effective_style, dpi=effective_dpi
        )

        if output:
            if verbose:
                typer.echo(f"  Plot saved to {output}")

        if not no_show:
            if verbose:
                typer.echo("  Displaying plot...")
            import matplotlib.pyplot as plt

            plt.show()  # type: ignore[reportUnknownMemberType]

        typer.echo(typer.style("✓ Visualization complete", fg=typer.colors.GREEN))

    except FileNotFoundError as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)
    except (ValueError, ValidationError, KeyError) as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(typer.style(f"Unexpected error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)


@app.command("batch")
def batch(
    models: list[Path] = typer.Argument(
        ...,
        help="Model file paths (JSON or CSV)",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Output directory"
    ),
    fmt: Literal["csv", "json", "npy"] | None = typer.Option(
        None, "--format", help="Output format: csv, json, npy"
    ),
    plot: bool = typer.Option(False, "--plot", help="Save a plot for each model"),
    parallel: bool = typer.Option(
        False, "--parallel", help="Process models concurrently"
    ),
    workers: int | None = typer.Option(
        None, "--workers", help="Number of parallel workers"
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--no-continue-on-error",
        help="Continue processing after errors",
    ),
    summary: bool = typer.Option(
        False, "--summary", help="Write batch_summary.csv with cross-model statistics"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Run forward model calculation on multiple model files.

    Processes each model through the full load → compute → export pipeline.
    Results are written to OUTPUT_DIR with filenames matching input stems.
    """
    cfg = load_config()
    effective_output_dir = output_dir or cfg.output.directory or Path("results")
    effective_fmt: Literal["csv", "json", "npy"] = (
        fmt or cfg.output.format or "csv"  # type: ignore[assignment]
    )

    if verbose:
        typer.echo(
            f"Processing {len(models)} model(s)"
            f" → {effective_output_dir} ({effective_fmt})"
        )

    try:
        result = batch_calculate(
            model_paths=models,
            output_dir=effective_output_dir,
            fmt=effective_fmt,
            parallel=parallel,
            max_workers=workers,
            continue_on_error=continue_on_error,
            write_summary=summary,
            plot=plot,
        )
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)

    if verbose:
        for path in result.succeeded:
            typer.echo(typer.style(f"  ✓ {path}", fg=typer.colors.GREEN))
        for path, err in result.failed.items():
            typer.echo(typer.style(f"  ✗ {path}: {err}", fg=typer.colors.RED))

    n_ok = len(result.succeeded)
    n_fail = len(result.failed)
    typer.echo(f"Batch complete: {n_ok} succeeded, {n_fail} failed")

    if result.summary is not None:
        typer.echo(f"Summary written to {effective_output_dir / 'batch_summary.csv'}")

    if n_fail > 0:
        sys.exit(1)
