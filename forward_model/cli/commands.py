"""CLI commands for forward-model."""

# ruff: noqa: B008
# Typer requires function calls in defaults - this is the correct pattern

import json
import sys
from pathlib import Path

import typer
from pydantic import ValidationError

from forward_model import (
    calculate_anomaly,
    load_model,
    plot_combined,
    write_csv,
    write_json,
)

app = typer.Typer(
    name="forward-model",
    help="2D forward magnetic modeling using Talwani algorithm",
    add_completion=False,
)


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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Run forward model calculation.

    Loads a model from JSON, computes the magnetic anomaly, and optionally
    exports results and generates visualizations.
    """
    try:
        # Load model
        if verbose:
            typer.echo(f"Loading model from {input_file}...")
        model = load_model(input_file)
        if verbose:
            typer.echo(f"  Loaded {len(model.bodies)} bodies")
            typer.echo(f"  {len(model.observation_x)} observation points")

        # Calculate anomaly
        if verbose:
            typer.echo("Calculating magnetic anomaly...")
        anomaly = calculate_anomaly(model)
        if verbose:
            min_val = anomaly.min()
            max_val = anomaly.max()
            typer.echo(f"  Anomaly range: {min_val:.2f} to {max_val:.2f} nT")

        # Export results
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
            # Import here to avoid circular dependency during module load
            from forward_model.io.writers import write_numpy

            write_numpy(output_npy, model.observation_x, anomaly)

        # Generate plot
        if not no_plot:
            if verbose:
                typer.echo("Generating plot...")
            plot_combined(model, anomaly, save_path=plot)
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
    style: str = typer.Option(
        "default", "--style", help="Plot style (default, publication)"
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
            typer.echo(f"Generating plot with style '{style}'...")

        # Create plot with style and DPI parameters
        plot_combined(model, anomaly, save_path=output, style=style, dpi=dpi)

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
