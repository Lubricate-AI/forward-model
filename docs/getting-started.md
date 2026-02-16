# Getting Started

This guide will help you install Forward Model and run your first magnetic anomaly calculation.

## Installation

### From PyPI (Recommended)

```bash
pip install forward-model
```

!!! note
    Forward Model requires Python 3.12 or higher.

### From Source

For development or to use the latest features:

```bash
git clone https://github.com/Lubricate-AI/forward-model.git
cd forward-model
make install
```

This will install the package using [uv](https://github.com/astral-sh/uv), a fast Python package manager.

## Verify Installation

Check that the installation was successful:

```bash
forward-model --help
```

You should see the command-line interface help message.

## Your First Model

Let's create a simple model of a vertical dyke and calculate its magnetic anomaly.

### Create a Model File

Create a file named `my_dyke.json`:

```json
{
  "bodies": [
    {
      "name": "Vertical Dyke",
      "susceptibility": 0.08,
      "vertices": [[20.0, 50.0], [30.0, 50.0], [30.0, 200.0], [20.0, 200.0]]
    }
  ],
  "field": {
    "intensity": 50000.0,
    "inclination": 60.0,
    "declination": 0.0
  },
  "observation_x": [-100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
  "observation_z": 0.0
}
```

This defines:

- A **rectangular dyke** from x=20 to x=30 meters, extending from z=50 to z=200 meters depth
- **Magnetic susceptibility** of 0.08 SI (typical for mafic rocks)
- **Earth's magnetic field** with 50,000 nT intensity at 60° inclination
- **Observation points** along a profile from x=-100 to x=150 meters at the surface (z=0)

### Run the Calculation

#### Using the CLI

```bash
forward-model run my_dyke.json --plot dyke_anomaly.png --verbose
```

Output:
```
Loading model from my_dyke.json...
  Loaded 1 bodies
  11 observation points
Calculating magnetic anomaly...
  Anomaly range: -45.23 to 287.56 nT
Generating plot...
  Plot saved to dyke_anomaly.png
✓ Calculation complete
```

#### Using Python

```python
from forward_model import load_model, calculate_anomaly, plot_combined

# Load the model
model = load_model("my_dyke.json")

# Calculate the anomaly
anomaly = calculate_anomaly(model)

# Create visualization
fig = plot_combined(model, anomaly, save_path="dyke_anomaly.png")

# Access the results
print(f"Anomaly range: {anomaly.min():.2f} to {anomaly.max():.2f} nT")
```

### Understanding the Output

The calculation produces a plot with two panels:

1. **Top panel**: Cross-section showing the geological body geometry
2. **Bottom panel**: Calculated magnetic anomaly profile

The magnetic anomaly shows the characteristic dipolar signature of a vertical body magnetized by Earth's field.

## Next Steps

Now that you have Forward Model working, you can:

- **[Explore the CLI](user-guide/cli.md)** - Learn about all available commands and options
- **[Customize visualizations](user-guide/visualization.md)** - Create publication-quality figures
- **[Try more examples](user-guide/examples.md)** - See models of different geological scenarios
- **[Read the API docs](api-reference/index.md)** - Integrate Forward Model into your workflows

## Common Issues

### Python Version Error

If you see an error about Python version:

```
Error: Python 3.12 or higher is required
```

Install a newer Python version from [python.org](https://www.python.org/downloads/).

### Module Not Found

If the `forward-model` command is not found after installation:

1. Make sure your Python scripts directory is in your PATH
2. Alternatively, use: `python -m forward_model` instead of `forward-model`

### File Not Found

If you get "File not found" errors:

- Ensure you're in the correct directory
- Use absolute paths: `forward-model run /path/to/my_dyke.json`
- Check that the file extension is `.json` (not `.txt`)

## Getting Help

- Use `forward-model --help` to see all available commands
- Use `forward-model COMMAND --help` for help on specific commands
- Report issues at [GitHub Issues](https://github.com/Lubricate-AI/forward-model/issues)
