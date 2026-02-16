# CLI Usage

Forward Model provides a command-line interface for running magnetic anomaly calculations, validating models, and generating visualizations.

## Installation Methods

The CLI can be invoked in two ways:

### Option 1: Console Script (After pip install)

```bash
forward-model COMMAND [OPTIONS]
```

This is the recommended method after installing with `pip install forward-model`.

### Option 2: Python Module (Direct from source)

```bash
python -m forward_model COMMAND [OPTIONS]
```

This works directly from the source code without installation.

!!! tip
    Both methods support the same commands and options. Use whichever fits your workflow.

## Available Commands

### `run` - Run Forward Model Calculation

Calculate magnetic anomalies and optionally generate plots.

```bash
forward-model run INPUT_FILE [OPTIONS]
```

**Arguments:**

- `INPUT_FILE`: Path to model file (JSON or CSV format)

**Options:**

- `--output-csv PATH` - Save results to CSV file
- `--output-json PATH` - Save complete model and results to JSON
- `--output-npy PATH` - Save results to NumPy format (.npy or .npz)
- `--plot PATH` - Save plot to file (PNG, PDF, SVG, EPS)
- `--no-plot` - Skip plot generation (useful for batch processing)
- `--verbose, -v` - Enable verbose output with progress information
- `--help` - Show command help

**Examples:**

```bash
# Basic calculation with plot
forward-model run model.json --plot output.png

# Export to multiple formats
forward-model run model.json \
  --output-csv results.csv \
  --output-json results.json \
  --output-npy results.npy \
  --plot figure.pdf

# Batch processing (no plot)
forward-model run model.json --output-csv results.csv --no-plot

# Verbose output
forward-model run model.json --plot output.png --verbose
```

**Verbose Output Example:**

```
Loading model from model.json...
  Loaded 3 bodies
  25 observation points
Calculating magnetic anomaly...
  Processing body 1/3: Dyke
  Processing body 2/3: Sill
  Processing body 3/3: Basement
  Anomaly range: -125.45 to 456.78 nT
Generating plot...
  Plot saved to output.png
✓ Calculation complete
```

---

### `validate` - Validate Model Files

Check model files for errors without running calculations.

```bash
forward-model validate INPUT_FILE [OPTIONS]
```

**Arguments:**

- `INPUT_FILE`: Path to model file to validate

**Options:**

- `--verbose, -v` - Show detailed validation information
- `--help` - Show command help

**Examples:**

```bash
# Quick validation
forward-model validate model.json

# Detailed validation output
forward-model validate model.json --verbose
```

**Output:**

```
✓ Model is valid
  Bodies: 2
  Observation points: 50
  Field intensity: 50000.0 nT
  Field inclination: 60.0°
```

**Validation Checks:**

- JSON/CSV syntax errors
- Required fields present
- Data types correct
- Physical parameter ranges (e.g., susceptibility, field intensity)
- Polygon validity (at least 3 vertices, closed shapes)
- Coordinate system consistency

---

### `visualize` - Visualize Existing Results

Create plots from previously computed results.

```bash
forward-model visualize RESULTS_FILE [OPTIONS]
```

**Arguments:**

- `RESULTS_FILE`: Path to JSON results file (from `--output-json`)

**Options:**

- `--output, -o PATH` - Save plot to file
- `--style TEXT` - Plot style: `default`, `publication`, or `presentation`
- `--dpi INTEGER` - DPI for saved figure (default: 100)
- `--no-show` - Don't display plot interactively
- `--verbose, -v` - Enable verbose output
- `--help` - Show command help

**Examples:**

```bash
# Create default plot
forward-model visualize results.json --output plot.png

# Publication-quality figure
forward-model visualize results.json \
  --output figure.pdf \
  --style publication \
  --dpi 300

# Presentation style
forward-model visualize results.json \
  --output slide.png \
  --style presentation \
  --dpi 150
```

---

## Input Formats

### JSON Format (Recommended)

```json
{
  "bodies": [
    {
      "name": "Body Name",
      "susceptibility": 0.05,
      "vertices": [[x1, z1], [x2, z2], [x3, z3], [x4, z4]]
    }
  ],
  "field": {
    "intensity": 50000.0,
    "inclination": 60.0,
    "declination": 0.0
  },
  "observation_x": [x1, x2, x3, ...],
  "observation_z": 0.0
}
```

**Field Descriptions:**

- `bodies`: List of geological bodies
  - `name`: Descriptive name (string)
  - `susceptibility`: Magnetic susceptibility in SI units (float)
  - `vertices`: List of [x, z] coordinate pairs (meters)
- `field`: Earth's magnetic field parameters
  - `intensity`: Field strength in nanoTesla (nT)
  - `inclination`: Angle from horizontal (-90° to 90°)
  - `declination`: Azimuth angle (-180° to 180°)
- `observation_x`: X-coordinates for observation points (meters)
- `observation_z`: Z-coordinate for observation level (meters, usually 0)

### CSV Format

```csv
# Field parameters: intensity,inclination,declination,observation_z
50000.0,60.0,0.0,0.0
# Bodies: name,susceptibility,x1,z1,x2,z2,...
Dyke,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
# Observations: x-coordinates
-100,-50,0,50,100
```

!!! note
    CSV format is more compact but less readable than JSON. Use JSON for complex models.

---

## Output Formats

### CSV Output

Tabular format with columns:

```csv
x,anomaly
-100.0,-12.34
-50.0,45.67
0.0,123.45
```

### JSON Output

Complete model and results for reproducibility:

```json
{
  "model": { ... },
  "results": {
    "x": [-100, -50, 0, 50, 100],
    "anomaly": [-12.34, 45.67, 123.45, 89.01, 23.45]
  }
}
```

### NumPy Output

Binary format for numerical analysis:

- `.npy` - Single 2D array: `[[x1, a1], [x2, a2], ...]`
- `.npz` - Named arrays: `x` and `anomaly`

**Load in Python:**

```python
import numpy as np

# .npy format
data = np.load("results.npy")
x = data[:, 0]
anomaly = data[:, 1]

# .npz format
data = np.load("results.npz")
x = data["x"]
anomaly = data["anomaly"]
```

---

## Working with Examples

Forward Model includes example model files in the `examples/` directory:

```bash
# List examples
ls examples/

# Run an example
forward-model run examples/simple_dyke.json --plot output.png

# Validate an example
forward-model validate examples/multi_body.json --verbose
```

---

## Batch Processing

Process multiple models using shell scripts:

```bash
#!/bin/bash
# process_models.sh

for model in models/*.json; do
  basename=$(basename "$model" .json)
  echo "Processing $basename..."

  forward-model run "$model" \
    --output-csv "results/${basename}.csv" \
    --no-plot
done

echo "✓ Batch processing complete"
```

---

## Tips and Best Practices

### Performance

- Use `--no-plot` for faster batch processing
- CSV output is faster than JSON for large datasets
- NumPy format is most efficient for numerical analysis

### Debugging

- Use `--verbose` to see detailed progress
- Use `validate` before `run` to catch errors early
- Check anomaly ranges in verbose output for sanity checks

### Reproducibility

- Always use `--output-json` to save complete model parameters
- Version control your model files
- Document field parameters and coordinate system

### File Organization

```
project/
├── models/          # Input model files
├── results/         # Output data files
├── plots/           # Generated figures
└── scripts/         # Processing scripts
```

---

## See Also

- [Visualization Guide](visualization.md) - Customize plot styles and formats
- [Examples](examples.md) - Learn from example models
- [API Reference](../api-reference/index.md) - Python API for custom workflows
