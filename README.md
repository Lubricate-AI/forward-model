# forward-model

A Python package for 2D forward magnetic modeling using the Talwani algorithm.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://lubricate-ai.github.io/forward-model/)

Compute magnetic anomalies from geological cross-sections with arbitrary polygonal shapes. Ideal for exploration geophysics, geological modeling, and educational purposes.

## Features

- **🚀 Fast analytical solutions** - No numerical integration required
- **📐 Arbitrary polygon geometry** - Model complex geological structures
- **🎨 Publication-quality visualizations** - Multiple plot styles and formats
- **💻 CLI and Python API** - Use from command line or integrate into workflows
- **📊 Multiple I/O formats** - JSON, CSV input; CSV, JSON, NumPy output
- **📚 Comprehensive documentation** - Theory, tutorials, and examples included
- **✅ Fully tested** - >90% code coverage with extensive test suite

## Quick Start

### Installation

```bash
pip install forward-model
```

Or install from source:

```bash
git clone https://github.com/yourusername/forward-model.git
cd forward-model
make install
```

### Command-Line Interface

The CLI can be invoked in two ways:

**Option 1: Installed console script** (after `pip install`)
```bash
forward-model validate examples/simple_dyke.json
forward-model run examples/simple_dyke.json --plot output.png
```

**Option 2: Python module** (works directly from source)
```bash
python -m forward_model validate examples/simple_dyke.json
python -m forward_model run examples/simple_dyke.json --plot output.png
```

Both approaches support all commands and options:

```bash
# Export results in multiple formats
forward-model run examples/multi_body.json \
  --output-csv results.csv \
  --output-json results.json \
  --output-npy results.npy \
  --plot figure.pdf

# Or equivalently:
python -m forward_model run examples/multi_body.json \
  --output-csv results.csv \
  --output-json results.json \
  --output-npy results.npy \
  --plot figure.pdf
```

### Python API

```python
from forward_model import load_model, calculate_anomaly, plot_combined

# Load model from JSON
model = load_model("examples/simple_dyke.json")

# Calculate magnetic anomaly
anomaly = calculate_anomaly(model)

# Visualize results
fig = plot_combined(model, anomaly, save_path="output.png")

# Or create publication-quality figures
fig = plot_combined(
    model, anomaly,
    save_path="publication.pdf",
    style="publication",
    dpi=300
)
```

## Example: Simple Dyke

Model a vertical mafic dyke:

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

Run the calculation:

```bash
# Using console script
forward-model run examples/simple_dyke.json --plot dyke_anomaly.png --verbose

# Or using Python module (no install needed)
python -m forward_model run examples/simple_dyke.json --plot dyke_anomaly.png --verbose
```

Output:
```
Loading model from examples/simple_dyke.json...
  Loaded 1 bodies
  11 observation points
Calculating magnetic anomaly...
  Anomaly range: -45.23 to 287.56 nT
Generating plot...
  Plot saved to dyke_anomaly.png
✓ Calculation complete
```

## Documentation

📚 **[View Complete Documentation](https://lubricate-ai.github.io/forward-model/)**

The full documentation is available online, including:
- Getting started guide and tutorials
- Complete CLI reference
- API documentation with examples
- Development and contribution guidelines

### 📖 Theory and Background

See [`docs/theory.md`](docs/theory.md) for:
- Mathematical formulation of the Talwani algorithm
- Physics of magnetic anomalies
- Coordinate system conventions
- Assumptions and limitations

### 🎓 Tutorials and Examples

See [`docs/examples.md`](docs/examples.md) for:
- Step-by-step tutorials
- Creating custom models
- Batch processing
- Sensitivity analysis
- Integration with other tools

### 📦 Example Models

The `examples/` directory contains:
- `simple_dyke.json` - Basic vertical dyke model
- `multi_body.json` - Multiple interacting bodies
- `README.md` - Guide to creating your own models

## Input Formats

### JSON (Recommended)

```json
{
  "bodies": [
    {
      "name": "Body name",
      "susceptibility": 0.05,
      "vertices": [[x1, z1], [x2, z2], ...]
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

### CSV

```csv
# Field parameters: intensity,inclination,declination,observation_z
50000.0,60.0,0.0,0.0
# Bodies: name,susceptibility,x1,z1,x2,z2,...
Dyke,0.05,0.0,100.0,50.0,100.0,50.0,200.0,0.0,200.0
# Observations: x-coordinates
-100,-50,0,50,100
```

## Output Formats

- **CSV** - Tabular format with x-coordinates and anomaly values
- **JSON** - Complete model and results for reproducibility
- **NumPy** - `.npy` (2D array) or `.npz` (named arrays) for numerical analysis
- **Plots** - PNG, PDF, SVG, EPS with multiple style options

## Visualization Styles

Choose from multiple plot styles:

```python
# Default style
plot_combined(model, anomaly, style="default")

# Publication style (serif fonts, high DPI)
plot_combined(model, anomaly, style="publication", dpi=300)

# Presentation style (large fonts)
plot_combined(model, anomaly, style="presentation")
```

Additional visualization options:
- Color bodies by index or susceptibility
- Show/hide observation lines
- Custom figure size and DPI
- Vector and raster output formats

## Development

### Requirements

- Python 3.12+
- See `pyproject.toml` for dependencies

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/forward-model.git
cd forward-model

# Install dependencies
make install

# Run tests
make test

# Run linting
make lint

# Type checking
make type-checking
```

### Running Tests

```bash
# All tests
make test

# Specific test file
uv run pytest tests/unit/io/test_loaders.py -v

# With coverage
uv run pytest --cov=forward_model --cov-report=html
```

## Project Structure

```
forward-model/
├── forward_model/          # Main package
│   ├── cli/               # Command-line interface
│   ├── compute/           # Talwani algorithm implementation
│   ├── io/                # Input/output handlers
│   ├── models/            # Data models (Pydantic)
│   └── viz/               # Visualization functions
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── examples/              # Example model files
├── docs/                  # Documentation
├── Makefile              # Development commands
└── pyproject.toml        # Project configuration
```

## CLI Commands

> **Note:** All commands shown below use the `forward-model` console script syntax. You can also use `python -m forward_model` instead if running directly from source without installation.

### `forward-model run`

Run forward model calculation.

```bash
forward-model run INPUT_FILE [OPTIONS]
# Or: python -m forward_model run INPUT_FILE [OPTIONS]

Options:
  --output-csv PATH      Save results to CSV
  --output-json PATH     Save results to JSON
  --output-npy PATH      Save results to NumPy format
  --plot PATH            Save plot to file
  --no-plot              Skip plot generation
  --verbose, -v          Enable verbose output
```

### `forward-model validate`

Validate a model file without running calculations.

```bash
forward-model validate INPUT_FILE [OPTIONS]

Options:
  --verbose, -v          Show detailed validation info
```

### `forward-model visualize`

Visualize results from a previously computed model.

```bash
forward-model visualize RESULTS_FILE [OPTIONS]

Options:
  --output, -o PATH      Save plot to file
  --style TEXT           Plot style (default, publication, presentation)
  --dpi INTEGER          DPI for saved figure
  --no-show              Don't display plot
  --verbose, -v          Enable verbose output
```

## API Reference

### Core Functions

```python
from forward_model import (
    # I/O
    load_model,              # Load model from JSON
    load_model_from_csv,     # Load model from CSV
    write_csv,               # Write results to CSV
    write_json,              # Write results to JSON
    write_numpy,             # Write results to NumPy format

    # Computation
    calculate_anomaly,       # Calculate magnetic anomaly

    # Visualization
    plot_model,              # Plot cross-section
    plot_anomaly,            # Plot anomaly profile
    plot_combined,           # Combined cross-section and anomaly

    # Models
    ForwardModel,            # Main model class
    GeologicBody,            # Body definition
    MagneticField,           # Field parameters
)
```

### Data Models

All models use Pydantic for validation:

```python
from forward_model.models import ForwardModel, GeologicBody, MagneticField

# Create model programmatically
body = GeologicBody(
    name="My Body",
    susceptibility=0.05,
    vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]]
)

field = MagneticField(
    intensity=50000.0,
    inclination=60.0,
    declination=0.0
)

model = ForwardModel(
    bodies=[body],
    field=field,
    observation_x=[-100, -50, 0, 50, 100],
    observation_z=0.0
)
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `make lint` and `make test`
5. Submit a pull request

See `CLAUDE.md` for development conventions.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{forward_model,
  title = {forward-model: 2D Forward Magnetic Modeling},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/forward-model}
}
```

## References

**Talwani Algorithm**:
- Talwani, M., & Heirtzler, J. R. (1964). Computation of magnetic anomalies caused by two-dimensional structures of arbitrary shape. *Computers in the Mineral Industries*, 464-480.

**Potential Field Methods**:
- Blakely, R. J. (1995). *Potential Theory in Gravity and Magnetic Applications.* Cambridge University Press.

See `docs/theory.md` for complete references.

## License

MIT License - see LICENSE file for details.

## Support

- 📖 Documentation: See `docs/` directory
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/forward-model/issues)
- 📧 Contact: your.email@example.com

## Acknowledgments

This implementation builds on the seminal work of Manik Talwani and colleagues in developing efficient algorithms for potential field calculations.
