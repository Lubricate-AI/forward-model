# Forward Model

A Python package for 2D forward magnetic modeling using the Talwani algorithm.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Lubricate-AI/forward-model/blob/main/LICENSE)

Compute magnetic anomalies from geological cross-sections with arbitrary polygonal shapes. Ideal for exploration geophysics, geological modeling, and educational purposes.

## Features

- **🚀 Fast analytical solutions** - No numerical integration required
- **📐 Arbitrary polygon geometry** - Model complex geological structures
- **🎨 Publication-quality visualizations** - Multiple plot styles and formats
- **💻 CLI and Python API** - Use from command line or integrate into workflows
- **📊 Multiple I/O formats** - JSON, CSV input; CSV, JSON, NumPy output
- **📚 Comprehensive documentation** - Theory, tutorials, and examples included
- **✅ Fully tested** - >90% code coverage with extensive test suite

## Quick Example

Model a vertical mafic dyke:

```python
from forward_model import load_model, calculate_anomaly, plot_combined

# Load model from JSON
model = load_model("examples/simple_dyke.json")

# Calculate magnetic anomaly
anomaly = calculate_anomaly(model)

# Visualize results
fig = plot_combined(model, anomaly, save_path="output.png")
```

Or use the CLI:

```bash
forward-model run examples/simple_dyke.json --plot output.png
```

![Simple Dyke Example](images/simple_dyke.png)

*Figure: Calculated magnetic anomaly from a vertical dyke. Top panel shows the cross-section geometry, bottom panel shows the resulting magnetic anomaly profile.*

## Getting Started

New to Forward Model? Start with the [Getting Started guide](getting-started.md) for installation instructions and a basic tutorial.

## Documentation Sections

- **[Getting Started](getting-started.md)** - Installation and basic usage
- **[User Guide](user-guide/cli.md)** - Detailed guides for CLI usage, visualization, and examples
- **[API Reference](api-reference/index.md)** - Complete Python API documentation
- **[Development](development/contributing.md)** - Contributing guidelines and architecture

## Theory and Background

Forward Model implements the Talwani algorithm for computing magnetic anomalies from 2D polygonal bodies. The method uses analytical solutions to calculate the magnetic field produced by bodies with arbitrary cross-sectional shapes.

Key references:

- **Talwani, M., & Heirtzler, J. R. (1964).** Computation of magnetic anomalies caused by two-dimensional structures of arbitrary shape. *Computers in the Mineral Industries*, 464-480.
- **Blakely, R. J. (1995).** *Potential Theory in Gravity and Magnetic Applications.* Cambridge University Press.

## License

MIT License - see [LICENSE](https://github.com/Lubricate-AI/forward-model/blob/main/LICENSE) for details.

## Support

- 📖 **Documentation**: You're reading it!
- 💬 **Issues**: [GitHub Issues](https://github.com/Lubricate-AI/forward-model/issues)
- 🔗 **Repository**: [github.com/Lubricate-AI/forward-model](https://github.com/Lubricate-AI/forward-model)
