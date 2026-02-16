# API Reference

This section provides complete API documentation for the Forward Model package, auto-generated from Python docstrings.

## Overview

Forward Model is organized into several modules:

- **`forward_model.models`** - Data models (Pydantic classes)
- **`forward_model.compute`** - Core calculation algorithms
- **`forward_model.io`** - Input/output handlers
- **`forward_model.viz`** - Visualization functions
- **`forward_model.cli`** - Command-line interface

## Quick Reference

### Core Functions

```python
from forward_model import (
    # I/O Functions
    load_model,              # Load model from JSON
    load_model_from_csv,     # Load model from CSV

    # Computation
    calculate_anomaly,       # Calculate magnetic anomaly

    # Visualization
    plot_model,              # Plot cross-section
    plot_anomaly,            # Plot anomaly profile
    plot_combined,           # Combined plot (recommended)

    # Data Models
    ForwardModel,            # Main model class
    GeologicBody,            # Body definition
    MagneticField,           # Field parameters
)
```

### I/O Functions

```python
from forward_model.io import (
    write_csv,               # Write results to CSV
    write_json,              # Write results to JSON
    write_numpy,             # Write results to NumPy format
)
```

## Data Models

All models use [Pydantic](https://docs.pydantic.dev/) for validation and serialization.

### ForwardModel

The main model class containing all parameters for a forward calculation:

```python
from forward_model.models import ForwardModel, GeologicBody, MagneticField

model = ForwardModel(
    bodies=[...],           # List of GeologicBody objects
    field=MagneticField(...),  # Magnetic field parameters
    observation_x=[...],    # X-coordinates for observations
    observation_z=0.0       # Z-coordinate for observation level
)
```

### GeologicBody

Represents a 2D polygonal geological body:

```python
body = GeologicBody(
    name="Body Name",           # Descriptive name
    susceptibility=0.05,        # Magnetic susceptibility (SI)
    vertices=[[x1, z1], ...]    # Polygon vertices (x, z in meters)
)
```

### MagneticField

Earth's magnetic field parameters:

```python
field = MagneticField(
    intensity=50000.0,      # Field strength (nT)
    inclination=60.0,       # Inclination angle (degrees)
    declination=0.0         # Declination angle (degrees)
)
```

## Usage Examples

### Basic Workflow

```python
from forward_model import load_model, calculate_anomaly, plot_combined

# Load model
model = load_model("model.json")

# Calculate anomaly
anomaly = calculate_anomaly(model)

# Visualize
plot_combined(model, anomaly, save_path="output.png")
```

### Creating Models Programmatically

```python
from forward_model.models import ForwardModel, GeologicBody, MagneticField
from forward_model import calculate_anomaly
import numpy as np

# Define body
body = GeologicBody(
    name="Dyke",
    susceptibility=0.08,
    vertices=[[0, 50], [10, 50], [10, 200], [0, 200]]
)

# Define field
field = MagneticField(
    intensity=50000.0,
    inclination=60.0,
    declination=0.0
)

# Create model
model = ForwardModel(
    bodies=[body],
    field=field,
    observation_x=list(range(-100, 101, 5)),
    observation_z=0.0
)

# Calculate
anomaly = calculate_anomaly(model)
```

### Exporting Results

```python
from forward_model.io import write_csv, write_json, write_numpy
import numpy as np

x = np.array(model.observation_x)

# Export to different formats
write_csv("results.csv", x, anomaly)
write_json("results.json", model, x, anomaly)
write_numpy("results.npy", x, anomaly)
```

## Module Documentation

Detailed API documentation for each module is auto-generated from the source code:

- **[Models](forward_model/models/index.md)** - Data models and validation
- **[Compute](forward_model/compute/index.md)** - Calculation algorithms
- **[I/O](forward_model/io/index.md)** - Input/output functions
- **[Visualization](forward_model/viz/index.md)** - Plotting functions
- **[CLI](forward_model/cli/index.md)** - Command-line interface

!!! note
    The module documentation pages are automatically generated from Python docstrings using `mkdocstrings`.

## Type Hints

Forward Model uses comprehensive type hints for better IDE support and type checking:

```python
from forward_model import load_model, calculate_anomaly
from forward_model.models import ForwardModel
import numpy as np
import numpy.typing as npt

# Type hints for clarity
model: ForwardModel = load_model("model.json")
anomaly: npt.NDArray[np.float64] = calculate_anomaly(model)
```

## Error Handling

All functions raise appropriate exceptions for invalid inputs:

```python
from forward_model import load_model
from pydantic import ValidationError

try:
    model = load_model("invalid_model.json")
except FileNotFoundError:
    print("Model file not found")
except ValidationError as e:
    print(f"Model validation failed: {e}")
except json.JSONDecodeError:
    print("Invalid JSON format")
```

## Performance Considerations

### Calculation Complexity

The Talwani algorithm has computational complexity:

- **Per body**: O(n × m) where n = observation points, m = vertices
- **Total**: O(n × Σm_i) for all bodies

### Optimization Tips

1. **Reduce observation points** for faster calculations
2. **Simplify polygons** while maintaining geometry
3. **Use NumPy output** for batch processing
4. **Vectorize** parameter studies using NumPy arrays

Example optimization:

```python
import numpy as np
from forward_model import calculate_anomaly

# Vectorized parameter study
susceptibilities = np.linspace(0.01, 0.15, 50)
anomalies = []

for susc in susceptibilities:
    model_copy = model.model_copy(deep=True)
    model_copy.bodies[0].susceptibility = susc
    anomalies.append(calculate_anomaly(model_copy))

anomalies = np.array(anomalies)  # Shape: (50, n_observations)
```

## See Also

- [Getting Started](../getting-started.md) - Basic usage tutorial
- [User Guide](../user-guide/cli.md) - Command-line interface
- [Examples](../user-guide/examples.md) - Example workflows
- [Development](../development/contributing.md) - Contributing to Forward Model
