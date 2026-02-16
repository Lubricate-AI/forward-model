# Usage Examples and Tutorials

This guide provides practical examples for using forward-model, from basic usage to advanced techniques.

## Installation

Install using pip:

```bash
pip install forward-model
```

Or for development:

```bash
git clone https://github.com/Lubricate-AI/forward-model.git
cd forward-model
make install
```

## Quick Start

### Command-Line Interface

The CLI can be invoked in two ways:

**After installation** (cleaner syntax):
```bash
forward-model validate examples/simple_dyke.json
forward-model run examples/simple_dyke.json --plot output.png
```

**From source** (no installation needed):
```bash
python -m forward_model validate examples/simple_dyke.json
python -m forward_model run examples/simple_dyke.json --plot output.png
```

Both approaches are equivalent and support all features:

```bash
# Generate publication-quality figures
forward-model run examples/simple_dyke.json \
  --output-csv results.csv \
  --output-json results.json \
  --plot publication.pdf

# Same command using Python module
python -m forward_model run examples/simple_dyke.json \
  --output-csv results.csv \
  --output-json results.json \
  --plot publication.pdf
```

> **Tip:** Throughout this guide, commands are shown using `forward-model`. You can substitute `python -m forward_model` in any example.

### Python API

For programmatic access:

```python
from forward_model import load_model, calculate_anomaly, plot_combined

# Load model from JSON
model = load_model("examples/simple_dyke.json")

# Calculate anomaly
anomaly = calculate_anomaly(model)

# Visualize results
fig = plot_combined(model, anomaly, save_path="output.png")
```

## Tutorial 1: Simple Dyke Model

### Objective

Model a vertical mafic dyke and understand the characteristic anomaly pattern.

### Step 1: Examine the Model

Let's look at `examples/simple_dyke.json`:

```json
{
  "bodies": [
    {
      "name": "Vertical Dyke",
      "susceptibility": 0.08,
      "vertices": [
        [20.0, 50.0],
        [30.0, 50.0],
        [30.0, 200.0],
        [20.0, 200.0]
      ]
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

**Model description**:
- Single rectangular body (10m × 150m cross-section)
- Centered at x=25m, extending from 50-200m depth
- Susceptibility: 0.08 SI (typical basaltic dyke)
- Magnetic field: Mid-latitude Northern hemisphere
- 11 observation points spanning -100m to 150m

### Step 2: Validate the Model

```bash
# Using console script
forward-model validate examples/simple_dyke.json --verbose

# Or from source
python -m forward_model validate examples/simple_dyke.json --verbose
```

Expected output:
```
✓ Model is valid
  1 bodies defined
  11 observation points
  Field intensity: 50000.0 nT
  Field inclination: 60.0°
  Field declination: 0.0°
```

### Step 3: Calculate and Visualize

#### Using CLI:

```bash
# With console script
forward-model run examples/simple_dyke.json \
  --plot simple_dyke.png \
  --output-csv results.csv \
  --verbose

# Or with Python module
python -m forward_model run examples/simple_dyke.json \
  --plot simple_dyke.png \
  --output-csv results.csv \
  --verbose
```

#### Using Python:

```python
from forward_model import load_model, calculate_anomaly, plot_combined
import matplotlib.pyplot as plt

# Load and compute
model = load_model("examples/simple_dyke.json")
anomaly = calculate_anomaly(model)

# Create plot
fig = plot_combined(model, anomaly)
plt.show()

# Print results
for x, anom in zip(model.observation_x, anomaly):
    print(f"x = {x:6.0f} m: anomaly = {anom:7.2f} nT")
```

### Step 4: Interpret Results

**Expected observations**:

1. **Peak anomaly**: Positive peak directly above the dyke center (~25m)
2. **Amplitude**: Several hundred nT (depends on depth and susceptibility)
3. **Shape**: Asymmetric due to 60° inclination
4. **Width**: Broader than dyke width due to depth of burial

**Physical interpretation**:
- The dyke is more magnetic than surrounding rocks (positive susceptibility)
- When magnetized by Earth's field, it creates a dipolar anomaly
- At 60° inclination, the positive pole is shifted slightly north of the body
- The anomaly decreases with distance following dipole decay

### Step 5: Experiments

Try modifying the model to understand effects:

**A. Change susceptibility** (`susceptibility: 0.04` instead of `0.08`):
```bash
# Edit the JSON file, then:
forward-model run modified_model.json --plot
# Or: python -m forward_model run modified_model.json --plot
```
**Result**: Anomaly amplitude halves (linear relationship)

**B. Change depth** (move body closer to surface, `z: 20.0` instead of `50.0`):
**Result**: Larger, sharper anomaly (inverse cube law)

**C. Change inclination** (`inclination: 30.0` instead of `60.0`):
**Result**: More asymmetric anomaly pattern

**D. Add observation points** (denser spacing):
```json
"observation_x": [-100, -90, -80, ..., 140, 150]
```
**Result**: Smoother anomaly profile

## Tutorial 2: Multi-Body Model

### Objective

Model multiple interacting bodies and understand the superposition principle.

### Step 1: Load and Examine

```python
from forward_model import load_model, calculate_anomaly
import numpy as np

model = load_model("examples/multi_body.json")

print(f"Number of bodies: {len(model.bodies)}")
for body in model.bodies:
    print(f"  {body.name}: χ = {body.susceptibility}")
```

### Step 2: Calculate Individual and Combined Anomalies

```python
from forward_model.models import ForwardModel

# Calculate total anomaly
anomaly_total = calculate_anomaly(model)

# Calculate individual body contributions
anomalies_individual = []
for body in model.bodies:
    # Create model with single body
    single_body_model = ForwardModel(
        bodies=[body],
        field=model.field,
        observation_x=model.observation_x,
        observation_z=model.observation_z
    )
    anom = calculate_anomaly(single_body_model)
    anomalies_individual.append(anom)

# Verify superposition: sum of individuals should equal total
sum_individual = sum(anomalies_individual)
print(f"Max difference: {np.max(np.abs(anomaly_total - sum_individual)):.6f} nT")
# Should be ~0 (numerical precision)
```

### Step 3: Visualize Components

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

# Plot individual contributions
for i, (body, anom) in enumerate(zip(model.bodies, anomalies_individual)):
    ax.plot(model.observation_x, anom, '--',
            label=f"{body.name} (χ={body.susceptibility})", alpha=0.7)

# Plot total
ax.plot(model.observation_x, anomaly_total, 'k-', linewidth=2, label='Total')

ax.set_xlabel('X (m)')
ax.set_ylabel('Anomaly (nT)')
ax.set_title('Superposition of Magnetic Anomalies')
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('superposition.png', dpi=150)
```

### Step 4: Interpretation

**Key observations**:

1. **Dyke** (χ=0.05): Moderate positive anomaly
2. **Intrusion** (χ=0.08): Stronger positive anomaly (higher susceptibility)
3. **Void** (χ=-0.01): Small negative anomaly (diamagnetic effect)
4. **Total**: Complex pattern showing all three contributions

**Insights**:
- Bodies with positive susceptibility create positive anomalies
- Negative susceptibility (voids, some minerals) creates negative anomalies
- Total anomaly is linear sum of individual anomalies
- Anomalies can constructively or destructively interfere

## Tutorial 3: Custom Model Creation

### Creating Models from Scratch

#### JSON Format (Recommended)

```python
import json

model_dict = {
    "bodies": [
        {
            "name": "Sill",
            "susceptibility": 0.06,
            "vertices": [
                [0.0, 100.0],    # Top-left
                [100.0, 100.0],  # Top-right
                [100.0, 120.0],  # Bottom-right
                [0.0, 120.0]     # Bottom-left
            ]
        }
    ],
    "field": {
        "intensity": 48000.0,
        "inclination": 65.0,
        "declination": -5.0
    },
    "observation_x": list(range(-50, 151, 5)),
    "observation_z": 0.0
}

# Save to file
with open("my_model.json", "w") as f:
    json.dump(model_dict, f, indent=2)

# Use it
from forward_model import load_model, calculate_anomaly

model = load_model("my_model.json")
anomaly = calculate_anomaly(model)
```

#### CSV Format

```python
import csv

# Create CSV model
csv_content = [
    # Field parameters
    ["48000.0", "65.0", "-5.0", "0.0"],
    # Bodies (name, susceptibility, vertex x-z pairs)
    ["Sill", "0.06", "0.0", "100.0", "100.0", "100.0", "100.0", "120.0", "0.0", "120.0"],
    # Observation points
    [str(x) for x in range(-50, 151, 5)]
]

with open("my_model.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_content)

# Use it
from forward_model import load_model_from_csv, calculate_anomaly

model = load_model_from_csv("my_model.csv")
anomaly = calculate_anomaly(model)
```

### Programmatic Creation

```python
from forward_model.models import ForwardModel, GeologicBody, MagneticField
import numpy as np

# Define a body
body = GeologicBody(
    name="Pluton",
    susceptibility=0.1,
    vertices=[
        [0.0, 200.0],
        [50.0, 150.0],
        [100.0, 200.0],
        [100.0, 400.0],
        [0.0, 400.0]
    ]
)

# Define magnetic field
field = MagneticField(
    intensity=52000.0,
    inclination=70.0,
    declination=0.0
)

# Create model
model = ForwardModel(
    bodies=[body],
    field=field,
    observation_x=list(np.linspace(-100, 200, 61)),
    observation_z=0.0
)

# Calculate
from forward_model import calculate_anomaly
anomaly = calculate_anomaly(model)
```

## Tutorial 4: Output Formats and Visualization

### Multiple Output Formats

```bash
# Save in multiple formats at once (console script)
forward-model run examples/simple_dyke.json \
  --output-csv results.csv \
  --output-json results.json \
  --output-npy results.npy \
  --plot figure.png \
  --verbose

# Or (Python module)
python -m forward_model run examples/simple_dyke.json \
  --output-csv results.csv \
  --output-json results.json \
  --output-npy results.npy \
  --plot figure.png \
  --verbose
```

### Working with NumPy Output

```python
import numpy as np

# .npy format (2D array)
data = np.load("results.npy")
x_coords = data[:, 0]
anomaly_values = data[:, 1]

# .npz format (named arrays)
data = np.load("results.npz")
x_coords = data['x']
anomaly_values = data['anomaly']

# Use in further analysis
import matplotlib.pyplot as plt
plt.plot(x_coords, anomaly_values)
plt.xlabel('Position (m)')
plt.ylabel('Anomaly (nT)')
plt.show()
```

### Publication-Quality Figures

```python
from forward_model import load_model, calculate_anomaly, plot_combined

model = load_model("examples/multi_body.json")
anomaly = calculate_anomaly(model)

# Default style
fig = plot_combined(model, anomaly, save_path="default.png")

# Publication style (serif fonts, high DPI)
fig = plot_combined(
    model, anomaly,
    save_path="publication.pdf",
    style="publication",
    dpi=300,
    figsize=(10, 7)
)

# Presentation style (large fonts)
fig = plot_combined(
    model, anomaly,
    save_path="presentation.png",
    style="presentation",
    dpi=200
)
```

### Customizing Visualizations

```python
from forward_model import load_model, calculate_anomaly
from forward_model.viz import plot_model, plot_anomaly
import matplotlib.pyplot as plt

model = load_model("examples/simple_dyke.json")
anomaly = calculate_anomaly(model)

# Create custom layout
fig = plt.figure(figsize=(14, 10))

# Cross-section with susceptibility coloring
ax1 = plt.subplot(3, 1, 1)
plot_model(model, ax=ax1, color_by="susceptibility", show_observation_lines=True)

# Cross-section with index coloring
ax2 = plt.subplot(3, 1, 2)
plot_model(model, ax=ax2, color_by="index", show_observation_lines=False)

# Anomaly profile
ax3 = plt.subplot(3, 1, 3)
plot_anomaly(model.observation_x, anomaly, ax=ax3)

plt.tight_layout()
plt.savefig("custom_layout.png", dpi=200)
```

## Advanced Topics

### Batch Processing

Process multiple models:

```python
from pathlib import Path
from forward_model import load_model, calculate_anomaly, write_csv

model_dir = Path("models")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

for model_file in model_dir.glob("*.json"):
    # Load and calculate
    model = load_model(model_file)
    anomaly = calculate_anomaly(model)

    # Save results
    output_file = output_dir / f"{model_file.stem}_anomaly.csv"
    write_csv(output_file, model.observation_x, anomaly)

    print(f"Processed {model_file.name}")
```

### Sensitivity Analysis

Explore how anomaly changes with parameters:

```python
import numpy as np
import matplotlib.pyplot as plt
from forward_model import load_model, calculate_anomaly

# Load base model
model = load_model("examples/simple_dyke.json")

# Vary susceptibility
susceptibilities = np.linspace(0.02, 0.15, 8)
fig, ax = plt.subplots(figsize=(10, 6))

for chi in susceptibilities:
    # Modify susceptibility
    model.bodies[0].susceptibility = chi

    # Calculate
    anomaly = calculate_anomaly(model)

    # Plot
    ax.plot(model.observation_x, anomaly, label=f"χ = {chi:.3f}")

ax.set_xlabel('X (m)')
ax.set_ylabel('Anomaly (nT)')
ax.set_title('Sensitivity to Susceptibility')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('sensitivity_susceptibility.png', dpi=150)
```

### Integration with Other Tools

Export for use in other software:

```python
import pandas as pd
from forward_model import load_model, calculate_anomaly

model = load_model("examples/simple_dyke.json")
anomaly = calculate_anomaly(model)

# Create pandas DataFrame
df = pd.DataFrame({
    'x_m': model.observation_x,
    'anomaly_nT': anomaly,
    'body_name': model.bodies[0].name,
    'susceptibility': model.bodies[0].susceptibility
})

# Save to various formats
df.to_csv('results.csv', index=False)
df.to_excel('results.xlsx', index=False)
df.to_json('results_dataframe.json', orient='records')

print(df.describe())
```

## Troubleshooting

### Common Issues

**1. "Model validation failed"**
- Check JSON syntax (use a validator)
- Ensure all required fields present
- Verify vertices form closed polygon (minimum 3 points)

**2. "Invalid JSON"**
- Check for trailing commas
- Ensure proper quote usage
- Validate with `forward-model validate`

**3. Unexpected anomaly values**
- Verify susceptibility units (SI, not CGS)
- Check field intensity (should be in nT, not T)
- Confirm coordinate system (z positive downward)

**4. Plot doesn't display**
- Use `plt.show()` in Python scripts
- Use `--plot output.png` to save instead
- Check if running in non-interactive environment

### Getting Help

1. Check documentation: `docs/theory.md` for concepts
2. Validate models: `forward-model validate your_model.json`
3. Use verbose mode: `--verbose` flag for detailed output
4. Review examples: `examples/` directory
5. Check package help: `forward-model --help`

## Next Steps

- Read `theory.md` for mathematical background
- Explore `examples/` directory for more models
- Try creating your own geological scenarios
- Experiment with different field parameters
- Combine with inversion software for interpretation

## References

For theoretical background and mathematical details, see `docs/theory.md`.

For model file format specifications, see `examples/README.md`.
