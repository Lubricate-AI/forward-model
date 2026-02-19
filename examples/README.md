# Example Models

This directory contains example forward model definitions for learning and testing.

## Quick Start

Run an example using the CLI (two equivalent ways):

```bash
# After pip install:
forward-model run examples/simple_dyke.json --plot simple_dyke.png

# Or directly from source (no install needed):
python -m forward_model run examples/simple_dyke.json --plot simple_dyke.png
```

Or use the Python API:

```python
from forward_model import load_model, calculate_anomaly, plot_combined

model = load_model("examples/simple_dyke.json")
anomaly = calculate_anomaly(model)
plot_combined(model, anomaly, save_path="output.png")
```

## Example Files

### simple_dyke.json

A basic vertical dyke model demonstrating fundamental concepts:

- **Single body**: Vertical dyke with rectangular cross-section
- **Susceptibility**: 0.08 SI (typical for mafic intrusive rocks)
- **Dimensions**: 10m wide, extending from 50m to 200m depth
- **Position**: Centered at x=25m
- **Observation points**: 11 points spanning -100m to 150m

**Expected result**: A characteristic positive magnetic anomaly peak directly above the dyke, with the anomaly decreasing away from the dyke following the dipolar field pattern.

**Learning objectives**:
- Understanding basic model structure (JSON format)
- Observing the relationship between body geometry and anomaly shape
- Introduction to susceptibility values

### volcanic_body.json

A high-Q volcanic flow demonstrating remanent magnetization:

- **Single body**: Wide horizontal lava sheet (1000m × 150m), 50–200m depth
- **Susceptibility**: 0.01 SI (low induced component)
- **Remanent intensity**: 2.0 A/m — strong remanence from past eruption
- **Remanent inclination / declination**: −45° / 10° (reversed polarity, offset from present field)
- **Königsberger ratio Q ≈ 5**: Remanent magnetization dominates over induced

**Expected result**: A negative anomaly driven by the reversed remanent vector, opposite in sign to what the small induced component alone would produce.

**Learning objectives**:
- Understanding remanent vs. induced magnetization
- Königsberger ratio and when remanence matters
- Effect of reversed polarity on anomaly character

### high_susceptibility_intrusion.json

A magnetite-rich iron-ore intrusion demonstrating demagnetization correction:

- **Single body**: 500m wide × 400m tall intrusion, 100–500m depth
- **Susceptibility**: 1.0 SI (magnetite-rich — demagnetization is significant)
- **Demagnetization factor**: N_d = 0.3, giving χ_eff ≈ 0.769 (a 23% reduction)

**Expected result**: A broad positive anomaly with amplitude ~23% lower than an uncorrected model (N_d = 0) would predict. Omitting the correction at this susceptibility would systematically overestimate the anomaly.

**Learning objectives**:
- Understanding when demagnetization correction matters (χ > ~0.1 SI)
- Applying the `demagnetization_factor` field
- Estimating N_d from body geometry with `compute_demagnetization_factor()`

### multi_body.json

A more complex model with multiple interacting bodies:

- **Three bodies**:
  1. **Dyke**: χ=0.05 SI, vertical body at x=0-20m
  2. **Intrusion**: χ=0.08 SI, rectangular body at x=60-120m
  3. **Void**: χ=-0.01 SI, triangular cavity at x=150-170m

- **Observation points**: 31 points for a smooth anomaly profile
- **Field**: Same Earth field parameters as simple_dyke

**Expected result**: Complex anomaly pattern showing:
- Positive anomaly from the dyke
- Stronger positive anomaly from the intrusion (higher susceptibility)
- Small negative anomaly from the void (negative susceptibility)
- Superposition effects where anomalies interact

**Learning objectives**:
- Understanding the superposition principle
- Effect of varying susceptibility values
- Complex anomaly interpretation
- Different body geometries (rectangle, triangle)

## Creating Your Own Models

### JSON Format

Models are defined in JSON with four required sections:

```json
{
  "bodies": [
    {
      "name": "Body name",
      "susceptibility": 0.05,
      "vertices": [[x1, z1], [x2, z2], [x3, z3], ...]
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

### CSV Format

Alternatively, use CSV format:

```csv
# Field parameters: intensity,inclination,declination,observation_z
50000.0,60.0,0.0,0.0
# Bodies: name,susceptibility,x1,z1,x2,z2,...
Dyke,0.05,0.0,100.0,20.0,100.0,20.0,300.0,0.0,300.0
# Observations: x-coordinates
-50,-40,-30,-20,-10,0,10,20,30,40,50
```

### Tips for Model Design

1. **Coordinate system**:
   - X-axis: Horizontal position (meters)
   - Z-axis: Depth (meters, positive downward)
   - Observation points are at depth `observation_z` (usually 0)

2. **Susceptibility values** (SI units):
   - Sedimentary rocks: 0.001 - 0.01
   - Mafic intrusive rocks: 0.05 - 0.15
   - Ultramafic rocks: 0.1 - 0.3
   - Magnetite-rich rocks: 0.5+
   - Voids/cavities: negative values

3. **Body geometry**:
   - Define vertices in order (clockwise or counterclockwise)
   - Bodies will be automatically closed (first vertex connects to last)
   - Minimum 3 vertices (triangle)

4. **Observation points**:
   - More points = smoother anomaly profile
   - Extend beyond body edges to capture full anomaly
   - Typical spacing: 5-20% of body width

5. **Field parameters**:
   - **Intensity**: Total field strength in nT (Earth's field ~25,000-65,000 nT)
   - **Inclination**: Angle from horizontal (positive downward)
     - Equator: ~0°
     - Mid-latitudes: ~60°
     - Poles: ~90°
   - **Declination**: Angle from north (usually 0° for 2D profiles)

## Validation

Validate your model before running calculations:

```bash
# Using console script
forward-model validate my_model.json

# Or using Python module
python -m forward_model validate my_model.json
```

This checks:
- JSON syntax
- Required fields present
- Valid numeric ranges
- Body geometry (minimum 3 vertices)
- Observation points defined

## Further Reading

See `docs/examples.md` for detailed tutorials and `docs/theory.md` for mathematical background.
