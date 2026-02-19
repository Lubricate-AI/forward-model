# Theoretical Background

## Introduction

Forward magnetic modeling calculates the magnetic anomaly that would be observed above a geological structure with known geometry and magnetic properties. This is the "forward problem" - predicting observations from a model - as opposed to the "inverse problem" of inferring subsurface structure from observations.

This package implements the **Talwani algorithm** for 2D forward magnetic modeling, originally developed for gravity modeling (Talwani et al., 1959) and adapted for magnetics.

## The Talwani Method

### Historical Context

In 1965, Manik Talwani and colleagues published a seminal paper describing an efficient algorithm for computing the gravity anomaly of a 2D body with arbitrary cross-sectional shape. The method approximates the body as a polygon and uses analytical solutions for the gravitational effect of each edge. This approach was revolutionary because:

1. **Exact solution**: No numerical integration required
2. **Arbitrary geometry**: Any polygon shape can be modeled
3. **Computational efficiency**: Fast enough for interactive modeling

The same mathematical framework applies to magnetic modeling, though with additional complexity due to the vector nature of magnetic fields.

### Key Insight

The Talwani method recognizes that the field from a 2D body (infinite extent perpendicular to the profile) can be computed by summing the contributions from each edge of the polygonal cross-section. Each edge contributes analytically calculable terms involving logarithmic and arctangent functions.

## Physics of Magnetic Anomalies

### Magnetic Susceptibility

Magnetic susceptibility (χ) is a dimensionless quantity that describes how much a material becomes magnetized in an external magnetic field:

```
M = χ * H
```

where:
- **M**: Induced magnetization (A/m)
- **χ**: Magnetic susceptibility (SI units, dimensionless)
- **H**: Magnetic field strength (A/m)

In the SI system, susceptibility values for rocks typically range from:
- Sedimentary rocks: 0.001 - 0.01
- Mafic igneous rocks: 0.05 - 0.15
- Ultramafic rocks: 0.1 - 0.3
- Magnetite-rich rocks: 0.5+

### Induced Magnetization

This implementation models **induced magnetization** - the magnetization acquired by rocks in the present-day Earth magnetic field - as well as **remanent magnetization** (permanent magnetization acquired when rocks formed). See [Remanent Magnetization](#remanent-magnetization) below.

The induced magnetization vector is:

```
M = χ * (B₀ / μ₀)
```

where:
- **B₀**: Earth's magnetic field intensity (nT)
- **μ₀**: Permeability of free space (4π × 10⁻⁷ H/m)

The direction of M is parallel to the inducing field B₀.

### Remanent Magnetization

Remanent magnetization is a permanent magnetization vector fixed at the time of rock formation, independent of the present-day ambient field. Many volcanic and intrusive rocks carry significant remanence that can dominate the total magnetization.

The total magnetization is the vector sum of induced and remanent components:

```
M_total = M_induced + M_remanent
```

The **Königsberger ratio** Q quantifies the relative importance of remanence:

```
Q = |M_remanent| / |M_induced|
```

When Q > 1, remanent magnetization dominates and omitting it leads to significant modeling errors. Volcanic rocks (especially basalts and lava flows) commonly have Q values of 1–10 or higher.

The remanent vector is described by three parameters in the JSON schema:

| Field | Description | Units | Default |
|-------|-------------|-------|---------|
| `remanent_intensity` | Magnitude of remanent magnetization | A/m | 0.0 |
| `remanent_inclination` | Inclination of remanent vector (positive downward) | degrees | 0.0 |
| `remanent_declination` | Declination of remanent vector from north | degrees | 0.0 |

Setting `remanent_intensity=0.0` (the default) recovers the induced-only behaviour.

### Earth's Magnetic Field

The Earth's magnetic field is described by three parameters:

1. **Intensity (F)**: Total field strength in nanoTesla (nT)
   - Varies from ~25,000 nT at equator to ~65,000 nT at poles
   - Default value: 50,000 nT (typical mid-latitude)

2. **Inclination (I)**: Angle from horizontal plane
   - Positive downward
   - Varies with latitude: ~0° at equator, ~90° at poles
   - Default value: 60° (Northern mid-latitude)

3. **Declination (D)**: Angle from geographic north
   - Varies globally, typically ±30°
   - For 2D profiles, often set to 0°

### Magnetic Anomaly

The **magnetic anomaly** is the deviation of the observed magnetic field from the expected background (regional) field:

```
ΔB = B_observed - B_regional
```

In forward modeling, we calculate the anomaly caused by subsurface bodies with different susceptibilities than the surrounding rocks. The background field is implicitly assumed to be the Earth's field with no local variations.

## Mathematical Formulation

### 2D Approximation

The model assumes bodies have infinite extent perpendicular to the profile (the "strike" direction). This is valid when:
- Body length >> body width
- Profile is perpendicular to strike
- We're interested in the central portion of the body

For 2D bodies, the magnetic potential at point (x, z) due to a polygonal body is:

```
U(x,z) = (χ * F / μ₀) * Σ G_i
```

where G_i represents the contribution from edge i of the polygon.

### Edge Contributions

For each edge defined by vertices (x₁, z₁) and (x₂, z₂), the contribution involves:

1. **Geometric terms**: Functions of the edge position relative to observation point
2. **Field direction terms**: Components of the inducing field (inclination, declination)
3. **Logarithmic and arctangent functions**: Arising from integration along the edge

The total field anomaly at the observation point is:

```
ΔB = -∂U/∂x * cos(I) - ∂U/∂z * sin(I)
```

for a profile perpendicular to strike (declination = 0).

### Superposition Principle

When multiple bodies are present, the total anomaly is the sum of individual anomalies:

```
ΔB_total = Σ ΔB_i
```

This linearity arises from the linear relationship between magnetization and susceptibility in the induced-only case.

## Coordinate System

### Conventions

- **X-axis**: Horizontal position along profile (meters)
  - Typically increases left-to-right
  - No inherent origin; can be centered on bodies or set arbitrarily

- **Z-axis**: Depth below observation surface (meters)
  - **Positive downward** (following geophysics convention)
  - Origin (z=0) is typically the observation surface
  - Bodies are at positive z values (below surface)

- **Y-axis**: Perpendicular to profile (into the page)
  - Assumed infinite extent
  - Not explicitly modeled

### Observation Points

Observation points are typically along a horizontal line at z = 0 (ground surface) or z = constant (airborne survey). The algorithm computes the vertical component of the magnetic anomaly at each observation point.

## Algorithm Implementation

### Input Data

1. **Bodies**: List of polygons, each defined by:
   - Ordered vertices (x, z) defining the cross-section
   - Magnetic susceptibility (SI units)
   - Optional name for identification

2. **Magnetic field**: Earth's field parameters
   - Intensity (nT)
   - Inclination (degrees)
   - Declination (degrees)

3. **Observation points**: X-coordinates where anomaly is calculated
4. **Observation depth**: Z-coordinate of observation points (usually 0)

### Computation Steps

1. **Validate input**: Check geometry, ensure closed polygons, verify parameters

2. **For each observation point**:
   a. Initialize total anomaly to zero
   b. For each body:
      - Transform coordinates to observation point reference frame
      - For each edge of the body polygon:
        * Calculate geometric terms
        * Apply field direction factors
        * Sum edge contributions
      - Add body contribution to total
   c. Store computed anomaly value

3. **Output results**: Anomaly array corresponding to observation points

### Numerical Considerations

- **Singularities**: When observation point lies on a body edge, special handling prevents division by zero
- **Precision**: Double-precision (float64) used throughout
- **Stability**: Arctangent functions use atan2 for correct quadrant handling

## Assumptions and Limitations

### Assumptions

1. **2D geometry**: Bodies have infinite extent perpendicular to profile
2. **Total magnetization**: Both induced and remanent magnetization are modeled; remanent defaults to zero
3. **Uniform susceptibility**: Each body has constant susceptibility
4. **Non-interacting bodies**: Bodies don't affect each other's magnetization
5. **No demagnetization**: Shape of body doesn't affect its magnetization
6. **Flat observation surface**: All observation points at same elevation

### Limitations

1. **Not suitable for**:
   - 3D bodies (finite strike length)
   - Self-demagnetization effects
   - Very high susceptibility contrast (χ > 1)

2. **Accuracy depends on**:
   - Validity of 2D approximation
   - Knowledge of susceptibility values
   - Accuracy of field parameters
   - Appropriateness of polygon approximation

## Validation and Testing

The implementation can be validated by:

1. **Simple geometries**: Comparing to analytical solutions for simple shapes
2. **Known benchmarks**: Reproducing published examples
3. **Physical consistency**: Checking that anomalies behave as expected
4. **Conservation**: Verifying superposition principle

## References

1. **Talwani, M., Worzel, J. L., & Landisman, M. (1959).** Rapid gravity computations for two-dimensional bodies with application to the Mendocino submarine fracture zone. *Journal of Geophysical Research*, 64(1), 49-59.

2. **Talwani, M., & Heirtzler, J. R. (1964).** Computation of magnetic anomalies caused by two-dimensional structures of arbitrary shape. *Computers in the Mineral Industries*, 464-480.

3. **Blakely, R. J. (1995).** *Potential Theory in Gravity and Magnetic Applications.* Cambridge University Press. (Comprehensive textbook on potential field methods)

4. **Dobrin, M. B., & Savit, C. H. (1988).** *Introduction to Geophysical Prospecting* (4th ed.). McGraw-Hill. (Classical geophysics textbook)

5. **Sharma, P. V. (1997).** *Environmental and Engineering Geophysics.* Cambridge University Press. (Applied geophysics with magnetic methods)

## Further Reading

For practical usage examples, see `examples.md`.

For API documentation, see the package docstrings or generate documentation with:
```bash
python -m pydoc forward_model
```
