# Type-Dispatched `calculate_anomaly()` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generalize `calculate_anomaly()` to accept and dispatch on all model types (ForwardModel, GravityModel, HeatFlowModel), keeping existing magnetic behavior unchanged.

**Architecture:** Add `@overload` stubs and isinstance-based dispatch inside `calculate_anomaly()`. Rename `AnomalyComponents` to `MagneticComponents` with a backward-compat alias. Gravity delegates to the existing `calculate_gravity()`. HeatFlow raises `NotImplementedError`.

**Tech Stack:** Python, NumPy, Pydantic, pytest

---

## Context

`calculate_anomaly()` in `forward_model/compute/calculator.py` currently accepts only `ForwardModel` and is hardcoded to the magnetic domain.

> **Note:** `ForwardModel` is specifically the magnetic model (its discriminator is already `"magnetic"`). Renaming it to `MagneticModel` for consistency with `GravityModel`/`HeatFlowModel` is intentionally deferred to a separate ticket — this plan leaves `ForwardModel` unchanged. `GravityModel` and `HeatFlowModel` have models and compute kernels (gravity) or stubs (heat flow), but no path through the public API. This ticket wires them into the existing unified function, enabling `calculate_anomaly(gravity_model)` to return `GravityComponents` and correctly rejecting heat flow with a descriptive error.

---

## Critical Files

| File | Role |
|------|------|
| `forward_model/compute/talwani.py:17` | `AnomalyComponents` dataclass — rename to `MagneticComponents` |
| `forward_model/compute/calculator.py` | `calculate_anomaly()` — add overloads + dispatch |
| `forward_model/compute/gravity.py:221` | `calculate_gravity()` — called by gravity dispatch path |
| `forward_model/compute/__init__.py` | Export `MagneticComponents`, keep `AnomalyComponents` alias |
| `forward_model/__init__.py` | Top-level exports — add `MagneticComponents`, `GravityComponents` |
| `tests/unit/compute/test_calculator.py` | Add dispatch tests |
| `tests/conftest.py` | `gravity_model` and `heat_flow_model` fixtures (already exist) |

---

### Task 1: Rename `AnomalyComponents` → `MagneticComponents` in talwani.py

**Files:**
- Modify: `forward_model/compute/talwani.py:17-35`

**Step 1: Write the failing test**

In `tests/unit/compute/test_talwani.py`, add:
```python
from forward_model.compute.talwani import MagneticComponents

def test_magnetic_components_alias():
    """MagneticComponents is the canonical name; AnomalyComponents is a backward-compat alias."""
    from forward_model.compute.talwani import AnomalyComponents
    assert MagneticComponents is AnomalyComponents
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/compute/test_talwani.py::test_magnetic_components_alias -v`
Expected: `ImportError: cannot import name 'MagneticComponents'`

**Step 3: Rename the dataclass in talwani.py**

In `forward_model/compute/talwani.py`, change:
```python
@dataclass
class AnomalyComponents:
    ...
```
to:
```python
@dataclass
class MagneticComponents:
    """All magnetic anomaly components at each observation point.

    Attributes:
        bz: Vertical component of the magnetic anomaly (nT).
        bx: Horizontal component of the magnetic anomaly (nT).
        total_field: Total field anomaly ΔT (nT), the projection of the anomaly
                     vector onto the inducing field direction.
        amplitude: Vector amplitude |ΔB| = sqrt(Bx² + Bz²) in nT.
        gradient: Horizontal gradient of the total field anomaly d(ΔT)/dx
                  (nT/m). Forward model of what a total-field gradiometer
                  measures along the profile.
    """

    bz: NDArray[np.float64]
    bx: NDArray[np.float64]
    total_field: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    gradient: NDArray[np.float64]


# Backward-compatible alias — external code using AnomalyComponents keeps working
AnomalyComponents = MagneticComponents
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/compute/test_talwani.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add forward_model/compute/talwani.py tests/unit/compute/test_talwani.py
git commit -m "refactor: rename AnomalyComponents → MagneticComponents, keep alias"
```

---

### Task 2: Update `calculator.py` to use `MagneticComponents`

**Files:**
- Modify: `forward_model/compute/calculator.py:10,101,110,227`

**Step 1: Write the failing test**

In `tests/unit/compute/test_calculator.py`, add:
```python
from forward_model.compute.talwani import MagneticComponents

def test_calculate_anomaly_all_returns_magnetic_components(simple_model: ForwardModel) -> None:
    """calculate_anomaly(..., component='all') returns a MagneticComponents instance."""
    result = calculate_anomaly(simple_model, component="all")
    assert isinstance(result, MagneticComponents)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/compute/test_calculator.py::TestCalculateAnomaly::test_calculate_anomaly_all_returns_magnetic_components -v`
Expected: `ImportError: cannot import name 'MagneticComponents'` or assertion failure

**Step 3: Update imports in calculator.py**

Change the import block at the top of `forward_model/compute/calculator.py`:
```python
from forward_model.compute.talwani import (
    MagneticComponents,
    AnomalyComponents,   # keep for any existing internal references during migration
    PolygonComponents,
    compute_polygon_anomaly,
    compute_polygon_anomaly_2_5d,
    compute_polygon_anomaly_2_75d,
    field_to_magnetization,
)
```

Then update every reference of `AnomalyComponents` in the file body to `MagneticComponents`:
- Line 101: `@overload` return type `AnomalyComponents` → `MagneticComponents`
- Line 110: implementation return type `AnomalyComponents` → `MagneticComponents`
- Line 227: `return AnomalyComponents(...)` → `return MagneticComponents(...)`

Remove the `AnomalyComponents` import once all references are updated (the alias in talwani.py still preserves external compat).

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/compute/test_calculator.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add forward_model/compute/calculator.py tests/unit/compute/test_calculator.py
git commit -m "refactor: update calculator.py to use MagneticComponents"
```

---

### Task 3: Add dispatch to `calculate_anomaly()` for GravityModel and HeatFlowModel

**Files:**
- Modify: `forward_model/compute/calculator.py`

**Step 1: Write failing tests**

In `tests/unit/compute/test_calculator.py`, add a new test class:

```python
from forward_model.compute.gravity import GravityComponents
from forward_model.models import GravityModel, HeatFlowModel


class TestCalculateAnomalyDispatch:
    """Tests for type-dispatched calculate_anomaly."""

    def test_gravity_model_returns_gravity_components(
        self, gravity_model: GravityModel
    ) -> None:
        """calculate_anomaly(gravity_model) returns GravityComponents."""
        result = calculate_anomaly(gravity_model)
        assert isinstance(result, GravityComponents)
        assert result.gz.shape == (7,)
        assert np.all(np.isfinite(result.gz))

    def test_gravity_model_parallel(self, gravity_model: GravityModel) -> None:
        """calculate_anomaly(gravity_model, parallel=True) matches serial result."""
        serial = calculate_anomaly(gravity_model, parallel=False)
        parallel = calculate_anomaly(gravity_model, parallel=True)
        assert np.allclose(serial.gz, parallel.gz, rtol=1e-12)

    def test_gravity_model_correct_units(self, gravity_model: GravityModel) -> None:
        """Gravity result is in mGal (reasonable range for crustal anomalies)."""
        result = calculate_anomaly(gravity_model)
        # Crustal anomalies are typically < 1000 mGal; a simple body should be < 100 mGal
        assert np.all(np.abs(result.gz) < 1000.0)

    def test_heat_flow_model_raises_not_implemented(
        self, heat_flow_model: HeatFlowModel
    ) -> None:
        """calculate_anomaly(heat_flow_model) raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Heat flow"):
            calculate_anomaly(heat_flow_model)

    def test_forward_model_unchanged(self, simple_model: ForwardModel) -> None:
        """Backward compat: ForwardModel still returns NDArray by default."""
        result = calculate_anomaly(simple_model)
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/compute/test_calculator.py::TestCalculateAnomalyDispatch -v`
Expected: Multiple failures — `TypeError` (wrong model type) and `NotImplementedError` not raised

**Step 3: Update calculate_anomaly() in calculator.py**

Add imports at the top of `forward_model/compute/calculator.py`:
```python
from forward_model.compute.gravity import GravityComponents, calculate_gravity
from forward_model.models.gravity_model import GravityModel
from forward_model.models.heatflow_model import HeatFlowModel
```

Replace the existing two `@overload` stubs and function signature with:

```python
@overload
def calculate_anomaly(
    model: ForwardModel,
    parallel: bool = ...,
    component: Literal["bz", "bx", "total_field", "amplitude", "gradient"] = ...,
) -> NDArray[np.float64]: ...


@overload
def calculate_anomaly(
    model: ForwardModel,
    parallel: bool = ...,
    *,
    component: Literal["all"],
) -> MagneticComponents: ...


@overload
def calculate_anomaly(
    model: GravityModel,
    parallel: bool = ...,
) -> GravityComponents: ...


def calculate_anomaly(
    model: ForwardModel | GravityModel | HeatFlowModel,
    parallel: bool = False,
    component: Literal[
        "bz", "bx", "total_field", "amplitude", "gradient", "all"
    ] = "bz",
) -> NDArray[np.float64] | MagneticComponents | GravityComponents:
    """Calculate anomaly for a forward model, dispatching on model type.

    Computes the anomaly using the appropriate algorithm for the model type:
    - ForwardModel (magnetic): Talwani (1965) algorithm, returns NDArray or MagneticComponents
    - GravityModel: Talwani (1959) algorithm, returns GravityComponents (gz in mGal)
    - HeatFlowModel: Not yet implemented; raises NotImplementedError

    Args:
        model: A ForwardModel, GravityModel, or HeatFlowModel instance.
        parallel: If True, compute each body's contribution in a separate process.
        component: For ForwardModel only — which magnetic component to return.
                   Ignored for GravityModel. One of:
                   ``"bz"`` (default), ``"bx"``, ``"total_field"``,
                   ``"amplitude"``, ``"gradient"``, ``"all"``.

    Returns:
        - ForwardModel: ``NDArray[np.float64]`` or ``MagneticComponents`` (when component="all")
        - GravityModel: ``GravityComponents`` with gz (mGal) and gz_gradient (mGal/m)

    Raises:
        NotImplementedError: If model is a HeatFlowModel.

    Example:
        >>> mag_model = load_model("magnetic.json")
        >>> anomaly = calculate_anomaly(mag_model)
        >>> grav_model = load_model("gravity.json")
        >>> components = calculate_anomaly(grav_model)
        >>> print(f"Max gz: {components.gz.max():.3f} mGal")
    """
    if isinstance(model, GravityModel):
        return calculate_gravity(model, parallel=parallel)

    if isinstance(model, HeatFlowModel):
        raise NotImplementedError(
            "Heat flow compute not yet implemented. "
            "Track progress in the project issue tracker."
        )

    # ForwardModel path — all existing logic unchanged below this point
    ...
```

The rest of the function body (from `observation_points = model.get_observation_points()` onward) stays unchanged.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/compute/test_calculator.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add forward_model/compute/calculator.py tests/unit/compute/test_calculator.py
git commit -m "feat: type-dispatched calculate_anomaly for GravityModel and HeatFlowModel"
```

---

### Task 4: Update `compute/__init__.py` exports

**Files:**
- Modify: `forward_model/compute/__init__.py`

**Step 1: Write the failing test**

In `tests/unit/compute/test_calculator.py`, add at module level:
```python
def test_magnetic_components_exported_from_compute() -> None:
    """MagneticComponents is importable from forward_model.compute."""
    from forward_model.compute import MagneticComponents  # noqa: F401
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/compute/test_calculator.py::test_magnetic_components_exported_from_compute -v`
Expected: `ImportError: cannot import name 'MagneticComponents' from 'forward_model.compute'`

**Step 3: Update compute/__init__.py**

```python
"""Magnetic and gravity anomaly computation algorithms."""

from forward_model.compute.batch import BatchResult, batch_calculate
from forward_model.compute.calculator import calculate_anomaly
from forward_model.compute.gravity import GravityComponents, calculate_gravity
from forward_model.compute.talwani import (
    AnomalyComponents,
    MagneticComponents,
    compute_demagnetization_factor,
    compute_polygon_anomaly,
    compute_polygon_anomaly_2_5d,
    compute_polygon_anomaly_2_75d,
    field_to_magnetization,
)

__all__ = [
    "AnomalyComponents",   # backward-compat alias
    "MagneticComponents",
    "BatchResult",
    "batch_calculate",
    "calculate_anomaly",
    "calculate_gravity",
    "compute_demagnetization_factor",
    "compute_polygon_anomaly",
    "compute_polygon_anomaly_2_5d",
    "compute_polygon_anomaly_2_75d",
    "field_to_magnetization",
    "GravityComponents",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/compute/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add forward_model/compute/__init__.py
git commit -m "chore: export MagneticComponents from forward_model.compute"
```

---

### Task 5: Update top-level `forward_model/__init__.py` exports

**Files:**
- Modify: `forward_model/__init__.py`

**Step 1: Write the failing test**

In `tests/unit/test_config.py` or a new dedicated test, add:
```python
def test_top_level_exports_magnetic_components() -> None:
    """MagneticComponents and GravityComponents are importable from forward_model."""
    from forward_model import GravityComponents, MagneticComponents  # noqa: F401
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest -k "test_top_level_exports_magnetic_components" -v`
Expected: `ImportError`

**Step 3: Update forward_model/__init__.py**

Add to the import block (after existing `AnomalyComponents` import):
```python
from forward_model.compute import AnomalyComponents as AnomalyComponents
from forward_model.compute import BatchResult as BatchResult
from forward_model.compute import GravityComponents as GravityComponents
from forward_model.compute import MagneticComponents as MagneticComponents
from forward_model.compute import batch_calculate as batch_calculate
from forward_model.compute import calculate_anomaly
```

Add to `__all__`:
```python
"GravityComponents",
"MagneticComponents",
```

Keep `"AnomalyComponents"` in `__all__` for backward compat.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (393+ tests)

**Step 5: Commit**

```bash
git add forward_model/__init__.py
git commit -m "chore: export MagneticComponents and GravityComponents from top-level package"
```

---

## Verification

Run the full test suite:
```bash
make test
```

Run linting and type-checking (ruff, yamllint, bandit, pyright):
```bash
make lint
```

Both commands should exit cleanly with no errors.

Spot-check the acceptance criteria manually in a Python REPL:

```python
from forward_model import load_model, calculate_anomaly, GravityComponents, MagneticComponents

# AC1: GravityModel returns GravityComponents in mGal
grav = load_model("examples/simple_dyke.json")  # swap for a gravity JSON fixture
# (or construct one inline — see tests/conftest.py gravity_model fixture)
result = calculate_anomaly(grav)
assert isinstance(result, GravityComponents)
print(result.gz)  # values should be in mGal

# AC2: ForwardModel behaviour unchanged
mag = load_model("examples/simple_dyke.json")
anomaly = calculate_anomaly(mag)                     # returns NDArray by default
components = calculate_anomaly(mag, component="all") # returns MagneticComponents
assert isinstance(components, MagneticComponents)

# AC3: HeatFlowModel raises NotImplementedError with a clear message
from forward_model import HeatFlowModel, GeologicBody, ThermalProperties
heat = HeatFlowModel(
    bodies=[GeologicBody(
        vertices=[[0,100],[50,100],[50,200],[0,200]],
        thermal=ThermalProperties(conductivity=2.5, heat_generation=2.0),
        name="test",
    )],
    observation_x=[0.0, 50.0, 100.0],
    observation_z=0.0,
)
try:
    calculate_anomaly(heat)
    assert False, "Should have raised"
except NotImplementedError as e:
    print(e)  # message should mention "Heat flow"
```
