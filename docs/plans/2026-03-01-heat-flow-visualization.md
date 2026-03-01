# Heat Flow Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make visualization layer heat-flow-aware so users can plot heat flow anomalies with appropriate units, labels, and thermal conductivity coloring.

**Architecture:** Extend existing mode-detection pattern (already used for GravityModel) to HeatFlowModel. Add heat flow component labels, thermal conductivity color-by option, and auto-detection in plot functions. Use warm sequential colormap for thermal properties.

**Tech Stack:** Matplotlib, NumPy, Pydantic models (HeatFlowModel), existing test patterns

---

## Task 1: Add heat flow component labels to `_COMPONENT_LABELS`

**Files:**
- Modify: `forward_model/viz/plotter.py:409-418`

**Step 1: Write the failing test**

Add to `tests/unit/viz/test_plotter.py` at line 792 (in TestComponentLabels class):

```python
def test_heatflow_ylabel(self) -> None:
    from forward_model.viz.plotter import _COMPONENT_LABELS

    ylabel, _ = _COMPONENT_LABELS["heatflow"]
    assert ylabel == "Heat Flow Anomaly (mW/m²)"

def test_heatflow_title(self) -> None:
    from forward_model.viz.plotter import _COMPONENT_LABELS

    _, title = _COMPONENT_LABELS["heatflow"]
    assert title == "Heat Flow Anomaly (mW/m²)"

def test_heatflow_gradient_ylabel(self) -> None:
    from forward_model.viz.plotter import _COMPONENT_LABELS

    ylabel, _ = _COMPONENT_LABELS["heatflow_gradient"]
    assert ylabel == "Heat Flow Gradient (mW/m²/m)"

def test_heatflow_gradient_title(self) -> None:
    from forward_model.viz.plotter import _COMPONENT_LABELS

    _, title = _COMPONENT_LABELS["heatflow_gradient"]
    assert title == "Heat Flow Gradient (mW/m²/m)"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/viz/test_plotter.py::TestComponentLabels::test_heatflow_ylabel -v`
Expected: FAIL with "KeyError: 'heatflow'"

**Step 3: Add heat flow entries to _COMPONENT_LABELS**

In `forward_model/viz/plotter.py`, update the `_COMPONENT_LABELS` dict (lines 409-418):

```python
_COMPONENT_LABELS: dict[str, tuple[str, str]] = {
    "bz": ("Bz (nT)", "Vertical Component (Bz)"),
    "bx": ("Bx (nT)", "Horizontal Component (Bx)"),
    "total_field": ("ΔT (nT)", "Total Field Anomaly (ΔT)"),
    "amplitude": ("|ΔB| (nT)", "Anomaly Amplitude (|ΔB|)"),
    "gradient": ("d(ΔT)/dx (nT/m)", "Horizontal Gradient d(ΔT)/dx"),
    # Gravity components
    "gz": ("gz (mGal)", "Vertical Gravity Anomaly (gz)"),
    "gz_gradient": ("gz gradient (mGal/m)", "Horizontal Gradient d(gz)/dx"),
    # Heat flow components
    "heatflow": ("Heat Flow Anomaly (mW/m²)", "Heat Flow Anomaly (mW/m²)"),
    "heatflow_gradient": ("Heat Flow Gradient (mW/m²/m)", "Heat Flow Gradient (mW/m²/m)"),
}
```

**Step 4: Run all component label tests to verify they pass**

Run: `pytest tests/unit/viz/test_plotter.py::TestComponentLabels -v`
Expected: PASS (all 8 tests including the 4 new heat flow tests)

**Step 5: Commit**

```bash
git add forward_model/viz/plotter.py tests/unit/viz/test_plotter.py
git commit -m "feat: add heat flow component labels to _COMPONENT_LABELS"
```

---

## Task 2: Add thermal conductivity color-by support to `plot_model()`

**Files:**
- Modify: `forward_model/viz/plotter.py:69-286`
- Test: `tests/unit/viz/test_plotter.py`
- Import: Add HeatFlowModel import at line 20

**Step 1: Import HeatFlowModel**

Update the imports in `forward_model/viz/plotter.py` at line 20:

```python
from forward_model.models.body import GeologicBody
from forward_model.models.gravity_model import GravityModel
from forward_model.models.heatflow_model import HeatFlowModel
from forward_model.models.model import ForwardModel
```

**Step 2: Write tests for thermal conductivity coloring**

Add to `tests/unit/viz/test_plotter.py` (add new test class after TestPlotModelGravity):

```python
class TestPlotModelHeatFlow:
    """Tests for plot_model with HeatFlowModel input."""

    def test_heatflow_model_renders_without_error(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )

        ax = plot_model(model)
        assert ax is not None
        assert len(ax.patches) > 0
        plt.close()

    def test_heatflow_body_label_shows_conductivity(self) -> None:
        from matplotlib.text import Annotation, Text

        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )

        ax = plot_model(model)
        texts = [
            c
            for c in ax.get_children()
            if isinstance(c, Text)
            and not isinstance(c, Annotation)
            and "HeatFlowBody" in c.get_text()
        ]
        assert len(texts) == 1
        assert "2.5" in texts[0].get_text()
        assert "W/m·K" in texts[0].get_text()
        assert "χ=" not in texts[0].get_text()
        plt.close()

    def test_color_by_thermal_conductivity(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.0),
            name="Body1",
        )
        body2 = GeologicBody(
            vertices=[[60.0, 100.0], [110.0, 100.0], [110.0, 200.0], [60.0, 200.0]],
            thermal=ThermalProperties(conductivity=4.0),
            name="Body2",
        )
        model = HeatFlowModel(
            bodies=[body1, body2],
            observation_x=[0.0, 55.0, 110.0],
            observation_z=0.0,
        )

        ax = plot_model(model, color_by="thermal_conductivity")
        assert len(ax.patches) > 0
        plt.close()

    def test_heatflow_default_color_by_is_thermal_conductivity(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.0),
            name="Body1",
        )
        body2 = GeologicBody(
            vertices=[[60.0, 100.0], [110.0, 100.0], [110.0, 200.0], [60.0, 200.0]],
            thermal=ThermalProperties(conductivity=4.0),
            name="Body2",
        )
        model = HeatFlowModel(
            bodies=[body1, body2],
            observation_x=[0.0, 55.0, 110.0],
            observation_z=0.0,
        )

        # Test with color_by=None (should default to thermal_conductivity for HeatFlowModel)
        fig, ax = plt.subplots()
        plot_model(model, ax=ax, show_colorbar=True)
        # Colorbar should be present, indicating continuous coloring
        assert len(fig.axes) == 2  # colorbar axis added
        plt.close()

    def test_heatflow_colorbar_label_is_conductivity(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.0),
            name="Body1",
        )
        body2 = GeologicBody(
            vertices=[[60.0, 100.0], [110.0, 100.0], [110.0, 200.0], [60.0, 200.0]],
            thermal=ThermalProperties(conductivity=4.0),
            name="Body2",
        )
        model = HeatFlowModel(
            bodies=[body1, body2],
            observation_x=[0.0, 55.0, 110.0],
            observation_z=0.0,
        )

        fig, ax = plt.subplots()
        plot_model(model, ax=ax, show_colorbar=True)
        assert len(fig.axes) == 2  # colorbar axis added
        colorbar_ax = fig.axes[1]
        ylabel = colorbar_ax.get_ylabel()
        assert "W/m·K" in ylabel or "Thermal" in ylabel
        plt.close()
```

**Step 3: Run the new tests to verify they fail**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotModelHeatFlow -v`
Expected: FAIL (HeatFlowModel not imported, thermal_conductivity not supported)

**Step 4: Update plot_model() signature and implementation**

Modify the `plot_model()` function signature (line 69-72):

```python
def plot_model(
    model: ForwardModel | GravityModel | HeatFlowModel,
    ax: Axes | None = None,
    color_by: Literal["index", "susceptibility", "density", "thermal_conductivity"] | None = None,
    show_observation_lines: bool = True,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_colorbar: bool = True,
    equal_aspect: bool = True,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    show_label_arrows: bool | dict[str, bool] = False,
) -> Axes:
```

Update the docstring (line 86-93) to add thermal_conductivity option:

```python
    color_by: How to color bodies. "index" uses different colors for each
             body, "susceptibility" uses a colormap based on susceptibility,
             "density" uses a colormap based on density contrast,
             "thermal_conductivity" uses a colormap based on thermal conductivity.
             If None (default), resolves to "density" for GravityModel,
             "thermal_conductivity" for HeatFlowModel, and "susceptibility"
             for ForwardModel.
```

Update the color resolution logic (lines 114-124) to include HeatFlowModel:

```python
    # Resolve None sentinel
    _is_gravity = isinstance(model, GravityModel)
    _is_heatflow = isinstance(model, HeatFlowModel)
    if color_by is None:
        _effective_color_by: Literal["index", "susceptibility", "density", "thermal_conductivity"] = (
            "density" if _is_gravity
            else ("thermal_conductivity" if _is_heatflow else "susceptibility")
        )
    else:
        _effective_color_by = color_by
```

Add thermal conductivity coloring logic after the susceptibility block (after line 176), before the "index" case:

```python
    elif _effective_color_by == "thermal_conductivity":
        thermal_values = [
            body.thermal.conductivity
            for body in model.bodies
            if body.thermal is not None
        ]
        if not thermal_values or len(set(thermal_values)) == 1:
            colors = [
                cmap(0.5) if body.thermal is not None else _FALLBACK_COLOR
                for body in model.bodies
            ]
        else:
            _colorbar_norm = plt.Normalize(vmin=min(thermal_values), vmax=max(thermal_values))  # type: ignore
            colors = [
                (
                    cmap(_colorbar_norm(body.thermal.conductivity))
                    if body.thermal is not None
                    else _FALLBACK_COLOR
                )
                for body in model.bodies
            ]  # type: ignore
            _auto_colorbar = True
```

Update the label logic (lines 205-210) to include thermal conductivity:

```python
        if body.gravity is not None:
            label = f"{body.name}\n(ρ={body.gravity.density_contrast:.1f} kg/m³)"
        elif body.thermal is not None:
            label = f"{body.name}\n(κ={body.thermal.conductivity:.2f} W/m·K)"
        elif body.magnetic is not None:
            label = f"{body.name}\n(χ={body.magnetic.susceptibility:.3f})"
        else:
            label = body.name
```

Update the colorbar labels (lines 240-249) to include thermal conductivity:

```python
    # Add colorbar if coloring with multiple values
    if _auto_colorbar and show_colorbar:
        if _effective_color_by == "density":
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=_colorbar_norm)  # type: ignore
            sm.set_array([])  # type: ignore
            plt.colorbar(sm, ax=ax, label="Density Contrast (kg/m³)")  # type: ignore
        elif _effective_color_by == "thermal_conductivity":
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=_colorbar_norm)  # type: ignore
            sm.set_array([])  # type: ignore
            plt.colorbar(sm, ax=ax, label="Thermal Conductivity (W/m·K)")  # type: ignore
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=_colorbar_norm)  # type: ignore
            sm.set_array([])  # type: ignore
            plt.colorbar(sm, ax=ax, label="Susceptibility (SI)")  # type: ignore
```

**Step 5: Run the new tests to verify they pass**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotModelHeatFlow -v`
Expected: PASS (all 5 tests)

**Step 6: Run all visualization tests to ensure no regression**

Run: `pytest tests/unit/viz/test_plotter.py -v`
Expected: PASS (all existing tests + new heat flow tests)

**Step 7: Commit**

```bash
git add forward_model/viz/plotter.py tests/unit/viz/test_plotter.py
git commit -m "feat: add thermal conductivity coloring to plot_model"
```

---

## Task 3: Add heat flow model detection to `plot_anomaly()`

**Files:**
- Modify: `forward_model/viz/plotter.py:421-503`
- Test: `tests/unit/viz/test_plotter.py`

**Step 1: Write tests for heat flow anomaly plotting**

Add to `tests/unit/viz/test_plotter.py` (after the TestPlotAnomaly class):

```python
def test_heatflow_component_ylabel(self) -> None:
    obs_x = [0.0, 10.0, 20.0]
    anomaly = np.array([50.0, 65.0, 40.0])
    ax = plot_anomaly(obs_x, anomaly, component="heatflow")
    assert ax.get_ylabel() == "Heat Flow Anomaly (mW/m²)"
    plt.close()

def test_heatflow_component_title(self) -> None:
    obs_x = [0.0, 10.0, 20.0]
    anomaly = np.array([50.0, 65.0, 40.0])
    ax = plot_anomaly(obs_x, anomaly, component="heatflow")
    assert ax.get_title() == "Heat Flow Anomaly (mW/m²)"
    plt.close()

def test_heatflow_gradient_component_ylabel(self) -> None:
    obs_x = [0.0, 10.0, 20.0]
    anomaly = np.array([0.1, 0.05, -0.02])
    ax = plot_anomaly(obs_x, anomaly, component="heatflow_gradient")
    assert ax.get_ylabel() == "Heat Flow Gradient (mW/m²/m)"
    plt.close()

def test_heatflow_gradient_component_title(self) -> None:
    obs_x = [0.0, 10.0, 20.0]
    anomaly = np.array([0.1, 0.05, -0.02])
    ax = plot_anomaly(obs_x, anomaly, component="heatflow_gradient")
    assert ax.get_title() == "Heat Flow Gradient (mW/m²/m)"
    plt.close()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotAnomaly::test_heatflow_component_ylabel -v`
Expected: FAIL (KeyError accessing _COMPONENT_LABELS["heatflow"])

**Step 3: Verify plot_anomaly uses _COMPONENT_LABELS correctly**

No changes needed to `plot_anomaly()` — it already uses `_COMPONENT_LABELS.get()` which will find our newly added heat flow entries from Task 1.

**Step 4: Run the new tests to verify they pass**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotAnomaly::test_heatflow_component_ylabel tests/unit/viz/test_plotter.py::TestPlotAnomaly::test_heatflow_component_title tests/unit/viz/test_plotter.py::TestPlotAnomaly::test_heatflow_gradient_component_ylabel tests/unit/viz/test_plotter.py::TestPlotAnomaly::test_heatflow_gradient_component_title -v`
Expected: PASS (all 4 tests)

**Step 5: Run all anomaly tests**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotAnomaly -v`
Expected: PASS (all existing + new tests)

**Step 6: Commit**

```bash
git add tests/unit/viz/test_plotter.py
git commit -m "test: add heat flow component labeling tests for plot_anomaly"
```

---

## Task 4: Add heat flow mode detection to `plot_combined()`

**Files:**
- Modify: `forward_model/viz/plotter.py:506-670`
- Test: `tests/unit/viz/test_plotter.py`

**Step 1: Write tests for heat flow combined plotting**

Add a new test class to `tests/unit/viz/test_plotter.py` (after TestPlotCombinedGravity):

```python
class TestPlotCombinedHeatFlow:
    """Tests for plot_combined with HeatFlowModel."""

    def test_heatflow_model_creates_figure(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )
        anomaly = np.array([45.0, 60.0, 50.0])

        fig = plot_combined(model, anomaly)
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close()

    def test_anomaly_ylabel_is_mw_per_m2(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )
        anomaly = np.array([45.0, 60.0, 50.0])

        fig = plot_combined(model, anomaly)
        ylabel = fig.axes[1].get_ylabel()
        assert "mW/m²" in ylabel
        plt.close()

    def test_cross_section_label_shows_conductivity(self) -> None:
        from matplotlib.text import Annotation, Text

        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )
        anomaly = np.array([45.0, 60.0, 50.0])

        fig = plot_combined(model, anomaly)
        cross_ax = fig.axes[0]
        texts = [
            c
            for c in cross_ax.get_children()
            if isinstance(c, Text)
            and not isinstance(c, Annotation)
            and "HeatFlowBody" in c.get_text()
        ]
        assert len(texts) == 1
        assert "W/m·K" in texts[0].get_text()
        assert "χ=" not in texts[0].get_text()
        plt.close()

    def test_explicit_component_heatflow(self) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )
        anomaly = np.array([45.0, 60.0, 50.0])

        fig = plot_combined(model, anomaly, component="heatflow")
        assert fig.axes[1].get_ylabel() == "Heat Flow Anomaly (mW/m²)"
        plt.close()

    def test_saves_to_file(self, tmp_path: Path) -> None:
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        body = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5),
            name="HeatFlowBody",
        )
        model = HeatFlowModel(
            bodies=[body],
            observation_x=[0.0, 25.0, 50.0],
            observation_z=0.0,
        )
        anomaly = np.array([45.0, 60.0, 50.0])
        output = tmp_path / "heatflow_combined.png"

        fig = plot_combined(model, anomaly, save_path=output)
        assert output.exists()
        plt.close(fig)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotCombinedHeatFlow::test_heatflow_model_creates_figure -v`
Expected: FAIL (HeatFlowModel not in type union for plot_combined)

**Step 3: Update plot_combined() signature**

Modify the `plot_combined()` function signature (line 507):

```python
def plot_combined(
    model: ForwardModel | GravityModel | HeatFlowModel,
    ...
```

Update the docstring (line 534-535) to mention heat flow:

```python
        anomaly: Anomaly values (nT for magnetic components, mGal for gravity
                components, mW/m² for heat flow components).
```

**Step 4: Update plot_combined() mode detection logic**

Modify the mode detection section (lines 569-578):

```python
    # Resolve sentinel defaults based on model type
    _is_gravity = isinstance(model, GravityModel)
    _is_heatflow = isinstance(model, HeatFlowModel)
    _effective_component: str = (
        ("gz" if _is_gravity
         else ("heatflow" if _is_heatflow else "total_field"))
        if component is None else component
    )
    _effective_color_by: Literal["index", "susceptibility", "density", "thermal_conductivity"] = (
        ("density" if _is_gravity
         else ("thermal_conductivity" if _is_heatflow else "susceptibility"))
        if color_by is None
        else color_by
    )
```

**Step 5: Run the new tests to verify they pass**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotCombinedHeatFlow -v`
Expected: PASS (all 5 tests)

**Step 6: Run all combined plot tests**

Run: `pytest tests/unit/viz/test_plotter.py::TestPlotCombined -v`
Expected: PASS (all existing tests)

**Step 7: Commit**

```bash
git add forward_model/viz/plotter.py tests/unit/viz/test_plotter.py
git commit -m "feat: add heat flow mode detection to plot_combined"
```

---

## Task 5: Verify integration and run full test suite

**Files:**
- Test: `tests/unit/viz/test_plotter.py`

**Step 1: Run all visualization tests**

Run: `pytest tests/unit/viz/test_plotter.py -v`
Expected: PASS (all tests, including new heat flow tests)

**Step 2: Run integration tests**

Run: `pytest tests/integration/test_workflow.py -v`
Expected: PASS (existing workflows should still work)

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS (all tests should pass)

**Step 4: Commit (if any fixes were needed)**

If no fixes were needed, skip this step. Otherwise:

```bash
git add -A
git commit -m "fix: address test failures from heat flow visualization changes"
```

---

## Task 6: Create integration test for end-to-end heat flow workflow

**Files:**
- Modify: `tests/unit/viz/test_plotter.py` (add new test class)

**Step 1: Write an integration test**

Add to `tests/unit/viz/test_plotter.py`:

```python
class TestHeatFlowWorkflow:
    """Integration test: end-to-end heat flow modeling workflow."""

    def test_heatflow_complete_workflow(self, tmp_path: Path) -> None:
        """Smoke test: Load model, compute anomaly, plot combined view."""
        from forward_model.models import GeologicBody
        from forward_model.models.heatflow_model import HeatFlowModel
        from forward_model.models.properties import ThermalProperties

        # Create model
        body1 = GeologicBody(
            vertices=[[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            thermal=ThermalProperties(conductivity=2.5, heat_generation=1.0),
            name="Conductor",
        )
        body2 = GeologicBody(
            vertices=[[60.0, 100.0], [110.0, 100.0], [110.0, 200.0], [60.0, 200.0]],
            thermal=ThermalProperties(conductivity=1.5, heat_generation=0.5),
            name="Insulator",
        )
        model = HeatFlowModel(
            bodies=[body1, body2],
            observation_x=[0.0, 25.0, 50.0, 75.0, 100.0, 125.0],
            observation_z=0.0,
            background_heat_flow=65.0,
        )

        # Simulate anomaly (would come from compute.calculator in real workflow)
        anomaly = np.array([55.0, 65.0, 75.0, 70.0, 60.0, 50.0])
        gradient = np.array([0.2, 0.1, -0.05, -0.1, -0.15, -0.2])

        # Plot combined view with all features
        output = tmp_path / "heatflow_workflow.png"
        fig = plot_combined(
            model,
            anomaly,
            save_path=output,
            color_by="thermal_conductivity",
            gradient=gradient,
            component="heatflow",
        )

        # Verify output
        assert output.exists()
        assert len(fig.axes) >= 3  # cross-section, anomaly, colorbar, gradient twin

        # Verify labels
        cross_ax = fig.axes[0]
        anomaly_ax = fig.axes[1]

        assert "W/m·K" in cross_ax.patches[0].__repr__()  # body is rendered
        assert "mW/m²" in anomaly_ax.get_ylabel()

        plt.close(fig)
```

**Step 2: Run the integration test**

Run: `pytest tests/unit/viz/test_plotter.py::TestHeatFlowWorkflow::test_heatflow_complete_workflow -v`
Expected: PASS

**Step 3: Run all visualization tests one more time**

Run: `pytest tests/unit/viz/test_plotter.py -v`
Expected: PASS (all tests including integration)

**Step 4: Commit**

```bash
git add tests/unit/viz/test_plotter.py
git commit -m "test: add end-to-end heat flow workflow integration test"
```

---

## Acceptance Criteria

✓ Heat flow component labels ("heatflow", "heatflow_gradient") added to `_COMPONENT_LABELS`
✓ `plot_model()` accepts HeatFlowModel and displays thermal conductivity in labels
✓ `plot_model()` supports `color_by="thermal_conductivity"` with warm colormap
✓ `plot_model()` defaults to thermal_conductivity coloring for HeatFlowModel
✓ `plot_anomaly()` correctly labels heat flow components with mW/m² units
✓ `plot_combined()` auto-detects HeatFlowModel and applies appropriate defaults
✓ All existing tests pass (no regressions)
✓ New tests comprehensively cover heat flow visualization
✓ Integration test demonstrates complete end-to-end workflow

---

## Notes

- Heat flow unit: **mW/m²** (milliwatts per square meter)
- Thermal conductivity unit: **W/m·K** (watts per meter-kelvin)
- Colormap: Uses viridis (warm scale appropriate for heat)
- Mode detection: Follows existing pattern used for GravityModel
- Label format: `κ=X.XX W/m·K` (Greek kappa for conductivity)
