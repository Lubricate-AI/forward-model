# ForwardModel → MagneticModel Rename Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename `ForwardModel` to `MagneticModel` for API naming consistency across all three domain model types (magnetic, gravity, heat flow).

**Architecture:** Pure mechanical rename with no logic changes. Rename the class and file, update all imports/references across source, tests, benchmarks, and docs in dependency order. Verify with tests and grep at the end.

**Tech Stack:** Python, pytest, ruff, pyright, git

---

## Task 1: Rename model file and class

**Files:**
- Rename: `forward_model/models/model.py` → `forward_model/models/magnetic_model.py`
- Modify: (same file, after rename)

**Step 1: Rename the file**

```bash
cd /Users/joshuapoirier/conductor/workspaces/forward-model/yeosu
git mv forward_model/models/model.py forward_model/models/magnetic_model.py
```

Expected: File renamed, no output errors.

**Step 2: Open the renamed file and change the class name**

File: `forward_model/models/magnetic_model.py`

Find line ~15 (the class definition) and change:
```python
# OLD
class ForwardModel(BaseModel):
    """2D magnetic forward modeling using Talwani (1965) algorithm."""

# NEW
class MagneticModel(BaseModel):
    """2D magnetic forward modeling using Talwani (1965) algorithm."""
```

**Step 3: Verify file compiles**

```bash
python -c "from forward_model.models.magnetic_model import MagneticModel; print('OK')"
```

Expected: Error (other modules still import old name). Proceed anyway.

**Step 4: Commit**

```bash
git add forward_model/models/magnetic_model.py
git commit -m "refactor: rename model.py to magnetic_model.py and ForwardModel to MagneticModel"
```

---

## Task 2: Update forward_model/models/__init__.py

**Files:**
- Modify: `forward_model/models/__init__.py:1-30`

**Step 1: Update the import statement**

Line ~1: Change:
```python
# OLD
from forward_model.models.model import ForwardModel

# NEW
from forward_model.models.magnetic_model import MagneticModel
```

**Step 2: Update the AnyForwardModel union type**

Line ~3: Change:
```python
# OLD
AnyForwardModel = ForwardModel | GravityModel | HeatFlowModel

# NEW
AnyForwardModel = MagneticModel | GravityModel | HeatFlowModel
```

**Step 3: Update __all__ exports**

Line ~5-10 (find the `__all__` list): Change:
```python
# OLD
__all__ = ["ForwardModel", "GravityModel", "HeatFlowModel", "AnyForwardModel"]

# NEW
__all__ = ["MagneticModel", "GravityModel", "HeatFlowModel", "AnyForwardModel"]
```

**Step 4: Verify imports work**

```bash
python -c "from forward_model.models import MagneticModel, AnyForwardModel; print('OK')"
```

Expected: OK

**Step 5: Commit**

```bash
git add forward_model/models/__init__.py
git commit -m "refactor: update models/__init__.py for MagneticModel import"
```

---

## Task 3: Update forward_model/__init__.py

**Files:**
- Modify: `forward_model/__init__.py:1-30`

**Step 1: Update the re-export**

Find the line importing from models (line ~1-5): Change:
```python
# OLD
from forward_model.models import ForwardModel, GravityModel, HeatFlowModel, AnyForwardModel

# NEW
from forward_model.models import MagneticModel, GravityModel, HeatFlowModel, AnyForwardModel
```

**Step 2: Update __all__**

Find the `__all__` list: Change:
```python
# OLD
__all__ = [..., "ForwardModel", ...]

# NEW
__all__ = [..., "MagneticModel", ...]
```

**Step 3: Verify top-level imports**

```bash
python -c "from forward_model import MagneticModel; print('OK')"
```

Expected: OK

**Step 4: Commit**

```bash
git add forward_model/__init__.py
git commit -m "refactor: update top-level __init__.py to export MagneticModel"
```

---

## Task 4: Update forward_model/compute/calculator.py

**Files:**
- Modify: `forward_model/compute/calculator.py:1-200`

**Step 1: Update the import**

Line ~1-15: Change:
```python
# OLD
from forward_model.models.model import ForwardModel

# NEW
from forward_model.models.magnetic_model import MagneticModel
```

**Step 2: Update @overload signatures**

Lines ~30-50 (find all `@overload` decorators for magnetic models): Change:
```python
# OLD (appears 4 times)
@overload
def calculate_anomaly(model: ForwardModel, ...) -> MagneticComponents:
    ...

# NEW
@overload
def calculate_anomaly(model: MagneticModel, ...) -> MagneticComponents:
    ...
```

**Step 3: Update union type hint**

Find the line with union type (looks like `ForwardModel | GravityModel | HeatFlowModel`): Change:
```python
# OLD
model: ForwardModel | GravityModel | HeatFlowModel

# NEW
model: MagneticModel | GravityModel | HeatFlowModel
```

**Step 4: Update docstring references**

Find docstring mentions (search for "ForwardModel" in docstrings): Change all occurrences to `MagneticModel`.

**Step 5: Update inline comment**

Find any comment mentioning ForwardModel and update it.

**Step 6: Verify types**

```bash
cd /Users/joshuapoirier/conductor/workspaces/forward-model/yeosu
pyright forward_model/compute/calculator.py --outputjson | grep -c error || echo "No errors"
```

Expected: 0 errors (or only pre-existing)

**Step 7: Commit**

```bash
git add forward_model/compute/calculator.py
git commit -m "refactor: update calculator.py @overload signatures and docstrings for MagneticModel"
```

---

## Task 5: Update forward_model/compute/batch.py

**Files:**
- Modify: `forward_model/compute/batch.py:1-100`

**Step 1: Update the import**

Line ~1-10: Change:
```python
# OLD
from forward_model.models.model import ForwardModel

# NEW
from forward_model.models.magnetic_model import MagneticModel
```

**Step 2: Update isinstance check**

Find the line with `isinstance(model, ForwardModel)` and change:
```python
# OLD
if isinstance(model, ForwardModel):

# NEW
if isinstance(model, MagneticModel):
```

**Step 3: Commit**

```bash
git add forward_model/compute/batch.py
git commit -m "refactor: update batch.py imports and isinstance check for MagneticModel"
```

---

## Task 6: Update forward_model/cli/commands.py

**Files:**
- Modify: `forward_model/cli/commands.py:1-150`

**Step 1: Update the import**

Line ~1-10: Change:
```python
# OLD
from forward_model.models import ForwardModel, GravityModel, HeatFlowModel

# NEW
from forward_model.models import MagneticModel, GravityModel, HeatFlowModel
```

**Step 2: Update isinstance checks in run() function**

Find two lines with `isinstance(model, ForwardModel)` and change to `isinstance(model, MagneticModel)`.

**Step 3: Update validate() function**

Find `isinstance(model, ForwardModel)` in validate() and change to `isinstance(model, MagneticModel)`.

**Step 4: Update visualize() function**

Find `ForwardModel.model_validate(...)` and change to `MagneticModel.model_validate(...)`.

**Step 5: Commit**

```bash
git add forward_model/cli/commands.py
git commit -m "refactor: update cli/commands.py for MagneticModel"
```

---

## Task 7: Update forward_model/io/loaders.py

**Files:**
- Modify: `forward_model/io/loaders.py:1-150`

**Step 1: Update the import**

Line ~1-10: Change:
```python
# OLD
from forward_model.models import ForwardModel, AnyForwardModel

# NEW
from forward_model.models import MagneticModel, AnyForwardModel
```

**Step 2: Update TypeAdapter usage**

Find the line with `TypeAdapter(ForwardModel)` and change:
```python
# OLD
TypeAdapter(ForwardModel)

# NEW
TypeAdapter(MagneticModel)
```

**Step 3: Update docstring**

Find any docstring mentioning `ForwardModel` and change to `MagneticModel`.

**Step 4: Update model_validate call**

Find `ForwardModel.model_validate(model_data)` and change to `MagneticModel.model_validate(model_data)`.

**Step 5: Commit**

```bash
git add forward_model/io/loaders.py
git commit -m "refactor: update io/loaders.py for MagneticModel"
```

---

## Task 8: Update forward_model/viz/plotter.py

**Files:**
- Modify: `forward_model/viz/plotter.py:1-30`

**Step 1: Update the import**

Line ~1-10: Change:
```python
# OLD
from forward_model.models import ForwardModel

# NEW
from forward_model.models import MagneticModel
```

**Step 2: Commit**

```bash
git add forward_model/viz/plotter.py
git commit -m "refactor: update viz/plotter.py imports for MagneticModel"
```

---

## Task 9: Update forward_model/models/gravity_model.py docstring

**Files:**
- Modify: `forward_model/models/gravity_model.py:10`

**Step 1: Find and update docstring cross-reference**

Line ~10 (find the docstring line mentioning "Unlike ForwardModel"): Change:
```python
# OLD
Unlike `ForwardModel`, this model does not require...

# NEW
Unlike `MagneticModel`, this model does not require...
```

**Step 2: Commit**

```bash
git add forward_model/models/gravity_model.py
git commit -m "refactor: update gravity_model.py docstring cross-reference to MagneticModel"
```

---

## Task 10: Update test conftest.py

**Files:**
- Modify: `tests/conftest.py` (all occurrences)

**Step 1: Replace all ForwardModel with MagneticModel**

Use a global find-replace in the file (8 occurrences total):

```bash
cd /Users/joshuapoirier/conductor/workspaces/forward-model/yeosu
sed -i '' 's/ForwardModel/MagneticModel/g' tests/conftest.py
```

**Step 2: Verify syntax**

```bash
python -m py_compile tests/conftest.py
```

Expected: OK (no output)

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: update conftest.py for MagneticModel"
```

---

## Task 11: Update tests/unit/models/test_model.py

**Files:**
- Modify: `tests/unit/models/test_model.py` (~30 occurrences)

**Step 1: Replace all ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' tests/unit/models/test_model.py
```

**Step 2: Update test class name if applicable**

Check if the test class is named `TestForwardModel` and change to `TestMagneticModel`:

```bash
sed -i '' 's/class TestForwardModel/class TestMagneticModel/g' tests/unit/models/test_model.py
```

**Step 3: Run the tests**

```bash
pytest tests/unit/models/test_model.py -v
```

Expected: All tests PASS (should have same behavior as before)

**Step 4: Commit**

```bash
git add tests/unit/models/test_model.py
git commit -m "test: update test_model.py for MagneticModel"
```

---

## Task 12: Update remaining test files

**Files:**
- Modify: `tests/unit/compute/test_calculator.py`
- Modify: `tests/unit/compute/test_talwani.py`
- Modify: `tests/unit/compute/test_batch.py`
- Modify: `tests/unit/cli/test_commands.py`
- Modify: `tests/unit/io/test_loaders.py`
- Modify: `tests/unit/io/test_writers.py`
- Modify: `tests/unit/io/test_csv_loader.py`
- Modify: `tests/unit/viz/test_plotter.py`
- Modify: `tests/integration/test_workflow.py`

**Step 1: Batch replace in all test files**

```bash
for file in tests/unit/compute/test_calculator.py \
            tests/unit/compute/test_talwani.py \
            tests/unit/compute/test_batch.py \
            tests/unit/cli/test_commands.py \
            tests/unit/io/test_loaders.py \
            tests/unit/io/test_writers.py \
            tests/unit/io/test_csv_loader.py \
            tests/unit/viz/test_plotter.py \
            tests/integration/test_workflow.py; do
  sed -i '' 's/ForwardModel/MagneticModel/g' "$file"
done
```

**Step 2: Run full test suite**

```bash
make test
```

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: update all test files for MagneticModel"
```

---

## Task 13: Update benchmarks/bench_calculator.py

**Files:**
- Modify: `benchmarks/bench_calculator.py` (3 occurrences)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' benchmarks/bench_calculator.py
```

**Step 2: Verify syntax**

```bash
python -m py_compile benchmarks/bench_calculator.py
```

Expected: OK

**Step 3: Commit**

```bash
git add benchmarks/bench_calculator.py
git commit -m "refactor: update bench_calculator.py for MagneticModel"
```

---

## Task 14: Update README.md

**Files:**
- Modify: `README.md` (3 code block occurrences)

**Step 1: Replace ForwardModel with MagneticModel in code blocks**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' README.md
```

**Step 2: Verify the file looks correct**

```bash
head -100 README.md | grep -i magnetic
```

Expected: Shows updated code examples with `MagneticModel`

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README.md for MagneticModel"
```

---

## Task 15: Update docs/examples.md

**Files:**
- Modify: `docs/examples.md` (4 occurrences)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' docs/examples.md
```

**Step 2: Commit**

```bash
git add docs/examples.md
git commit -m "docs: update examples.md for MagneticModel"
```

---

## Task 16: Update docs/user-guide/examples.md

**Files:**
- Modify: `docs/user-guide/examples.md` (7 occurrences)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' docs/user-guide/examples.md
```

**Step 2: Commit**

```bash
git add docs/user-guide/examples.md
git commit -m "docs: update user-guide/examples.md for MagneticModel"
```

---

## Task 17: Update docs/api-reference/index.md

**Files:**
- Modify: `docs/api-reference/index.md` (8 occurrences)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' docs/api-reference/index.md
```

**Step 2: Commit**

```bash
git add docs/api-reference/index.md
git commit -m "docs: update api-reference/index.md for MagneticModel"
```

---

## Task 18: Update docs/theory.md

**Files:**
- Modify: `docs/theory.md` (1 prose occurrence)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' docs/theory.md
```

**Step 2: Commit**

```bash
git add docs/theory.md
git commit -m "docs: update theory.md for MagneticModel"
```

---

## Task 19: Update docs/development/architecture.md

**Files:**
- Modify: `docs/development/architecture.md` (~15 pseudocode occurrences)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' docs/development/architecture.md
```

**Step 2: Commit**

```bash
git add docs/development/architecture.md
git commit -m "docs: update architecture.md for MagneticModel"
```

---

## Task 20: Update docs/development/contributing.md

**Files:**
- Modify: `docs/development/contributing.md` (6 occurrences)

**Step 1: Replace ForwardModel with MagneticModel**

```bash
sed -i '' 's/ForwardModel/MagneticModel/g' docs/development/contributing.md
```

**Step 2: Commit**

```bash
git add docs/development/contributing.md
git commit -m "docs: update contributing.md for MagneticModel"
```

---

## Task 21: Final verification

**Files:**
- (read-only verification, no changes)

**Step 1: Run full test suite**

```bash
cd /Users/joshuapoirier/conductor/workspaces/forward-model/yeosu
make test
```

Expected: All tests PASS

**Step 2: Run linting and type checking**

```bash
make lint
```

Expected: No errors

**Step 3: Verify no remaining ForwardModel references (outside archived plans)**

```bash
grep -r "ForwardModel" . --include="*.py" --include="*.md" \
  --exclude-dir=docs/plans 2>/dev/null | wc -l
```

Expected: 0 (zero matches)

**Step 4: List all commits in this branch**

```bash
git log --oneline main..HEAD
```

Expected: 21 small, focused commits

**Step 5: View final diff summary**

```bash
git diff --stat main..HEAD
```

Expected: Shows ~28 files changed, ~240 lines added/deleted

---

## Summary

All 21 tasks are purely mechanical renames with no logic changes. Commits are small and focused. The final verification ensures:
- All tests pass
- No type/lint errors
- Zero remaining `ForwardModel` references (except archived plan docs)
- Clean git history with focused commits
