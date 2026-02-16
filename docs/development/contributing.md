# Contributing

Thank you for your interest in contributing to Forward Model! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Clone and Install

```bash
# Clone repository
git clone https://github.com/Lubricate-AI/forward-model.git
cd forward-model

# Install dependencies
make install

# Verify installation
python -m forward_model --help
```

## Repository Conventions

### Branch Naming

Follow the pattern: `<issue-number>-brief-description`

```bash
# Good examples
git checkout -b 15-add-remanence-support
git checkout -b 23-fix-csv-parser-bug
git checkout -b 8-improve-documentation

# Bad examples
git checkout -b fix-bug        # No issue number
git checkout -b issue-15       # Not descriptive
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <description>

[optional body]

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring (no behavior change)
- `test`: Add or update tests
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples:**

```bash
# Good commit messages
feat: add support for remanent magnetization
fix: handle empty observation arrays in CSV loader
docs: update installation instructions
test: add integration tests for multi-body models

# Bad commit messages
fixed stuff
update
WIP
```

### Pull Request Process

1. **Create an issue** describing the problem or feature
2. **Create a branch** from `main` following naming conventions
3. **Make changes** with tests and documentation
4. **Run quality checks**:
   ```bash
   make lint
   make type-checking
   make test
   ```
5. **Create pull request** using the PR template
6. **Address review feedback**
7. **Merge** after approval

## Development Workflow

### Make Commands

The `Makefile` provides convenient development commands:

```bash
# Install/update dependencies
make install

# Run all linting
make lint

# Type checking with pyright
make type-checking

# Auto-format code
make format

# Run tests
make test

# Run tests with coverage
uv run pytest --cov=forward_model --cov-report=html
```

### Code Quality Standards

#### Linting

All code must pass `ruff` linting:

```bash
# Check for issues
make lint

# Auto-fix issues
make format
```

#### Type Hints

Use type hints for public APIs:

```python
from typing import List, Tuple
import numpy as np
import numpy.typing as npt

def calculate_anomaly(
    model: ForwardModel,
) -> npt.NDArray[np.float64]:
    """Calculate magnetic anomaly."""
    ...
```

#### Testing

- Write tests for new features
- Maintain >90% code coverage
- Include unit and integration tests

```python
# tests/unit/compute/test_talwani.py
import pytest
from forward_model import calculate_anomaly
from forward_model.models import ForwardModel

def test_calculate_anomaly_simple_dyke():
    """Test anomaly calculation for simple vertical dyke."""
    model = ForwardModel(...)
    anomaly = calculate_anomaly(model)
    assert len(anomaly) == len(model.observation_x)
    assert anomaly.max() > 0
```

#### Documentation

- Add docstrings to public functions and classes
- Use Google-style docstring format
- Update user guides for new features

```python
def calculate_anomaly(model: ForwardModel) -> npt.NDArray[np.float64]:
    """Calculate magnetic anomaly using Talwani algorithm.

    Args:
        model: Forward model with bodies, field, and observation points.

    Returns:
        Array of magnetic anomaly values in nanoTesla (nT).

    Raises:
        ValueError: If model has no bodies or observation points.

    Example:
        >>> model = load_model("model.json")
        >>> anomaly = calculate_anomaly(model)
        >>> print(f"Peak anomaly: {anomaly.max():.1f} nT")
    """
    ...
```

## Testing

### Running Tests

```bash
# All tests
make test

# Specific test file
uv run pytest tests/unit/io/test_loaders.py -v

# Specific test
uv run pytest tests/unit/io/test_loaders.py::test_load_json -v

# With coverage
uv run pytest --cov=forward_model --cov-report=html
open htmlcov/index.html
```

### Test Organization

```
tests/
├── unit/              # Unit tests
│   ├── compute/       # Algorithm tests
│   ├── io/            # I/O tests
│   ├── models/        # Model validation tests
│   └── viz/           # Visualization tests
└── integration/       # Integration tests
    └── test_workflows.py
```

### Writing Tests

Use pytest fixtures for common setup:

```python
import pytest
from forward_model.models import ForwardModel, GeologicBody, MagneticField

@pytest.fixture
def simple_dyke_model():
    """Fixture for simple vertical dyke model."""
    body = GeologicBody(
        name="Test Dyke",
        susceptibility=0.08,
        vertices=[[0, 50], [10, 50], [10, 200], [0, 200]]
    )
    field = MagneticField(
        intensity=50000.0,
        inclination=60.0,
        declination=0.0
    )
    return ForwardModel(
        bodies=[body],
        field=field,
        observation_x=[-100, 0, 100],
        observation_z=0.0
    )

def test_anomaly_calculation(simple_dyke_model):
    """Test basic anomaly calculation."""
    from forward_model import calculate_anomaly

    anomaly = calculate_anomaly(simple_dyke_model)

    assert len(anomaly) == 3
    assert anomaly[1] > anomaly[0]  # Peak over body
```

## Package Management

### Using uv

Forward Model uses [uv](https://github.com/astral-sh/uv) for dependency management.

**Do NOT use `pip install` directly!**

#### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Add with version constraint
uv add "package-name>=1.0.0"
```

#### Updating Dependencies

```bash
# Update all dependencies
make install

# Update specific package
uv add package-name@latest
```

#### Lock File

The `uv.lock` file ensures reproducible installs. Always commit it:

```bash
git add uv.lock
git commit -m "chore: update dependencies"
```

## Documentation

### Building Documentation Locally

```bash
# Serve documentation with live reload
make docs-serve

# Build documentation
make docs-build

# View in browser
open site/index.html
```

### Documentation Structure

```
docs/
├── index.md                 # Home page
├── getting-started.md       # Installation and quickstart
├── user-guide/             # User documentation
│   ├── cli.md
│   ├── visualization.md
│   └── examples.md
├── api-reference/          # Auto-generated API docs
│   └── index.md
└── development/            # Developer docs
    ├── contributing.md
    └── architecture.md
```

### Writing Documentation

- Use Markdown with MkDocs extensions
- Include code examples
- Add cross-references
- Test all code snippets

**Admonitions:**

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a helpful tip.
```

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG** with release notes
3. **Create release branch**: `release/v0.2.0`
4. **Run full test suite**
5. **Create GitHub release** with tag
6. **Publish to PyPI** (maintainers only)

## Getting Help

- 💬 **Questions**: Open a [GitHub Discussion](https://github.com/Lubricate-AI/forward-model/discussions)
- 🐛 **Bugs**: Create an [Issue](https://github.com/Lubricate-AI/forward-model/issues)
- 📧 **Contact**: Reach out to maintainers

## Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
