# Contributing to PromptMill

Thank you for your interest in contributing to PromptMill!

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/kekzl/PromptMill.git
cd PromptMill

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run the application
python app.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_app.py::TestGPUDetection
```

### Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

## Making Changes

### Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters

### Code Style

- Follow the existing code style
- Add type hints to new functions
- Include docstrings for public functions (Google style)
- Keep functions focused and reasonably sized

### Adding New Roles

To add a new AI target role:

1. Add the role definition to the `ROLES` dictionary in `app.py`
2. Include a category prefix in the key (e.g., `"[Video] New Model"`)
3. Provide a comprehensive `system_prompt` following existing patterns
4. Update the counts in `README.md` and `CLAUDE.md`

## Pull Request Guidelines

- Describe what your PR does and why
- Reference any related issues
- Ensure all tests pass
- Keep PRs focused on a single feature/fix
- Update documentation if needed

## Questions?

Open an issue or start a discussion if you have questions about contributing.
