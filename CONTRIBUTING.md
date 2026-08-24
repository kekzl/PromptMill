# Contributing to PromptMill

Thank you for your interest in contributing to PromptMill!

## Development Setup

Everything runs in containers. Docker is the only prerequisite.

```bash
git clone https://github.com/kekzl/PromptMill.git
cd PromptMill

# Lint, format check and the full test suite
docker compose -f docker-compose.yml -f docker-compose.build.yml \
  --profile test run --rm test

# Run the app from source (CPU)
docker compose -f docker-compose.yml -f docker-compose.build.yml \
  --profile cpu up -d --build
```

The test image carries ruff, mypy and pytest but deliberately no
`llama-cpp-python`: it is imported lazily inside the adapter and mocked in the
tests, so the suite runs in seconds without a compiler.

That laziness has a cost worth knowing: a broken inference runtime does not show
up in the test suite or in the health check, only on the first real generation.
CI imports `llama_cpp` in the built image explicitly for that reason.

<details>
<summary>Local toolchain instead</summary>

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run python -m promptmill
uv run pytest tests -q
uv run ruff check --fix . && uv run ruff format .
```

</details>

### Running Tests

```bash
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.build.yml --profile test run --rm test"

$COMPOSE pytest tests -q                       # everything
$COMPOSE pytest tests/unit -q                  # unit only
$COMPOSE pytest --cov=src/promptmill           # with coverage
$COMPOSE pytest tests/unit/presentation -q     # one layer
```

### Code Quality

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting; both are
enforced in CI.

```bash
$COMPOSE ruff check .
$COMPOSE ruff check --fix .
$COMPOSE ruff format .
```

## Architecture

PromptMill uses hexagonal architecture (ports and adapters) with DDD. See
[CLAUDE.md](CLAUDE.md) for the details that matter before changing behaviour.

```
src/promptmill/
├── domain/           # entities, value objects, ports - NO external imports
├── application/      # use cases + services
├── infrastructure/   # adapters, config, roles data
└── presentation/     # Gradio UI, REST API, examples, history, theme
```

The rule that keeps the structure honest: **the domain layer imports nothing
external.** New infrastructure goes behind a port.

Tests mirror the layers: `tests/unit/{domain,application,infrastructure,presentation}`
plus `tests/integration`.

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
- Dataclasses are `frozen=True, slots=True` unless there is a reason otherwise
- Keep functions focused and reasonably sized

### Adding a New Prompt Target

1. Add an entry to `ROLES_DATA` in
   `src/promptmill/infrastructure/persistence/roles_data.py`, inside the section
   for its category
2. Give it `category`, `description` and `system_prompt`. Follow the structure of
   the neighbouring entries: prompt structure, best practices, and an output
   format that ends with "Output ONLY the prompt"
3. Update the counts in the module docstring and the section header comment
4. If the category has no starter examples yet, add six to
   `src/promptmill/presentation/examples.py`
5. Update the counts in `README.md`

Do **not** add a hardcoded count to a test. Tests derive counts from `ROLES_DATA`
so that adding a target cannot break unrelated assertions.

### Adding a New Model Tier

1. Add a `Model` to `MODEL_CONFIGS` in
   `src/promptmill/infrastructure/config/model_configs.py` and to
   `MODEL_KEYS_ORDERED`
2. Extend the `select_model_by_vram` match block
3. `vram_required` is weights **plus KV cache at the configured context length**,
   not weights alone
4. Set `chat_format` if the base architecture is not Llama 3

## Pull Request Guidelines

- Describe what your PR does and why
- Reference any related issues
- Ensure all tests pass
- Keep PRs focused on a single feature/fix
- Update documentation if needed

## Questions?

Open an issue or start a discussion if you have questions about contributing.
