# Contributing to slm-eval

Thank you for your interest in contributing! This project follows a standard open-source workflow.

## Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/yourusername/slm-agentic-eval-harness.git
cd slm-agentic-eval-harness
pip install -e .[dev,train,modal,serve,ocr,viz]
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_gate.py -v

# With coverage
pytest --cov=slm_eval tests/
```

## Code Style

```bash
# Lint with Ruff
ruff check .

# Auto-fix
ruff check . --fix

# Format
ruff format .
```

## Project Structure

```
slm_eval/           # Main package
├── backends/       # Cloud backends (Modal)
├── analysis/       # Gate, error, dataset analysis
├── agents/         # Advisory agent contracts (AC-11)
├── cli.py          # CLI entry point
├── pipeline.py     # Orchestrator (canonical pipeline)
├── gate.py         # Gate logic
├── scoring.py      # Scoring primitives
└── ...
tests/              # Test suite
scripts/            # Utility scripts (calibration, fixtures)
configs/            # Example configs (not committed)
data/               # Example datasets (not committed)
runs/               # Evaluation runs (not committed)
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make changes with tests
3. Run `pytest` and `ruff check .` — both must pass
4. Open a PR with a clear description of changes
5. CI will run tests automatically

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `chore:` — Maintenance
- `docs:` — Documentation
- `refactor:` — Code restructuring
- `test:` — Test additions/changes

Example: `feat: add T4 GPU support for Modal training`

## Architecture

The Python `Orchestrator` in `pipeline.py` is the canonical evaluation pipeline. Rust components were removed as experimental.

Key components:
- **Gate** (`gate.py`) — F1 + 95% CI promotion logic
- **Scoring** (`scoring.py`) — Micro-F1, bootstrap CI, validators
- **Pipeline** (`pipeline.py`) — Stages 1-7 with checkpoints
- **Modal backend** (`backends/modal.py`) — Cloud GPU training/deployment
- **Agents** (`agents.py`) — Advisory lane (never scores/gates)

## Adding Tests

- Unit tests: `tests/test_<module>.py`
- Integration tests: `tests/integration/`
- Fixtures: `tests/fixtures/`
