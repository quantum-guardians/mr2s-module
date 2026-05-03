# Repository Guidelines

## Project Structure & Module Organization
`mr2s_module/` contains the library code. Keep domain models in `mr2s_module/domain/`, cycle extraction logic in `mr2s_module/cycle/`, QUBO generation and solving in `mr2s_module/qubo/`, orchestration in `mr2s_module/solver/`, evaluation code in `mr2s_module/evaluator/`, and shared helpers in `mr2s_module/util/`. Package metadata lives in `pyproject.toml`; built artifacts are written to `dist/` and should not be edited manually.

## Build, Test, and Development Commands
Use Python 3.11 or newer.

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -e .`: install the package in editable mode with runtime dependencies such as `dwave-ocean-sdk`.
- `python -m build`: build source and wheel distributions from `pyproject.toml`.
- `pytest`: run the test suite once tests are added.

If you add new tooling, document the exact command here rather than assuming contributors will infer it.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for modules and functions, `PascalCase` for classes, and explicit type hints on public APIs. Prefer small dataclasses and protocol-driven interfaces, matching files such as `domain/graph.py` and `protocols.py`. Keep imports absolute from `mr2s_module`. No formatter or linter is configured yet, so keep style changes narrow and consistent with surrounding code.

## Testing Guidelines
Add tests under a top-level `tests/` package that mirrors the module layout, for example `tests/solver/test_qubo_mr2s_solver.py`. Use `pytest` test names like `test_run_returns_score_for_connected_graph`. Cover new solver, evaluator, and graph transformation paths with focused unit tests; include small deterministic graph fixtures instead of large generated inputs.

## Commit & Pull Request Guidelines
Create feature branches with the pattern `feat/ISSUE-{ISSUE_NUM}`, for example `feat/ISSUE-11`. Recent history favors short imperative commits, usually with a Conventional Commit prefix such as `feat:` or a direct maintenance message. Keep commit subjects concise and specific, for example `feat: add APSP evaluator`. Pull requests should describe the behavioral change, note any graph or QUBO assumptions, link the related issue, and include sample input/output when algorithm results change.

## Security & Configuration Tips
Do not hardcode credentials or solver endpoints. Treat third-party solver configuration as environment-specific, and keep `.pypirc` or publishing settings out of feature changes unless packaging work requires them.
