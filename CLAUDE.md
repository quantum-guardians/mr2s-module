# Repository Guidelines

## Agent Instruction Scope
This file is the repository-level source of truth. Repository-specific rules
below take precedence over defaults in `.claude/docs/`, `.claude/skills/`, and
`.claude/agents/`.

Before non-trivial work, read `.claude/docs/project.md` and follow
`.claude/docs/workflow.md` and `.claude/docs/testing.md`. For tracked Git work,
also use the issue, branch, commit, and pull-request documents in
`.claude/docs/`. Use project-local skills only when their trigger applies.

## Project Structure & Module Organization
`mr2s_module/` contains the library code. Keep domain models in `mr2s_module/domain/`, cycle extraction logic in `mr2s_module/cycle/`, QUBO generation and solving in `mr2s_module/qubo/`, orchestration in `mr2s_module/solver/`, evaluation code in `mr2s_module/evaluator/`, and shared helpers in `mr2s_module/util/`. Package metadata lives in `pyproject.toml`; built artifacts are written to `dist/` and should not be edited manually.

## Build, Test, and Development Commands
Use Python 3.11 or newer.

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -e ".[test]"`: install the package in editable mode with runtime and test dependencies.
- `ruff check mr2s_module tests` and `ruff format --check mr2s_module tests`: lint and format check (ruff 0.16.0).
- `pyright mr2s_module tests`: type check (pyright 1.1.411, the version Pylance matches).
- `pytest -m "not slow"`: run the fast suite; `pytest -m slow` runs the full QUBO pipeline tests.
- `python -m build`: build source and wheel distributions from `pyproject.toml`.

`.github/workflows/ci.yml` runs lint, format check, pyright, and the fast test
suite on pushes to `main` and on pull requests; any failure fails the build, so
run all four locally first. Version numbers come from tags through `hatch-vcs`,
which is why CI checks out with full history.

If you add new tooling, document the exact command here rather than assuming contributors will infer it.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for modules and functions, `PascalCase` for classes, and explicit type hints on public APIs. Prefer small dataclasses and protocol-driven interfaces, matching files such as `domain/graph.py` and `protocols.py`. Keep imports absolute from `mr2s_module`. ruff owns formatting and import order (line length 88, target `py311`, rule set in `pyproject.toml`); run `ruff format` rather than hand-aligning, and fix findings instead of adding `# noqa`. `B008` is ignored on purpose so solver and evaluator defaults can be injected as argument defaults.

## Testing Guidelines
Tests live in the top-level `tests/` package, which mirrors the module layout (`tests/cycle/`, `tests/domain/`, `tests/edge_orient/`, `tests/evaluator/`, `tests/qubo/`, `tests/reduction/`); add new files there, for example `tests/solver/test_qubo_mr2s_solver.py`. Mark long-running full-pipeline tests with `@pytest.mark.slow` so `pytest -m "not slow"` stays fast. Use `pytest` test names like `test_run_returns_score_for_connected_graph`. Cover new solver, evaluator, and graph transformation paths with focused unit tests; include small deterministic graph fixtures instead of large generated inputs.

## Commit & Pull Request Guidelines
Create feature branches with the pattern `feat/ISSUE-{ISSUE_NUM}`, for example `feat/ISSUE-11`. Recent history favors short imperative commits, usually with a Conventional Commit prefix such as `feat:` or a direct maintenance message. Keep commit subjects concise and specific, for example `feat: add APSP evaluator`. Pull requests should describe the behavioral change, note any graph or QUBO assumptions, link the related issue, and include sample input/output when algorithm results change.

## Release Guidelines
Use `.claude/assets/release-template.md` for GitHub release notes. Before writing a release, review the full changelog range from the previous tag to the new tag and make the summary and change list reflect the whole release, not only the most recent pull request. Fill in the tag, summary, change list, validation result, related pull request, and full changelog link before publishing a release. Keep older releases on the same section structure when practical so release history stays consistent. For how to word the notes and how to edit an already-published release without destroying generated blocks, follow `.claude/docs/release.md`; the template and the rules above take precedence where they differ.

## Security & Configuration Tips
Do not hardcode credentials or solver endpoints. Treat third-party solver configuration as environment-specific, and keep `.pypirc` or publishing settings out of feature changes unless packaging work requires them.
