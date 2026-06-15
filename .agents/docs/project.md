# Project Context

Fill this document during project initialization. Agents must verify commands against repository configuration before running them.

## Overview

- Product: `mr2s-module`
- Primary users: Python developers solving MR2S edge-orientation problems
- Core domain: planar graph preprocessing, QUBO solving, edge orientation, and evaluation
- Runtime environment: Python 3.11 or newer

## Architecture

- Entry points: public package exports from `mr2s_module/__init__.py`
- Main modules: `domain`, `cycle`, `edge_orient`, `qubo`, `solver`, `evaluator`, `util`
- Dependency direction: domain and protocols support algorithm modules; solver orchestrates them
- External systems: optional D-Wave quantum annealing service
- Persistent data: none in the library

## Commands

| Purpose | Command |
|---|---|
| Install dependencies | `.venv/bin/python -m pip install -e ".[test]"` |
| Run locally | Not applicable; import as a Python library |
| Unit tests | `.venv/bin/python -m pytest -m "not slow"` |
| Integration tests | `.venv/bin/python -m pytest -m slow` |
| Build | `.venv/bin/python -m build` |

## Constraints

- Compatibility requirements: Python 3.11 or newer
- Performance constraints: solver cost scales with graph and QUBO size
- Security or privacy requirements: keep solver credentials and endpoints out of source

## Ownership

- Sensitive modules: `mr2s_module/qubo/`, `mr2s_module/solver/`
- Changes requiring explicit review: graph assumptions, QUBO formulation, solver behavior, and package release configuration
