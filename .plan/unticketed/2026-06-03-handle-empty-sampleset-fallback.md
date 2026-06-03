# 2026-06-03 — Handle empty SampleSet fallback for QUBO solver

- Date: 2026-06-03
- Ticket: None
- Status: Implemented

## Goal

Fix issue #47 so QUBO solving does not crash when the sampler returns an empty `SampleSet`, including the case where objective cancellation produces a zero-variable BQM while the graph still has undirected edges.

## Non-goals

- Do not add zero-coefficient variables to the BQM just to represent indifferent decisions.
- Do not make indifferent edge direction random by default.
- Do not change QUBO objective math in `FlowPolyGenerator` or `NHopPolyGenerator`.

## Context / Constraints

- `process_solution({}, edges)` already gives deterministic default directions for missing variables by treating missing bits as `0`.
- Current crash path is `select_best_sample(...)` calling `min()` over an empty `sample_set.samples()` iterator.
- Zero-variable BQMs are valid when all variable coefficients cancel; they should skip sampler work and use deterministic fallback output.
- Empty sampler responses should be handled even when the BQM has variables, because remote samplers can fail or return no reads.

## Approach (Checklist)
- [x] **Step 0: Recon** (Inspect existing code, locate files)
  - Confirm current local changes in `mr2s_module/qubo/qubo_solver.py` and `tests/qubo/test_qubo_solver.py`.
  - Inspect `mr2s_module/qubo/solution_processing.py` for defensive fallback placement.
- [x] **Step 1: Implementation** (Code changes, file paths)
  - Add `QuboSolver` helper for deterministic fallback solution from `{}` sample.
  - Use fallback before sampling when `len(qubo.variables) == 0`.
  - Use fallback after sampling when `len(sample_set) == 0`.
  - Add defensive empty-sample guard in `select_best_sample`.
- [x] **Step 2: Tests** (Unit tests, manual verification steps)
  - Add tests for zero-variable BQM with undirected edge.
  - Add tests for empty sampler response with non-empty BQM.
  - Add test for `select_best_sample` empty `SampleSet` fallback.
- [x] **Step 3: Rollout / Rollback** (Feature flags, migration steps)
  - No feature flag or migration needed.
  - Rollback via reverting PR commit if fallback behavior is not desired.

## Validation
- **Commands to run:**
  - `python3 -m pytest tests/qubo/test_qubo_solver.py -m "not slow"` (blocked: pytest is not installed)
  - `.venv/bin/python` direct smoke test for zero-variable and empty-sample fallback
- **Expected output:**
  - Direct smoke test passed.
  - Empty `SampleSet` no longer raises `min() iterable argument is empty`.
  - Zero-variable BQM returns deterministic default directed edges.

## Risks & Rollback
- **Risks:**
  - Empty sampler responses may hide upstream sampler failures if callers expect hard failure.
  - Synthetic fallback sample energy may not reflect real sampled energy when BQM had variables but no samples.
- **Rollback steps:** revert the PR commit with `git revert`.

## Open Questions
- Resolved: store `fallback_reason` in `sample_set.info`.
- Resolved: fallback synthetic sample uses BQM offset as energy.
