# 2026-06-02 — QuboSolveContext로 BQM/Embedding/Topology 일관성 보장

- Date: 2026-06-02
- Ticket: None
- Status: Done

## Goal

DnC embedding-aware solve에서 `Graph`, `BQM`, `EmbeddingEstimate`, 실제 HW `target_graph`를 하나의 실행 컨텍스트로 묶어 전달한다. reused embedding이 다른 BQM이나 다른 D-Wave topology에 사용되는 구조적 원인을 제거하고, 사용 직전 validation/fallback으로 안전성을 보강한다.

## Non-goals

- `Graph` 도메인 객체 안에 BQM, embedding, sampler topology, cache invalidation 로직을 넣지 않는다.
- live QPU 호출이 필요한 테스트는 추가하지 않는다.
- 전체 solver API를 대규모로 교체하지 않고, 기존 public path는 가능한 유지한다.

## Context / Constraints

현재 embedding estimate는 partition strategy에서 `DnCMr2sSolver.target_graph` 기준으로 생성되지만, 실제 reused solve는 `QuboSolver.fixed_embedding_child_sampler`의 topology 기준으로 `FixedEmbeddingComposite`가 검증한다. 기본 `target_graph=dnx.pegasus_graph(16)`은 실제 HW topology가 아니므로 disabled qubit/coupler 차이로 disconnected chain이 발생할 수 있다.

## Approach (Checklist)

- [x] **Step 0: Recon** (Inspect existing code, locate files)
- [x] **Step 1: Implementation** (Code changes, file paths)
- [x] **Step 2: Tests** (Unit tests, manual verification steps)
- [x] **Step 3: Rollout / Rollback** (Feature flags, migration steps)

## Validation

- **Commands to run:** `python3 -m pytest tests/qubo/test_qubo_solver.py tests/solver/test_qubo_mr2s_solver.py tests/solver/test_dnc_mr2s_solver.py`
- **Expected output:** reused embedding invalid 케이스는 fallback하고, valid reused embedding 케이스는 기존처럼 fixed embedding solve를 사용한다.
- **Observed output:** `136 passed, 19 deselected in 19.44s` with `-m "not slow"`.

## Risks & Rollback

- **Risks:** `build_bqm()`이 graph를 mutate하는 현재 동작 때문에 context 생성 시점이 중요하다. multiprocessing subgraph solve에서 context 객체가 pickle 가능해야 한다.
- **Rollback steps:** `QuboSolveContext` 전달 변경을 revert하고, 최소 fallback fix만 유지한다.

## Open Questions

- 없음.
