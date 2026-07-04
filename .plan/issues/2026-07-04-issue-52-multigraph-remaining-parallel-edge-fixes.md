# 2026-07-04 — 멀티그래프 잔여 작업: SA 재구성 붕괴 수정 + base/orienter 체인

- Date: 2026-07-04
- GitHub Issue: #52
- Status: Draft

## Goal

멀티그래프(병렬엣지 + 임의 weight) 지원을 전 경로에서 완성한다. 2026-07-04 3-에이전트 감사 + 실증 프로브 결과, QUBO 경로와 weight/flow score 통합은 완료 확인. 남은 것은 SA solver 최종 solution 재구성과 base/orienter 체인의 pair-collapse 제거.

## Non-goals

- t_join 수정 (deprecated 확정, DeprecationWarning 가드됨 — 도로 도메인이 duplication fix 거부)
- planar face/cycle 로직의 frozenset 키 제거 (planar dual 도메인은 본질적으로 simple-graph, 의도된 shim)
- flow score 재설계 (util/flow_score.flow_imbalance 단일 코어 확정, 변경 금지)

## Context / Constraints

2026-07-04 검증 결과 (브랜치 feat/ISSUE-52-taking-multigraph, uncommitted 작업트리 기준):

**완료 확인된 것:**
- QUBO 경로 end-to-end 멀티그래프 안전: `e_{edge_id}` 변수, `qubo/solution_processing.py::process_solution` id 키, poly ≡ `flow_imbalance` (weight 다른 병렬엣지 fixture 16개 할당 전수, 오차 0). 병렬 쌍 독립 방향 배정, flow 0 최적해 도달.
- w≠1 QUBO poly 버그 수정: `flow_poly_generator.py:16` (고정방향 ±w), `:23` (undirected 상수 −w).
- SA 목적함수 수정: 옛 `_build_flow_score`(pair_key last-wins) 삭제, `_directed_weighted_edges`(sa L99-115)가 엣지별 (u,v,w) 튜플 방출 → `flow_imbalance` 호출. anneal raw bits는 정답 도달 확인.
- `tests/qubo/test_flow_poly_equivalence.py`: weight 혼합/고정방향/병렬엣지 전수 검사 5개 통과.
- pytest 210 passed / 1 failed (`test_process_runner_allows_nested_multiprocessing` — WSL 세마포어 환경 문제, 본 작업 무관).

**남은 버그 (우선순위순):**
1. **`solver/sa_mr2s_solver.py:342-349` (치명, 실증 확인)** — 최종 `Solution.edges` 재구성이 `(u,v) in directed_edges` set 멤버십으로 방향 복원. 반평행 병렬 쌍이면 `(1,2)`, `(2,1)` 둘 다 set에 있어 두 id 모두 첫 분기 `(u,v)`로 붕괴. 관측 증상: raw bits는 flow 0 최적인데 Score는 `flow_score=72`, `apsp_sum=inf`, `strong_connect_rate=1.0`(raw bits 읽음) — 자기모순 Score.
2. **`solver/base_edge_orientation_solver.py:23, 37-44`** — `directed_edges = {edge.vertices}` pair-set 생성 후 끝점 멤버십으로 재구성. Robbin/ILS solver 전부 영향. L26-32 검증도 pair 단위라 병렬 누락 조용히 통과.
3. **orienter 체인 상류** — `util/graph_orient.py:18` robbins_orient + `edge_orient/iterated_local_search.py:12` 둘 다 `EdgeMap = dict[frozenset, Edge]`. 근원: `util/planar_graph.py::domain_graph_to_networkx:25-27`이 nx.Graph 변환에서 중복엣지 버림(최소 weight만 유지).
4. **경미** — `edge_orient/iterated_local_search.py:154-165 evaluate_score`: flow-imbalance 공식 4번째 독립 복사본. 현재 수치 일치하나 동치 테스트 미커버 → `flow_imbalance` 호출로 교체하거나 동치 테스트에 편입.

제약: 새 flow score 구현 금지 (단일 코어 원칙). 병렬 boundary 엣지 cycle 규칙은 anchor-one-free-rest 유지.

## Approach (Checklist)
- [ ] **Step 0: Recon** — 작업트리 uncommitted 변경(flow score 통합분) 먼저 커밋 여부 확인. `sa_mr2s_solver.py`의 `variable_edges`/`best_bits` 구조 재확인.
- [ ] **Step 1: SA 재구성 수정 (버그 1)** — L342-349 끝점 재매칭 삭제, `variable_edges[i].id → _build_direction(edge, best_bits[i])` 직접 매핑. fixed(directed) 엣지는 기존대로 `edge.vertices` 사용.
- [ ] **Step 2: Base solver 수정 (버그 2)** — orienter 반환을 pair-set 대신 edge.id 키 구조로 받도록 인터페이스 검토. 상류(버그 3)와 함께 결정: nx.MultiGraph 전환 또는 orienter 출력에 edge id 관통.
- [ ] **Step 3: evaluate_score 통합 (버그 4)** — `flow_imbalance` 호출로 교체.
- [ ] **Step 4: Tests** — weight 다른 병렬 쌍 fixture로 SA solver end-to-end 테스트 추가 (`Solution.edges`에 두 id가 서로 다른 방향으로 존재 + `flow_score=0` 검증). base/Robbin/ILS 경로도 동일 fixture 적용 또는 simple-graph 전제 명시적 guard 추가.

## Validation
- **Commands to run:** `pytest`, 특히 `pytest tests/qubo/test_flow_poly_equivalence.py tests/solver/`
- **Expected output:** 전체 통과 (process_runner WSL 실패 제외). 병렬엣지 SA 테스트: `Solution.edges[id_a] == (1,2)`, `Solution.edges[id_b] == (2,1)`, `flow_score == 0.0`, `apsp_sum` 유한, Score 자기모순 없음.

## Risks & Rollback
- **Risks:** SA 재구성 변경이 fixed-directed 엣지 처리와 얽힐 수 있음(변수 엣지 vs 고정 엣지 분리 주의). Base/orienter 체인은 nx 변환 근원까지 건드리면 planar 경로 회귀 위험 — planar 소비자는 shim 유지, orienter 소비자만 MultiGraph화.
- **Rollback steps:** `git revert` 커밋 단위. flow score 코어는 미변경이라 회귀 범위는 solver 재구성 로직에 국한.

## Open Questions
- Base/orienter 체인(버그 2-3)을 이번 issue 52 범위에 포함할지, 별도 이슈로 분리할지. (SA 수정만으로도 QUBO+SA 두 주력 경로는 멀티그래프 완성)
- Robbin/ILS가 구조적으로 simple-graph 전제라면 수정 대신 병렬엣지 입력 시 명시적 에러/경고로 가드하는 선택지도 있음.
