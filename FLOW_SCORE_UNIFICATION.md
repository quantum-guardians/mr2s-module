# Flow Score 단일화 + FlowPolyGenerator weight 버그 수정 (2026-07-04)

브랜치: `feat/ISSUE-52-taking-multigraph`
커밋: `c639634`, `59e1177`
계획 문서: `.plan/general/2026-07-02-unify-flow-score-single-source.md`

## 배경 / 문제

flow 균형 점수 `Σ_v (Σ_in w − Σ_out w)²`가 세 곳에 독립 구현되어 의미가 갈라져 있었다.

| 위치 | 상태 (수정 전) |
|---|---|
| `evaluator/evaluator.py` `eval_flow` | edge id 기반, ±w — 올바름 |
| `solver/sa_mr2s_solver.py` `_build_flow_score` | 수식은 올바르나 pair_key 기반이라 멀티그래프 평행 간선 collapse |
| `qubo/flow_poly_generator.py` | **w≠1에서 버그** |

QUBO poly 버그 상세:

- 미방향 간선 기여가 `2w·[나감] − 1` → 나감 `+(2w−1)`, 들어옴 `−1`. 올바른 값은 `±w`.
- 방향 고정 간선은 weight를 무시하고 `±1`.
- 결과: w≠1이면 무거운 간선이 "균형 페널티"가 아니라 "방향 편향"으로 작동해 SA 솔버와 QUBO 계열 솔버의 에너지 의미가 달라짐 (SA↔QA 파리티 저해).

## 변경 내용

### 1. `c639634` — refactor(flow): 수치 코어 신설 + 위임

- **`mr2s_module/util/flow_score.py` 신설**: `flow_imbalance(Iterable[tuple[int, int, float]]) -> float`.
  `(source, target, weight)` 튜플 기반 순수 함수 — 그래프/간선 정체성 무관, 평행 간선은 튜플 여러 개로 자연 합산. `util/__init__.py`에 export.
- **`Evaluator.eval_flow`**: edge id → weight 매핑 후 코어 위임 (동작 불변).
- **`SAMR2SSolver`**:
  - `_build_flow_score` 삭제 → `_directed_weighted_edges(graph, variable_edges, state_bits)` helper가 (source, target, weight) 튜플 생성, `_objective`에서 `flow_imbalance` 호출.
  - `_greedy_flow_seed_bits`의 `pair_key` weight 매핑 제거 — directed 간선 직접 순회로 대체, 시그니처에서 `fixed_edges` 파라미터 삭제.
  - 부수 효과(의도됨): flow 채점의 pair-collapse 제거, 평행 간선이 독립 합산 (ISSUE-52 방향과 일치).

### 2. `59e1177` — fix(qubo): poly 기여를 +w/−w로

`FlowPolyGenerator._get_a_term`:

- 미방향 간선: 상수 `−1` → `−w`. 기여가 `2w·indicator − w = ±w`.
- 방향 고정 간선: `±1` → `±w`.

**동치성 테스트 신설** `tests/qubo/test_flow_poly_equivalence.py`:

- 케이스: 비가중 삼각형 / 가중치 혼합 삼각형 / 방향 고정 간선(w=4) 포함 그래프 / 평행 간선 멀티그래프 / 단일 간선 w=7 회귀(에너지 `2w²`).
- 각 케이스에서 미방향 간선 전 비트 할당 전수로 `poly.energy(sample) == flow_imbalance(대응 방향 튜플들)` 검증.
- 이 테스트가 접착제: 이후 누가 poly나 수치 코어 어느 쪽을 고쳐도 의미가 갈라지면 즉시 잡힌다. (지금까지 세 구현이 조용히 갈라진 원인이 이 테스트 부재였음.)

## 검증

- 전체 스위트 **211개 통과** — Linux fs 임시 worktree에서 변경 전(206) / 변경 후(211, 신규 5개 포함) 모두 green.
- w=1 경로는 `±1` 동일이라 기존 테스트 기대값 변화 없음. w≠1 QUBO 에너지 기대값을 가진 기존 테스트 없음 확인.
- 알려진 flake: `/mnt/c` (WSL 9p 마운트)에서 전체 스위트 실행 시 `tests/solver/test_process_runner.py::test_process_runner_allows_nested_multiprocessing`이 간헐 실패 (`ProcessExecutionError: Empty` — multiprocessing 큐 타임아웃). 깨끗한 HEAD에서도 재현되고 격리 실행·Linux fs에서는 통과 — **이번 변경과 무관한 환경 flake**.

## Non-goals (유지)

- n-hop 항 통합 — SA 목적함수에 대응물 없음 (SA는 APSP 직접 계산). 파리티 격차로 남김.
- SA 정규화 정책 (`weight_scale`, `flow_weight`, `pair_scale`) 변경 없음.
- 멀티그래프 pair-collapse 전면 수정 — Base solver 쪽은 ISSUE-52 본 작업 범위로 남음.

## 이후 규칙

- flow 점수 새 구현 금지 — `mr2s_module.util.flow_imbalance` 호출.
- 다항식 쪽 수식 변경 시 `tests/qubo/test_flow_poly_equivalence.py` 동반 갱신.
