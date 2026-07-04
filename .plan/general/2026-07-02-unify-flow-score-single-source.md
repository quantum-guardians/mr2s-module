# 2026-07-02 — flow score 단일화 (수치 코어 + FlowPolyGenerator weight 버그 수정 + 동치성 테스트)

- Date: 2026-07-02
- GitHub Issue: None
- Status: Done (2026-07-04, on feat/ISSUE-52-taking-multigraph)

## Goal

flow 균형 점수 `Σ_v (Σ_in w − Σ_out w)²`의 구현을 단일 진실 공급원으로 통합한다.
현재 세 곳에 독립 구현되어 있고 의미가 갈라져 있다:

| 위치 | 구현 | 상태 |
|---|---|---|
| `mr2s_module/evaluator/evaluator.py:74-91` `eval_flow` | edge id 기반, ±w | 올바름 |
| `mr2s_module/solver/sa_mr2s_solver.py:97-117` `_build_flow_score` | pair_key 기반, ±w | 수식 올바름, 멀티그래프 pair-collapse |
| `mr2s_module/qubo/flow_poly_generator.py` + `util/qubo_util.py` | 다항식 | **w≠1에서 깨짐** |

QUBO poly 버그 상세:
- `flow_poly_generator.py:22-23` — 미방향 간선 기여가 `2w·[나감] − 1`. 나가면 `+(2w−1)`, 들어오면 `−1`. 올바른 값은 `±w` (상수를 `−1`이 아니라 `−w`로 빼야 함).
- `flow_poly_generator.py:16` — 방향 고정 간선이 weight 무시하고 `±1`.
- 결과: w≠1이면 무거운 간선이 "균형"이 아니라 "들어오는 방향 선호"로 편향. SA(SAMR2SSolver)와 QUBO(SA/QA 백엔드)의 에너지 의미가 달라짐.

배경 동기: SA 솔버와 QUBO QA 솔버가 비슷하게 동작하도록 만드는 파리티 작업의 1단계.

## Non-goals

- n-hop 항 통합 — SA 목적함수에 대응물 없음(SA는 APSP 직접 계산). 파리티 격차로 남기고 별도 논의.
- SA 정규화 정책(`weight_scale`, `flow_weight`, `pair_scale`) 변경 — 호출부 고유 정책으로 유지.
- 멀티그래프 pair-collapse 전면 수정 — ISSUE-52 브랜치 본 작업 범위. 단, SA flow 키잉이 코어 호출로 바뀌며 id 기반이 되는 부수 효과는 허용.

## Context / Constraints

- 다항식(BinaryPolynomial)은 기호식이라 수치 함수를 직접 호출 불가 → 완전 단일 함수는 불가능. 현실적 상한선 = **수치 코어 1개 + 다항식 1개 + 둘의 동치성 테스트**.
- 동치성 테스트가 진짜 접착제: 이후 누가 어느 쪽을 고쳐도 의미가 갈라지면 테스트가 잡는다. 지금까지 세 구현이 조용히 갈라진 원인이 이 테스트 부재.
- 현재 브랜치 `feat/ISSUE-52-taking-multigraph` 작업(멀티그래프 edge id) 과 순서 맞음 — SA 키잉 전환이 자연스럽게 겹침.

## Approach (Checklist)
- [x] **Step 0: Recon** — `util/` 배치 확인, BinaryPolynomial `.energy()` 평가 API 확인, w≠1 기대값 가진 기존 테스트 목록화 (w≠1 에너지 기대값 테스트 없음 확인)
- [x] **Step 1: 수치 코어** — `mr2s_module/util/flow_score.py`에 `flow_imbalance(directed: Iterable[tuple[int, int, float]]) -> float` (source, target, weight 튜플들 → 정점별 (Σin−Σout)² 합). 순수 함수, 그래프 객체 의존 없음.
  - `Evaluator.eval_flow` → id→weight 매핑 후 코어 위임
  - `SAMR2SSolver._build_flow_score` → 삭제, 코어 호출 (`_directed_weighted_edges` helper. `_greedy_flow_seed_bits`의 pair_key 매핑도 directed 간선 직접 순회로 대체)
- [x] **Step 2: FlowPolyGenerator 수정** — 같은 수식이 되도록:
  - `flow_poly_generator.py` 상수 `−1` → `−weight` (미방향 기여 ±w)
  - `flow_poly_generator.py` 방향 고정 간선 `±1` → `±weight`
- [x] **Step 3: 동치성 테스트** — `tests/qubo/test_flow_poly_equivalence.py`: 작은 그래프(가중치 혼합, 방향 고정 간선, 평행 간선 멀티그래프) × 전 비트 할당 전수에 대해 `poly.energy(sample) == flow_imbalance(해당 방향들)` 검증
- [x] **Step 4: 기존 테스트 정리** — 갱신 필요 없음 (w≠1 기대값 테스트 부재, 전체 스위트 그대로 통과)

## Validation
- **Commands to run:** `pytest tests/qubo/ tests/evaluator/ tests/solver/test_sa_mr2s_solver.py tests/solver/test_qubo_mr2s_solver.py`
- **Expected output:** 전체 통과. 동치성 테스트가 poly ↔ 수치 코어 일치 확인. w=1 경로 기존 결과 불변.

## Risks & Rollback
- **Risks:**
  - w≠1 그래프에서 QUBO 에너지 지형 변화 → DnC/QUBO 통합 테스트 기대값 변동 가능
  - `make_quadratic` penalty strength가 `max|coeff|` 기반(`qubo_util.py:17-20`)이라 계수 스케일 변화가 보조변수 페널티에 파급
  - SA flow 키잉 전환이 ISSUE-52 본 작업과 충돌할 수 있음 — 커밋 분리로 완화
- **Rollback steps:** 커밋 단위 `git revert` (코어 도입 / poly 수정 / 테스트를 별도 커밋으로)

## Open Questions
- FlowPolyGenerator 수정을 ISSUE-52 브랜치에 얹을지, 별도 이슈/브랜치로 뺄지 (에너지 의미 변화라 릴리스 노트 관점에서 분리가 깔끔할 수 있음)
- QA 하드웨어 경로에서 계수 스케일 변화가 체인 강도에 미치는 영향 확인 필요 여부
