# feat/ISSUE-52-taking-multigraph 브랜치 검증 정리 (2026-07-06)

브랜치: `feat/ISSUE-52-taking-multigraph` (base: `main`)
검증 범위: `5719b56`(Edge int id) ~ `b23fcc5`(robbins/ILS 멀티그래프) + 미커밋 `flow_score.py` import 정리 1건
관련 문서: `FLOW_SCORE_UNIFICATION.md`, `.plan/general/2026-07-02-unify-flow-score-single-source.md`,
`.plan/issues/2026-06-18-issue-52-multigraph-edge-int-id.md`

## 결론 요약

세 가지 목표(멀티그래프, APSP ranker 수정, flow 단일화)는 **모두 의도대로 구현되어 있다.**
전체 테스트 219개 중 218개 통과, 1개 실패는 이번 변경과 무관한 알려진 환경 flake
(`tests/solver/test_process_runner.py::test_process_runner_allows_nested_multiprocessing`,
`/mnt/c` WSL 9p 마운트에서만 간헐 발생 — `FLOW_SCORE_UNIFICATION.md` 검증 절에 기록된 것과 동일).

기능 버그는 발견하지 못했다. 다만 아래 "발견 사항"의 중복 구현·죽은 코드·의미론 미결 항목들이
있으며, 사용자 결정에 따라 **이 브랜치에서는 수정하지 않고 문서로만 기록**한다.

---

## 1. 멀티그래프 구현 — 확인됨

핵심 설계: `Edge.id` (자동 유니크 int, `domain/edge.py`)가 간선 정체성. 평행 간선은 같은
끝점이라도 독립 id. 모든 프로덕션 경로가 id 기반으로 왕복한다.

| 경로 | id 보존 방식 |
|---|---|
| QUBO 변수 | `e_{id}` 변수명 (`util/qubo_util.py:get_indicator_function`) — 평행 간선마다 독립 변수 |
| 해 복원 | `qubo/solution_processing.py:process_solution` — edge id → (u,v) dict |
| SA 솔버 | `sa_mr2s_solver.py:run` — 비트를 `variable_edges` 순서로 id별 복원 (8d0bed7) |
| Robbins/ILS | `util/graph_orient.py:robbins_orient` — nx.MultiGraph key=edge.id (dcf9405, b23fcc5) |
| nx 변환 | `util/planar_graph.py:domain_graph_to_networkx_multi` — key=id, 고립 정점 유지 |
| Base 솔버 | `base_edge_orientation_solver.py` — orienter 결과를 id로 검증·복원, 누락 시 ValueError |
| DnC merge | `dnc_mr2s_solver.py:merge_solutions` — 자식 solution 방향을 id로 키잉 |
| Solution | `Solution.edges: dict[int, tuple[int, int]]` (b0f2b68) |

평행 간선 특수 규칙:

- **Robbins**: 첫 copy가 DFS 방향, 나머지 copy는 교대로 반대 방향 (flow 상쇄).
  directed 간선 추가는 강연결을 깨지 않으므로 Robbins 보장 유지 (`graph_orient.py` docstring).
- **Face cluster boundary**: anchor-one-free-rest — 첫 copy만 directed anchor, 나머지는
  undirected 자유변수로 min(owning_macros) 한쪽에만 배치 (`face_cluster_partition.py:88-111`).
- **ILS n3**: 순환 방향과 일치하는 copy만 id별 독립 판정 후 뒤집음.

pair 기반 shim이 남은 곳 (의도된 잔존):

- `Graph.edge_by_pair` / `define_edge_direction` — **프로덕션 미사용**. 테스트·데모
  (`tests/run_sa_*_demo.py`, `tests/edge_orient/*`)에서만 호출. 평행 간선이면 첫 매치에
  방향을 박으므로 멀티그래프 테스트에 쓰면 collapse 위험 — 신규 테스트에서는 id 기반 사용 권장.
- `edge_orient/t_join.py` — DEPRECATED (DeprecationWarning 발행). `pair_to_edge`가 단순그래프
  전제라 평행 간선을 잃는 문제가 docstring에 명시되어 있음. 대체: ILS/Robbin.
  + 이요환 첨언 -> 분명히 t-join이 확신할 수 있는 부분을 확실하게 하고 다음 미확정을 solver 태우기였는데, 구현이 이상하게 되어있고, 사용도 안함. 제끼기.
- `domain_graph_to_networkx` (simple 뷰) — planar 임베딩 전용 shim으로 명시되어 있고,
  face cluster는 planar face 연산에만 사용하므로 문제없음.

## 2. APSP sum ranker 오류 수정 + 새 방식 — 확인됨

커밋 `1337eb0`. 수정 전 문제: hop-count 기반 단일 ranker. 수정 후:

- 단일 클래스 `ApspSumRanker(method=...)`, method ∈ {"stretch"(기본), "efficiency", "sum"}.
- 거리 규약 통일: **거리 = 1/weight** (`evaluator/distance_util.py:_inverse_distance`).
  weight가 클수록(좋은 길) 거리 짧음. weight ≤ 0이면 명시적 ValueError.
- 모든 method가 "낮을수록 좋음"으로 통일 — `select_best_sample`의 `min()` 계약 유지.
  efficiency도 `1 − E_dir/E_und`로 뒤집어 계약을 지킴.
- 같은 방향 평행 간선은 min distance만 유효 (최단거리 의미상 올바름,
  `build_directed_distance_graph`).
- 무방향 APSP 분모는 `UndirectedApspCache`로 그래프 인스턴스 단위 캐시.
- `Score.apsp_sum`의 의미가 **"거리 합" → "평균 stretch"로 변경됨** (필드명은 그대로).
- 테스트: `tests/evaluator/test_apsp_sum_ranker.py` 199줄 — method별 값, weight 스케일
  불변성, 비강연결 inf, 잘못된 method 거부 커버.

## 3. Flow 단일화 — 확인됨

커밋 `c639634`, `59e1177`. 상세는 `FLOW_SCORE_UNIFICATION.md` 참조. 검증 결과:

- 단일 코어 `mr2s_module/util/flow_score.py:flow_imbalance` — 순수 함수, (source, target,
  weight) 튜플 기반, 평행 간선 자연 합산.
- 위임 확인: `Evaluator.eval_flow`, `SAMR2SSolver._objective`,
  `edge_orient/iterated_local_search.py:evaluate_score` 모두 `flow_imbalance` 호출.
  독립 재구현 없음.
- QUBO poly ±w 수정: `FlowPolyGenerator._get_a_term` — 미방향 `2w·indicator − w`,
  방향 고정 `±w`. 동치성 테스트 `tests/qubo/test_flow_poly_equivalence.py`가
  poly 에너지 == flow_imbalance 를 전 비트 할당 전수로 고정.
- 미커밋 diff 1건: `flow_score.py`의 `typing.Iterable` → `collections.abc.Iterable`
  import 정리 (동작 동일, 커밋만 하면 됨).
- 규칙 (기존 문서 재확인): flow 점수 새 구현 금지, poly 수식 변경 시 동치성 테스트 동반 갱신.

참고 — 코어에 위임하지 **않는** 증분(incremental) flow 계산 두 곳은 전체 재계산이 아니라
델타 수식이므로 의도적 예외이나, 서로 중복이다 (아래 4-1-b).

---

## 4. 발견 사항 (4가지 기준 점검)

### 4-1. 중복 구현 (기준 1: 원래 있는 것을 무시하고 스스로 만든 경우)

**(a) mutable default 인스턴스 공유 + APSP 캐시 무한 성장**

`Evaluator()` 및 `QuboSolver.create_sa_solver(ranker=ApspSumRanker())`가 **기본 인자로
직접 생성**되어 있다:

- `qubo_mr2s_solver.py:27-28`, `sa_mr2s_solver.py:18`, `base_edge_orientation_solver.py:20`,
  `robbin_mr2s_solver.py`, `ils_mr2s_solver.py`

Python 기본 인자는 import 시 1회 평가되므로, 기본값으로 만든 모든 솔버 인스턴스가 **같은
Evaluator / 같은 QuboSolver / 같은 ApspSumRanker를 공유**한다. 이로 인해:

- `Evaluator._apsp_ranker._undirected_cache` (`UndirectedApspCache`)가 프로세스 수명 동안
  평가한 **모든 그래프의 APSP 결과 + 그래프 강참조**를 들고 있음 — eviction 없음,
  장기 실행 시 메모리 무한 성장.
- 공유 자체는 캐시 재사용 이득도 있으나, 암묵적 전역 상태라 테스트 격리·동시성에서 함정.

권장 수정(추후): 기본값을 `None`으로 받고 `__init__` 안에서 생성. 캐시는 상한 또는
`weakref` 기반으로.

**(b) 증분 flow 페널티 수식 중복**

동일한 수식 `(b_s−w)² + (b_t+w)² − b_s² − b_t²` (한 간선 방향 선택의 imbalance 델타)가
두 곳에 독립 구현:

- `sa_mr2s_solver.py:_greedy_flow_seed_bits` 내부 `direction_penalty`
- `dnc_mr2s_solver.py:_select_merge_direction` 내부 `flow_penalty` (+ `_apply_flow_balance`)

`util/flow_score.py`에 `flow_imbalance_delta(balance, direction, weight)` 같은 헬퍼로
합치면 코어와 같은 파일에서 관리 가능.

**(c) sample 디코딩 로직 2벌**

- `qubo/solution_processing.py:safe_lookup` ↔ `evaluator/evaluator.py:_safe_lookup` —
  완전 동일 (dimod SampleView의 ValueError 처리 포함).
- `evaluator.py:_sample_to_directed_edges` ↔ `solution_processing.py:process_solution` —
  같은 비트→방향 규약을 재구현. evaluator 쪽은 pair set이라 평행 간선이 collapse되지만,
  용도가 강연결 판정뿐이라 **결과는 정확함** (연결성은 평행 간선과 무관). 다만 규약이
  두 곳에 있어 향후 비트 의미 변경 시 갈라질 위험.

**(d) SA 내부 APSP ↔ 평가 APSP 의미 불일치** — 아래 4-3 미결 안건 참조.

### 4-2. 죽은/미사용 코드 (기준 2)

- `util/planar_graph.py:clone_edge` — export만 있고 (`util/__init__.py`) 호출처 0.
- `ApspSumRanker`의 `method="efficiency"` / `"sum"` — 프로덕션 전 경로가 기본값
  stretch만 사용. efficiency는 docstring이 "탐색 중간 해 평가용"이라 밝힌 정확히 그 용도
  (`select_best_sample`의 비강연결 sample 변별)에서 쓰이지 않음. 현재 stretch ranker는
  비강연결 sample을 전부 inf로 만들어 `min()`이 사실상 첫 sample을 고름.
  → **팀 회의 안건** (사용자 결정: 현상 유지, 회의에서 논의).
- 방어적 `is not None` 중 사실상 도달 불가:
  - `dnc_mr2s_solver.py:_apply_merged_directions`의 `if edge is not None` —
    `merge_solutions`가 `graph.edges`에서만 id를 뽑으므로 None 불가. 무해한 방어 코드.
  - `_solve_with_reused_embedding/context`의 `getattr(..., None)` 류는 프로토콜상
    다른 솔버 구현이 올 수 있어 죽은 코드 아님 (유지).

### 4-3. 논리적 미결점 (기준 3)

**(a) SA 솔버 내부 APSP는 hop-count(무가중치), 최종 채점은 1/weight stretch — [결정: 1/weight 전환, 별도 이슈]**

`sa_mr2s_solver.py:_build_apsp_and_disconnected_pair_count`는
`nx.single_source_shortest_path_length` (weight 무시, 간선 1개 = 거리 1)로 APSP를 계산한다.
1337eb0 이후 evaluator는 1/weight Dijkstra stretch를 채점 기준으로 쓰므로, **SA가
최적화하는 거리 척도와 채점 척도가 다르다.** weight 분산이 큰 그래프에서 SA가 채점 기준으로
열등한 해를 선호할 수 있다 (예: hop 1·w=1 경로를 hop 2·w=10 경로보다 선호하지만 채점은 반대).

논점 정리:

| 선택지 | 장점 | 단점 |
|---|---|---|
| SA도 1/weight Dijkstra로 전환 | 채점과 척도 정렬, SA↔QA 파리티 방향과 일치 | flip마다 Dijkstra (log factor 추가, 이미 무거운 경로) |
| hop 유지 (의도된 설계로 명시) | 저렴·robust, SA 목적함수는 자체 정규화 체계 | 채점과 계속 어긋남 |
| stretch까지 완전 정렬 | 완전 파리티 | 무방향 분모 캐시 필요, 가장 큰 변경 |

**결정 (2026-07-06, 사용자): 1/weight 전환으로 확정.** 이 브랜치에서는 수정하지 않고
별도 이슈로 진행한다. 범위: `_build_apsp_and_disconnected_pair_count`의 거리만
1/weight Dijkstra로 교체, 정규화 정책(`pair_scale`, `weight_scale`, treewidth)은 유지.

**(b) self-loop 입력 시 크래시 경로**

- QUBO: `get_indicator_function`이 i==j에서 ValueError → poly 생성 크래시.
- Robbin/ILS: `domain_graph_to_networkx_multi`가 self-loop skip → base solver가
  "was not oriented" ValueError.

사용자 결정: **도로망 도메인에 self-loop은 오지 않는다고 전제, 코드 수정 없음.**
전제가 깨지면 이 두 지점이 첫 크래시 위치라는 것만 기록해 둔다.

**(c) 오리엔터 간 실패 의미 불일치 (경미)**

- Robbin: 브릿지 존재/방향화 불가 시 `OrientedEdges()` 반환 → base solver가
  "Edge ... was not oriented" ValueError. 에러 메시지가 실제 원인(브릿지)을 안 알려줌.
- ILS: 브릿지 검사 없음 → 강연결 불가 그래프에서도 orientation을 반환하고 score만
  inf → base solver는 예외 없이 통과, Score에서만 드러남.

같은 "강연결 불가" 입력이 솔버에 따라 예외/조용한 inf로 갈라진다. 계약을 하나로 정하면 좋음.

**(d) 전량 directed 서브그래프의 Score 잡음 (외관상)**

`dnc_mr2s_solver.py:_solve_subgraph`의 directed-only 지름길이 빈 sample_set으로
평가를 돌려 `sample_score=inf`, `strong_connect_rate=0.0`이 나온다. 병합 점수 계산에서
`strong_connect_rate`를 곱하는 `score_merged_solution`에 이 0이 섞이면 전체 rate가
0으로 끌려갈 수 있다 — 실제 문제인지 사용 시나리오 확인 필요 (직접 재현은 안 함).

### 4-4. 분리하면 좋아 보이는 것 (기준 4)

- `dnc_mr2s_solver.py` (618줄): 로깅 래핑 + 파티션 전략 프록시 메서드
  (`_default_partition_strategy` 경유 7개) + merge 로직이 한 파일. merge 로직
  (`merge_solutions`/`_select_merge_direction`)은 순수 함수라 별도 모듈로 분리 가능.
- `face_cluster_partition.py:run` (Step 1~4 주석으로 구분된 80줄) — 컴포넌트 통합 /
  간선 분류 단계를 메서드로 쪼개면 anchor-one-free-rest 규칙 단위 테스트가 쉬워짐.
- `sa_mr2s_solver.py:_anneal_bits` — 온도 루프 + early-stop 판정이 얽혀 있음. 현재
  동작은 옳으나 early-stop 정책을 바꿀 일이 생기면 분리 권장.

---

## 5. 결정 기록

| 항목 | 결정 (2026-07-06, 사용자) |
|---|---|
| SA APSP 거리 규약 (4-3-a) | **1/weight 전환 확정, 별도 이슈로 진행**. 이 브랜치 코드 불변 |
| ranker method 기본값 (4-2) | 현상 유지, 팀 회의 안건 |
| self-loop (4-3-b) | 도메인상 없음 전제, 문서 기록만 |
| 중복/죽은 코드 4건 (4-1, 4-2) | 이 브랜치에서 수정 안 함, 문서 기록만 |

## 6. 검증 방법

- 전체 스위트: `pytest` — 218 passed / 1 failed (`test_process_runner_allows_nested_multiprocessing`,
  알려진 `/mnt/c` 환경 flake, 본 브랜치 변경과 무관).
- 정적 점검: id 왕복 경로 전수 추적 (§1 표), flow 위임 호출처 grep, `pair_key`/`define_edge_direction`
  잔존 호출처 grep (프로덕션 0건 확인), 기본 인자 인스턴스 공유 확인.
