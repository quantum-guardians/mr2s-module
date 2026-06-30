# 2026-06-18 — ISSUE-52 멀티그래프: Edge 정체성 frozenset → 자동 int id, QUBO id-native 전환

- Date: 2026-06-18
- GitHub Issue: #52
- Status: Draft (v2 — 평가 후 모델 정립)

## Goal
평행 간선(parallel edge)을 표현 가능한 멀티그래프 지원. 현재 `Edge.id = frozenset({v1,v2})`
가 정점 쌍을 정체성으로 써서 같은 두 정점 사이 간선 2개가 충돌(dict 덮어쓰기, QUBO 변수
충돌)한다. 간선마다 **자동 카운터 기반 유니크 int id** 를 부여하고, 정체성 경로를
id-native 로 전환한다.

핵심 두 덩어리:
1. **정체성 층**: `Edge.id`, `Graph.edges` 키, `to_key()` 를 int id 기반으로.
2. **QUBO 변수 생성**: 지금 변수명이 정점쌍(`e_{i}_{j}`) 기반이라 평행 간선이 같은 변수로
   뭉친다 → edge id 기반 변수명으로 전환(평행 간선마다 독립 이진변수 = 독립 방향 결정).

## 본질 모델 (v2 — 평가에서 정립)
`frozenset` 이 **두 일을 겸했다**:
1. **정체성** — 이게 어느 간선인가 (dict 키, QUBO 변수, solution 매핑)
2. **끝점 역lookup** — 정점쌍 `(u,v)` → 간선

**평행 간선은 (1)만 깨뜨린다** (간선 2개 → 키 1개). (2)는 멀쩡. 그래서 둘을 **분리**한다:

- **정체성 = int id, Edge 가 본래 들고 있는 속성.** Edge 객체를 손에 쥐면 `edge.id` 를 이미
  안다 → "모듈마다 id 반복 소지" 부담 없음. **bare `(u,v)` 튜플 대신 Edge / id 를 넘긴다.**
- **끝점 lookup = `pair_key()` + `Graph.edge_by_pair`.** nx 를 거쳐 Edge 정체성을 잃고
  `(u,v)` 만 남는 곳에서만 필요 = `t_join`, `face_cluster`, `sa_mr2s_solver`, `define_edge_direction`.

**"생성이 문제냐, graph 가 다 들고 있잖아" — 맞다.** graph 가 `{id: Edge}` 권위를 가진다.
코어에서 `(u,v)` 로 **새 Edge 를 맨땅 재구성하지 않는다.** 방향은 별도 상태(`Solution.edges:
dict[id, (u,v)]`)로 이미 분리돼 있으므로 코어 QUBO/solution/merge 경로엔 Edge 재생성이 없다.

기존 `Edge(u,v,w,True)` 재구성이 지금 안 깨진 이유: frozenset 정체성이라 끝점만 같으면 새
Edge 가 원본과 같은 id → dict 에서 원본을 덮어씀. int id 면 이 "공짜 정체성"이 사라지므로,
**기존 간선의 다른 버전(재배향/서브그래프 emission)이 필요한 곳은 원본 id 를 잇는다.**

## Non-goals
- `face_cluster_partition`, `t_join`, `iterated_local_search`, `sa_mr2s_solver` 의
  **멀티그래프 알고리즘화는 범위 밖**. 이들은 단순그래프(`domain_graph_to_networkx` 가 평행
  간선 병합, 또는 `(u,v)`-튜플 내부 표현) 전제로 계속 동작. 코어 변경으로 깨지는 최소 지점만
  pair 뷰 shim 으로 막는다.
- `define_edge_direction` 의 멀티그래프 완전 일반화는 하지 않는다. 외부 orienter 결과를
  **끝점으로 찾아 방향을 박는** 단순그래프 규칙으로만 처리.

## Context / Constraints
- 자동 카운터: `itertools.count` 를 `Edge` 클래스 변수로 두고 생성 시 유니크 id 자동 부여.
  생성부 폭발 없음. **생성 순서 결정적**이어야 재현성 유지.
- `iterated_local_search` 는 코어 `graph.edges` 를 안 건드리고 자체 nx + frozenset 맵
  (`util/graph_orient.py`, 로컬 `EdgeMap`) 으로만 돌아 **무관**(확인 완료). 코어 Edge.id 전환과
  분리됨.
- `Edge` 는 frozen dataclass 가 아닌 일반 클래스 → 방향 in-place 세팅(`set_direction`) 가능.

### 확정된 설계 결정
1. **`Solution.edges` 새 형** = `dict[int, tuple[int,int]]` (edge id → 방향 (u,v)).
   기존 `set[tuple[int,int]]` 은 평행 간선을 뭉개므로 폐기.
2. **재배향/emission 시 id 보존**: Edge API 로 처리(아래). 코어는 재구성 0개.
3. **`define_edge_direction`**: 외부 orienter(t_join/robbin/ILS)가 만든 독립 directed Edge 를
   받아, **끝점으로 원본을 찾아(`edge_by_pair`) 방향만 in-place 로 박는다**(`set_direction`).
   predefined 가 새 id 를 가져도 무관(끝점만 쓴다). 원본 Edge 의 id 유지.
4. **평행 간선 "가중치 선점" 규칙은 폐기.** 유니크 int id 면 평행 간선이 dict 키 충돌을
   안 하므로 선점 자체가 불필요해짐(평가에서 무의미 확인).

### Edge API 추가 (최소)
- `pair_key() -> frozenset[int]` : 끝점 lookup 키 (shim 소비자용).
- `to_key() -> str` : `f"e_{id}"` (QUBO 변수명과 단일 규칙).
- `oriented(tail, head) -> Edge` : directed=True, vertices=(tail,head), **id 복사한 새 Edge**.
  (face_cluster 서브그래프 emission 등 원본 비파괴 + DnC 정체성 왕복용)
- `set_direction(tail, head) -> None` : in-place 로 `vertices=(tail,head); directed=True`.
  (graph 에 방향 굽기: define_edge_direction, _apply_merged_directions)

## Approach (Checklist)

### Step 0: Recon — 완료
- [x] 정체성/룩업/QUBO 변수 생성 경로 추적, 파일별 영향 맵 작성.
- [x] 평가: 자동 int id 가 "재구성=정체성" 공짜를 깨는 본질 확인. 모델 정립(위).
- [x] 누락 발견: DnC merge frozenset 키 매칭, `sa_mr2s_solver` pair lookup, face_cluster
      서브그래프 emission id 왕복.

### Step 1: Implementation

**🔴 CORE — 정체성 층**
- [ ] `domain/edge.py`: `id: frozenset` → `id: int` (`itertools.count` 클래스 변수 자동 부여).
      `endpoints()`/`_endpoints` 유지. `pair_key()` 추가. `to_key()` → `f"e_{id}"`.
      `oriented(tail,head)`, `set_direction(tail,head)` 추가. `flip()` 도 id 복사 유지(ILS 안전).
- [ ] `domain/graph.py`: `edges: dict[frozenset, Edge]` → `dict[int, Edge]`.
      `__post_init__` `{edge.id: edge ...}` (자동 일치).
- [ ] `domain/graph.py`: `edge_by_pair(u,v) -> Edge | None`, `edges_by_pair(u,v) -> list[Edge]`
      헬퍼 추가(pair 뷰).
- [ ] `domain/graph.py:define_edge_direction`: predefined 마다 `edge_by_pair(*p.vertices)` 로
      원본 찾아 `set_direction(*p.vertices)`. 못 찾으면 skip. (replace-by-id 폐기)

**🔴 CORE — QUBO 변수 생성**
- [ ] `domain/adj_entry.py`: `AdjEntry` 에 `edge_id: int` 필드 추가(NamedTuple 끝에).
- [ ] `domain/graph.py:get_adjacency_dict`: `AdjEntry(..., edge.id)` 주입(3 군데).
- [ ] `util/qubo_util.py:get_indicator_function`: 시그니처 `(i, j, edge_id, weight)`.
      변수명 `f"e_{edge_id}"`. `i<j` 비교는 **부호 결정용으로만** 유지.
- [ ] `qubo/flow_poly_generator.py:21`: `edge.id` 전달.
- [ ] `qubo/n_hop_poly_generator.py:41`: `entry.edge_id` 전달.

**🔴 CORE — Solution 표현 + 처리**
- [ ] `domain/solution.py:12`: `edges: set[tuple]` → `dict[int, tuple[int,int]]`.
- [ ] `qubo/solution_processing.py:process_solution`: 반환형 `dict[int, tuple]`
      (`final[edge.id] = 방향`). `var_name = edge.to_key()` 유지(id 기반 자동 일치).
- [ ] `qubo/solution_processing.py:select_best_sample`: 반환형/타입힌트 동반 변경.

**🔴 CORE — Solution.edges 형 변경 여파 (정확히 items/values 구분)**
- [ ] `evaluator/evaluator.py:eval_flow`: `for eid,(s,t) in solution.edges.items()` +
      `edge_weights = {e.id: w}` id 룩업. (frozenset 룩업 제거; 평행 간선 가중치 정확해짐)
- [ ] `evaluator/apsp_sum_ranker.py:12`: `add_edges_from(solution.edges.values())`.
- [ ] `evaluator/evaluator.py:_sample_to_directed_edges`: 무변경(graph.edges + to_key 사용).
- [ ] `qubo/qubo_solver.py:245`, `dnc:132`: `{edge.vertices for ...}` → `{edge.id: edge.vertices}`
      (방향-only graph 솔루션도 id-keyed dict).
- [ ] `solver/dnc_mr2s_solver.py:271-291 _merge`: **재작성**. candidate 를 int edge.id 로 키잉
      (`solution.edges.items()`). `merged_edges` 를 `dict[int, tuple]` 로. `solution.edges` 의
      길이 로깅(155,170)은 `len()` 그대로 OK.
- [ ] `solver/dnc_mr2s_solver.py:_apply_merged_directions(407-421)`: **재작성**. `for eid,(u,v)
      in solution.edges.items(): graph.edges[eid].set_direction(u,v)`. 새 Edge/frozenset 제거.

**🟡 SHIM — 보류 알고리즘 최소 수정(단순그래프/pair 뷰 유지)**
- [ ] `edge_orient/t_join.py`: 시작에 `pair_to_edge = {e.pair_key(): e for e in
      graph.edges.values()}` 1회 구축. `:47` `set(graph.edges.keys())` →
      `set(pair_to_edge.keys())`. `:63` `graph.edges[frozenset({u,v})]` → `pair_to_edge[...]`
      또는 `graph.edge_by_pair(u,v)`. `:64` `Edge(u,v,...)` 는 결과 carrier 라 무변경
      (define_edge_direction 가 끝점만 씀).
- [ ] `cycle/face_cluster_partition.py:93,97,103`: 맵이 pair 기반 → `edge.id` → `edge.pair_key()`.
- [ ] `cycle/face_cluster_partition.py:95`: 입력이 전부 무방향이므로 `Edge(u,v,..,False)` 대신
      **원본 `edge` 재사용**(append edge). id 보존, 새 객체 0.
- [ ] `cycle/face_cluster_partition.py:100`: `Edge(a,b,..,True)` → `edge.oriented(a,b)`
      (id 보존, 비파괴). DnC merge 가 부모 edge.id 로 매칭하므로 필수.
- [ ] `cycle/face_cluster_partition.py:126,128,131 _validate_no_undirected_edge_overlap`:
      `edge.id` → `edge.pair_key()`. (`:95/:100` 이 같은 논리 간선의 인스턴스를 여러 subgraph 에
      넣는데, 이제 id 가 보존돼도 overlap 판정은 pair 기준이 맞음 — 같은 물리 간선 검출.)
- [ ] `solver/sa_mr2s_solver.py`: pair 뷰 shim. `:107,:186` `{edge.id: w}` →
      `{edge.pair_key(): w}` (lookup `:112,:191` 의 `frozenset({s,t})` 와 일치). `:208`
      `edge_weights[edge.id]` → `edge.weight` (edge 가 스코프에 있음). `:339` to_key 무변경.
- [ ] `solver/qubo_mr2s_solver.py:54`: `define_edge_direction` 호출 — graph.py 변경으로 자동
      대응(무변경, 회귀 확인만).

### Step 2: Tests
- [ ] 평행 간선 fixture 추가(같은 (u,v) 간선 2개, 다른 weight).
- [ ] `Edge` id 유니크/자동 부여 단위 테스트. `oriented()`/`set_direction()` id 보존 검증.
- [ ] `Graph.edge_by_pair`/`edges_by_pair` (평행 간선 → list 2개) 검증.
- [ ] QUBO 변수 충돌 안 함 검증(평행 간선 → 변수 2개, `e_{id}` 형).
- [ ] `Solution.edges` id-keyed 라운드트립 + `eval_flow` 평행 간선 가중치 합산 정확성.
- [ ] `define_edge_direction`: 끝점 매칭 + in-place set_direction, 원본 id 유지 검증
      (기존 `tests/.../test_qubo_mr2s_solver.py:112` `graph.edges == {frozenset: edge}` 갱신).
- [ ] DnC merge: 자식 solution id ↔ 부모 graph.edges id 왕복, 평행 간선 안 뭉갬.
- [ ] 기존 `.id`/`.to_key()`/`graph.edges[frozenset]` 검증 테스트 일괄 갱신(`tests/` 전반).
- [ ] 단순그래프 회귀: t_join / face_cluster / ILS / SA 결과 불변 확인.

### Step 3: Rollout / Rollback
- 단일 PR(코어 + ripple + shim + 테스트). 플래그 불필요(내부 표현 변경).
- 롤백: `git revert`.

## Validation
- **Commands to run:** `pytest`
- **Expected output:** 전체 통과. 평행 간선 신규 테스트 통과, 단순그래프 기존 테스트 회귀 없음.

## Risks & Rollback
- **Risks:**
  - DnC merge 가 frozenset 쌍 키로 매칭하던 부분 — id 전환 시 **조용히 0건 매칭**(빈 병합)
    위험. items()/int-id 재작성 + 왕복 테스트로 고정.
  - QUBO 변수명 id 전환 시 `to_key()` ↔ `get_indicator_function` 변수명 **불일치하면
    `safe_lookup` 이 0 으로 조용히 삼켜 틀린 방향** → 둘 다 `e_{id}` 단일 규칙, 테스트 고정.
  - face_cluster 서브그래프 emission 이 부모 id 를 잃으면 DnC merge 매칭 실패 → `:95` 재사용,
    `:100` `oriented()` 로 id 왕복 보장.
  - `sa_mr2s_solver`/`t_join` 의 `(u,v)`-튜플 내부 표현은 평행 간선에 본질적으로 단순그래프 —
    pair 뷰 shim 으로 단순그래프 회귀만 보장(멀티그래프 정확성은 non-goal).
  - 자동 카운터가 전역 상태 → 테스트는 **절대 id 값이 아닌 유니크성/매핑**을 단언(비결정 회피).
- **Rollback steps:** `git revert <merge_commit>`.

## Open Questions — 해소됨
- ~~"가중치 좋은 놈" 선점~~ → 폐기(유니크 id 면 충돌 없음).
- ~~QUBO 변수 라벨 string vs int~~ → string `f"e_{id}"` 확정(기존 스킴/`make_quadratic` 정합).
- ~~자동 카운터 위치~~ → `Edge` 클래스 변수 `itertools.count`. 테스트는 값 비의존.
