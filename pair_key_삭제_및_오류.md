# pair_key 삭제 및 멀티그래프 이관 잔재 감사 보고

- 일자: 2026-07-12
- 브랜치: `feat/ISSUE-52-edge-contraction`
- 방법: 적대적 리뷰 에이전트 4개(영역 분할) + runtime 검증 에이전트 2개 + 직접 콜사이트 검증
- 기준: (1) 한 군데서만 쓰이고 존재 이유가 불명확한 함수 (2) 말이 안 되는 `is not None` 류 방어 코드 (3) pair_key(정점쌍 키) 패턴 잔존 (4) 기타 비정상
- 참고: 전체 테스트 259 passed — 아래 버그들은 기존 테스트의 사각지대에 있음. 이 문서 작성 시점 기준 코드 수정 없음.

---

## 🔴 실행으로 크래시/오동작 확정

### 1. `tests/visualize_wall_protection.py` — 실행 즉시 크래시

```
File "tests/visualize_wall_protection.py", line 160, in compute_state
File "mr2s_module/cycle/face_cluster_partition.py", line 347, in _wall_protected_repair
    u, v = edge_endpoints[edge_id]
KeyError: frozenset({2, 3})
```

- 원인: 스크립트가 이관 이전(정점쌍 frozenset 키) 상태. `build_face_edges_map()` 의 frozenset 키를 edge-id 기반 `_collect_boundary_edges()` / `_wall_protected_repair()` 에 투입.
- `_wall_protected_repair` 는 subdivision 그래프(edge node 포함)를 전제하는데 일반 그래프를 넘겨 `_edge_endpoints()` 가 빈 dict 반환 → KeyError.
- 추가: line 160 을 고쳐도 `NoWallProtection` 경로(line 245)가 line 98 의 `tuple(sorted(e))` 에서 `TypeError: 'int' object is not iterable` 로 또 죽음 (파이프라인 본체는 int edge id 를 넘기므로). 실측 확인.
- 라이브러리 본체 `FaceClusterPartition.run` 은 정상 (실측 3 macros) — 스크립트만 깨짐.
- 조치안: edge-id 모델로 재작성 또는 삭제. (git 상 untracked 신규 파일.)

### 2. `tests/cycle/partition_visualization.py:122-146` — 면 색칠 100% 죽어있음 (조용한 오동작)

- `_best_face_owner(face_edges(face), macro_edge_ids)`: `face_edges()` 는 `frozenset({u,v})` 집합, `macro_edge_ids` 는 `set(sub_graph.edges.keys())` = int edge id 집합. 교집합 항상 공집합 → owner 항상 None.
- 실측: seed=42(108면)·seed=7(144면) 전부 owner=None, `draw_partition` 후 `ax.patches == 0` — 폴리곤 0개.
- `tests/cycle/test_partition_visualization.py` 는 PNG 존재/크기만 검사(6 passed)라 버그 불가시.
- 조치안: `face_edge_steps` 기반 edge-id 로 전환 + 테스트에 `len(ax.patches) > 0` 류 검증 추가.

---

## 🟡 pair_key 패턴 잔존 (기준 3)

### 3. `mr2s_module/util/planar_graph.py` — 구세대 정점쌍 키 삼형제

- `:10` `EdgeKey = frozenset[int]` 타입 별칭
- `:166-171` `face_edges()`
- `:229-237` `build_face_edges_map()`
- 프로덕션은 전부 `build_edge_id_face_edges_map`(edge-id) 사용. 이 셋의 소비자는 위 깨진 viz 스크립트 2개뿐. 평행 간선이 frozenset 키에서 충돌.
- 조치안: 셋 다 삭제 + `util/__init__.py` export 정리.

### 4. `(min(u,v), max(u,v))` 중복 제거 패턴 복붙

- `tests/util/graph_fixtures.py:44-55`, `tests/visualize_wall_protection.py:70-85` 두 곳에 동일 로직.
- Delaunay 는 평행 간선을 안 만들어 실해는 없으나, fixture 가 "멀티그래프 절대 안 나옴"을 조용히 보장.
- 조치안: 한 곳으로 통합 + 의도 주석 명시.

---

## 🟡 죽은/불명확 함수 (기준 1)

### 5. `mr2s_module/util/planar_graph.py:262` `clone_edge()` — 죽은 export + 잠재 버그

- 호출 0곳 (정의 + `util/__init__.py` export 만 존재).
- 실측: `orig.id=21 clone.id=22` — id 를 보존하지 않고 새로 발급. edge id 가 정체성인 현 모델에서 쓰는 순간 버그.
- `chain_contraction._clone_edges`(:70-79) 는 id 보존하는 자체 구현을 따로 씀 — 이 함수가 못 미더워서.
- 조치안: 삭제.

### 6. 임베딩 추정 3형제 — 호출 0곳 (테스트·protocol·getattr 동적 사용 전무)

- `solver/dnc_mr2s_solver.py:363` `_can_embed()`
- `solver/qubo_mr2s_solver.py:73` `estimate_embedding()`
- `solver/partition/embedding_aware.py:157` `can_embed()`
- 조치안: 삭제. (`estimate_embedding` 은 공개 API 의도였는지만 판단.)

### 7. `solver/mr2s_solver.py` `MR2SSolver` — 죽은 추상 스텁

- `run()` 이 `NotImplementedError`, 서브클래스 0 (runtime `__subclasses__() == []`), 인스턴스화 0.
- 단 `mr2s_module/__init__.py:42` + `__all__:88` top-level export → 제거는 공개 API 파괴. 판단 필요.

### 8. `solver/dnc_graph_partition_strategy.py` — 죽은 파일

- 순수 re-export 파일, repo 어디서도 import 안 함 (py/toml/md 전수 grep).
- 조치안: 삭제.

---

## 🟡 이상한 방어 코드 (기준 2) — runtime 검증으로 "죽은 방어"로 강등

### 9. `solver/dnc_mr2s_solver.py:421` `if edge is not None:` silent skip

- 메커니즘 실측 확인: bogus edge id 는 예외 없이 조용히 스킵.
- 단 유일한 프로덕션 호출자 `merge_solutions`(:269-301) 가 `graph.edges` 순회로 키를 만들어 bogus id 는 구조적으로 유입 불가 (실측: merge 단계에서 필터됨).
- 판정: 라이브 버그 아님, 죽은 방어 코드. lift 쪽 `lift_solution_edges` 는 반대로 raise — 일관성 없음.
- 조치안: raise 전환 권장 (직접 호출 시 버그 은폐 방지).

### 10. `solver/partition/vertex_count.py:55-63` — 안 읽히는 getattr 기본값 + 도달 불가 낭비 branch

- `getattr(face_cycle, "target_k", 2)` 기본값 2 는 절대 사용 안 됨 (set/restore 모두 `hasattr` 게이트).
- `target_k` 없는 전략이면 이진 탐색이 동일 `run(graph)` 를 반복 (stub 실측: 3회 동일 호출) — 순수 낭비.
- 단 실전 face_cycle 은 항상 `FaceClusterPartition`(`__init__` 에서 `target_k` 설정) → repo 내 도달 불가.
- 조치안: 생성자에서 `target_k` 지원 요구 or 미지원 시 1회 실행으로 단순화.

---

## 🟡 중복 로직

### 11. `qubo/qubo_solver.py:149-151` — `_fixed_embedding_child_sampler()`(:176-180) 본문 복붙

- 세 케이스(명시 sampler / `.child` fallback / None) 모두 helper 와 동작 동일 실측. `pytest -k embedding` 10 passed → 메서드 호출로 교체 안전.

---

## 🔵 소소

### 12. `Tjoin` — deprecated 인데 공개 API 잔존

- `mr2s_module/__init__.py` export 유지, 테스트 스위트에서 DeprecationWarning 10건 발생 중.
- 내부 `_endpoint_key()`(:12) 는 pair_key 패턴 그 자체 (deprecated 클래스 내부라 함께 삭제될 운명).
- 제거 시점 결정 필요.

### 13. `domain/edge.py:7` `weight: int` 어노테이션 거짓말

- harmonic super edge 실측 `weight=1.2000000000000002 (float)` (`chain_contraction.py:109-110`).
- int 전제 연산(`//`, `range`, 비트 연산 등) 0곳 — 실해 없음. DnC 쪽은 이미 `weight: float` 로 어노테이션.
- 조치안: `float` 로 수정.

---

## APSP / 평가 지표 단일 소스 감사

### 단일 소스 정상 ✓

- **stretch/APSP**: `evaluator/distance_util.py` 프리미티브(`build_directed_distance_graph*`, `stretch_totals`, `UndirectedApspCache`)를 `ApspSumRanker` 와 `SAMR2SSolver` 가 공유. 거리 = 1/weight 통일.
- **flow**: `util/flow_score.flow_imbalance` 단일 코어 — Evaluator·SA·ILS 위임, QUBO flow poly 는 동치성 테스트로 고정.
- **`Score` 생성**: `Evaluator.run` 단 한 곳.

### 자기만의 기준 발견

1. 🟡 **`solver/dnc_mr2s_solver.py:436-443`** — `score_merged_solution` 이 Evaluator 가 계산한 merged solution 의 `strong_connect_rate` 를 버리고 **자식 solution rate 의 곱**으로 덮어씀. 전역 강연결 실측값 ≠ 자식 rate 곱 — Evaluator 정의와 갈라진 유일한 지표. flow-score 3-way 갈라짐과 같은 종류의 문제.
2. 🟡 **`evaluator/evaluator.py:26-50`** — `_safe_lookup` + `_sample_to_directed_edges` 가 `qubo/solution_processing.py:15-52` 의 `safe_lookup` + `process_solution` 을 중복 구현. sample→방향 변환 로직 두 벌, 현재 동치이나 분기 위험.
3. 🔵 **ILS**(`edge_orient/iterated_local_search.py:132`) — flow 만 평가, APSP 무시. docstring 에 의도 명시된 결정이라 기록만.

---

## 기각된 주장 (에이전트 오판 정리)

- "`visualize_wall_protection.py:148` duck typing 으로 동작" → 오판. 실제 KeyError 크래시 (#1).
- `domain/edge.py:39,49` `tuple(sorted(...))` → pair_key 아님, 방향-끝점 검증용. 정상.
- `solver/sa_mr2s_solver.py:304` 평행 간선 붕괴 → treewidth 근사용이라 무해.

---

## 우선 수정 권장 순서

1. APSP-1: DnC `strong_connect_rate` 덮어쓰기 (지표 의미 오염)
2. #1·#2: viz 스크립트 edge-id 재작성 or 삭제 + `test_partition_visualization` 에 렌더 내용 검증 추가
3. #3·#5·#8: 죽은 pair-key 코드 일괄 삭제 (`EdgeKey`/`face_edges`/`build_face_edges_map`/`clone_edge`/`dnc_graph_partition_strategy.py`)
4. APSP-2·#11: 중복 로직 단일화 (sample→방향 변환, fixed embedding child sampler)
5. #9·#10·#13: 죽은 방어 코드 정리 + 어노테이션 수정
6. #7·#12: 공개 API 정리 (`MR2SSolver`, `Tjoin`) — breaking change 라 별도 결정
