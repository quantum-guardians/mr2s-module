# ISSUE-52 간선 축약 본 구현 정리 (2026-07-08)

브랜치 `feat/ISSUE-52-edge-contraction`. 실험용 코드(`tests/util/chain_contraction.py`)를
라이브러리로 승격하고, 축약 presolve 솔버 래퍼와 predefined 팩토리를 추가했다.
**전부 미커밋** — 3분할 커밋 계획은 하단 참고.

## 배경 (벤치마크 실증, 100시드)

- 가중 그래프에서 축약은 성능 최적화 이전에 **강연결 정확성 장치**:
  무축약 14/100 → harmonic 축약 85/100 성공.
- harmonic vs unit 직접 대결 64:30 → harmonic 채택.
- 변수 30% 절감(740→517), solve 시간 −40%.

## 변경 파일

| 파일 | 내용 |
|---|---|
| `mr2s_module/reduction/chain_contraction.py` | 신규 — contract + lift 본체 |
| `mr2s_module/reduction/reduced_solver.py` | 신규 — `ReductionMr2sSolver` 래퍼 |
| `mr2s_module/reduction/__init__.py` | export 확장 |
| `mr2s_module/solver/predefined.py` | `create_reduction_dnc_qubo_sa_solver()` 추가 (기존 팩토리 무수정) |
| `tests/reduction/test_chain_contraction.py` | 라이브러리 기준으로 이관 + 신규 케이스 |
| `tests/reduction/test_reduced_solver.py` | 신규 — 래퍼 단위 테스트 (스텁 솔버) |
| `tests/solver/test_predefined.py` | 팩토리 테스트 1개 추가 |
| `tests/util/chain_contraction.py` | 삭제 (승격 완료) |

## 핵심 설계

### contract_chains(graph, min_internal_vertices=1, super_edge_weight="harmonic")

- **path 체인** `a-x1-…-xk-b` → super edge 1개 (k 간선 → QUBO 변수 1개).
  가중치 harmonic `W = 1/Σ(1/wᵢ)`, 1 미만은 1 클램프 (APSP 거리 1/w 보존).
- **cycle 체인 (매달린 사이클, a==a)** → super edge 를 만들면 self-loop 라서
  (QUBO/nx 변환이 스킵, 방향 비트 무의미) **그래프에서 통째로 제거** = 변수 0개.
  `ContractionResult.cycle_chains` 에 기록, lift 때만 사용. ← 사용자 지시 반영.
- **고정점 반복**: cycle 제거는 부착점 차수를 2 깎아 새 degree-2 체인을 노출할
  수 있음 → cycle 이 나온 라운드 뒤엔 탐지-축약 재실행. path 축약은 끝점 차수
  불변이라 반복 유발 안 함 (cycle 없는 라운드에서 종료).
- **클론 정책**: 통과 간선 전부 클론 + 원본 id 유지. DnC 가 간선 방향을 in-place
  고정하므로 원본 보호. 체인이 없어도 클론 반환 (호출부 분기 제거).

### lift_solution_edges(solved, result)

- super edge 방향 → 체인 원본 간선들에 균일 전개. 고정점 반복 중 만들어진
  super edge 가 다음 라운드 체인(특히 cycle)에 들어갈 수 있어 **중첩 전개**를
  끝까지 수행.
- 제거된 cycle 은 **walk 순서로 균일 배향** 주입 — 어느 회전이든 부착점 경유
  왕복이 가능해 강연결 불변. 평가에만 참여.
- **안전장치**: solver 결과에 super edge id 누락 시 `ValueError` (조용한 미배향
  → 평가 왜곡 차단).

### ReductionMr2sSolver (reduced_solver.py)

- `run(graph)` = contract → inner solve(축약 그래프) → lift → **원본 그래프 기준**
  Solution 반환.
- `score=None`: inner score 는 축약 그래프(harmonic 가중치·사이클 제거) 기준이라
  폐기. 평가는 caller 가 lift 된 Solution 으로 수행 (evaluator 는 `solution.edges`
  + 원본 edge id 만 사용 → 무손실).
- **퇴화 가드**: 축약 후 간선 0개(그래프 전체가 사이클)면 solver 호출 생략,
  lift 만으로 배향 생성 (빈 SampleSet).
- 적용 범위: QUBO 계열 전용 (사용자 결정 — SA/ILS 경로 미적용).

### create_reduction_dnc_qubo_sa_solver()

`ReductionMr2sSolver(create_dnc_qubo_sa_solver())`. `target_graph` 미지원 —
파티션 전략이 보는 그래프가 축약 그래프라 원본 target 과 불일치.

## 사전 검사서 결론 (사용자 검토 반영)

| 항목 | 처리 |
|---|---|
| P1 self-loop (매달린 사이클) | 제거 + lift 균일 배향, 평가에만 참여 |
| P2 전체-사이클 퇴화 | solver 생략 가드 |
| P3 사이클 제거 연쇄 (부착점 차수 −2) | 고정점 반복 |
| P4 평행 super edge | 멀티그래프로 이미 안전 — 회귀 테스트만 |
| P5 forced 체인 | **비문제 확정** (입력 전부 무방향 → 발생 불가), 코드 작업 없음 |
| P6 score 정합 | score=None + lift 후 평가 |
| P7 super id 누락 | lift 에서 예외 |
| P8 Edge.id 충돌 | 클론 id 유지 + 원본 graph 객체로 Solution 재구성 |
| cycle 회전 방향 | walk 순서 고정 (n-hop w³ 지배 실측상 정밀화는 과투자) |

## 테스트 결과

- `pytest -m "not slow"`: **230 passed**, 2 failed, 1 skipped.
- 실패 2건은 `tests/solver/test_process_runner.py` — Windows venv python 에
  `os.killpg` 부재. **클린 트리에서도 동일 실패 (사전 존재, 본 작업 무관)**.
- 신규/이관 테스트 커버: 사이클 제거·균일 회전·SC / 전체-삼각형 퇴화 /
  고정점+중첩 super edge 전개(2단 케이스) / theta graph 평행 super edge 3개 /
  super id 누락·방향 불일치 예외 / 원본 비변형 / harmonic·unit·클램프 가중치.

## 커밋 계획 (미실행)

1. `feat(reduction): add degree-2 chain detector` — 탐지기 + 테스트
2. `feat(reduction): add chain contraction and lift` — 축약+lift + 테스트
3. `feat(solver): add reduction dnc qubo sa solver factory` — 래퍼+팩토리 + 테스트

## 남은 일

- 실데이터 그래프 확보 시 `print_chain_report` 로 축약률 확인 (<10% 면 재검토).
- 남은 15% SC 실패 원인(n-hop w³ 지배)의 항 정규화 — 별도 이슈.
