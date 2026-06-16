# 2026-06-16 — SA Solver 트리 너비(Treewidth) 기반 유량 정규화 도입

- Date: 2026-06-16
- GitHub Issue: None
- Status: Completed

## Goal
- 그래프 크기 $V$가 확장될 때 유량 제약식(`flow_score`)의 에너지 기여도가 소멸하는 스케일 불일치 문제를 해결하기 위해, 그래프 고유의 위상 상수인 **트리 너비(Treewidth)**를 적용한 새로운 정규화 공식을 도입합니다.
- 새로운 정규화 분모 공식: $weight\_scale = \frac{total\_weight^2}{num\_edges \times treewidth}$
- 이를 통해 그래프 스케일($V$)에 무관하게 `flow`와 `apsp` 항 간의 상대적 가중치 비율을 대칭적(Scale-invariant)으로 유지시킵니다.
- 정규화 적용 후 최적의 `flow_weight` 기본값(예: 1.0)을 재조정합니다.

## Non-goals
- simulated annealing 외 다른 솔버의 정규화 공식 수정.

## Context / Constraints
- 트리 너비는 NP-Hard 연산이므로 SA 루프 내부가 아닌 `SAMR2SSolver.run` 진입 시 **단 1회만 근사하여 계산(Treewidth Approximation)**하고 캐싱해 두어 연산 오버헤드를 막습니다.
- NetworkX의 `networkx.algorithms.approximation.treewidth_min_degree` 알고리즘을 사용합니다.

## Approach (Checklist)
- [ ] **Step 0: Recon**
  - `SAMR2SSolver`의 `run`, `_anneal_bits`, `_objective`, `_build_graph_weight_scale` 수정 지점 확인 완료.
- [ ] **Step 1: Implementation**
  - `mr2s_module/solver/sa_mr2s_solver.py`:
    - `run(self, graph: Graph)` 진입부에서 `treewidth_min_degree`를 사용하여 `treewidth` 계산.
    - `_anneal_bits` 및 `_objective` 메소드 시그니처에 `treewidth` 매개변수 추가하여 연쇄 전파.
    - `_build_graph_weight_scale(self, graph: Graph, treewidth: float)`로 수정하고, $O(V / tw)$ 스케일을 반영한 보정 분모 계산식 구현.
    - 정규화 보정 효과를 감안하여 `flow_weight` 생성자 기본값(예: 1.0) 조율.
- [ ] **Step 2: Tests**
  - 새로운 스케일 정규화 코드가 잘 작동하는지 벤치마크 스크립트로 동작 검증.
  - 전체 단위 테스트 재실행 (`.venv/bin/python -m pytest`).

## Validation
- **Commands to run:**
  1. 전체 테스트 재실행: `.venv/bin/python -m pytest`
  2. 대규모 벤치마크 테스트 재실행하여 정규화 효과 점검.
- **Expected output:**
  - 노드 수 20, 50, 100 등 크기가 달라도 `flow_weight` 튜닝값이 급격히 폭발하지 않고 일정한 최적 범위 내에서 머무르는지 확인.
