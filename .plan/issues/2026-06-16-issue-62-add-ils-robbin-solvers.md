# 2026-06-16 — predefine에 ILS, Robbin 추가 및 독립 솔버 구조 개선

- Date: 2026-06-16
- GitHub Issue: #62
- Status: Completed

## Goal
- `predefined.py`에 ILS, Robbin 알고리즘을 추가합니다.
- ILS와 Robbin은 모든 간선의 방향을 결정하기 때문에, SA나 QUBO 솔버의 전처리기로만 동작하는 오버헤드를 줄이기 위해 `Mr2sSolverProtocol`을 만족하는 독립형 솔버(`RobbinMR2SSolver`, `IlsMR2SSolver`)를 구현합니다.
- **부모 클래스 도입 및 중복 제거**: 코드 중복을 피하기 위해 공통 로직을 담은 부모 클래스 `BaseEdgeOrientationSolver`를 만들고 이를 상속받게 합니다.
- **모든 간선 정렬 검증**: 부모 클래스에서는 `EdgeOrientationProtocol` 실행 결과 중 정렬되지 않은 간선이 존재할 경우 예외(`ValueError`)를 터트리도록 엄격히 제한합니다.
- **유연성 보존**: 일부 간선의 방향만 정해주는 알고리즘(혹은 ILS/Robbin 자체)을 여전히 SA/QUBO 솔버의 전처리기(`edge_orienter`)로 쓸 수 있는 기존 구조도 완벽히 호환되도록 그대로 유지합니다.

## Non-goals
- ILS 나 Robbin의 핵심 알고리즘 로직 자체를 재작성하거나 대폭 수정하는 것.

## Context / Constraints
- Python 3.11+
- `Mr2sSolverProtocol`을 준수해야 함 (즉, `run(self, graph: Graph) -> Solution`을 제공하고 `evaluator: EvaluatorProtocol`을 가짐).
- `predefined.py` 내의 팩토리 함수들을 통해 사용자가 쉽게 인스턴스화할 수 있도록 지원해야 함.

## Approach (Checklist)
- [ ] **Step 0: Recon**
  - `mr2s_module/solver/predefined.py`의 구조와 `mr2s_module/protocols.py` 내 `Mr2sSolverProtocol` 구조 확인 완료.
  - `mr2s_module/edge_orient/robbin.py`와 `iterated_local_search.py`가 반환하는 `OrientedEdges` 포맷 확인 완료.
- [ ] **Step 1: Implementation**
  - [ ] `mr2s_module/solver/base_edge_orientation_solver.py` 생성 및 `BaseEdgeOrientationSolver` 구현 (모든 간선 정렬 여부 검증 및 SampleSet 생성 공통화).
  - [ ] `mr2s_module/solver/robbin_mr2s_solver.py`가 `BaseEdgeOrientationSolver`를 상속하도록 리팩토링.
  - [ ] `mr2s_module/solver/ils_mr2s_solver.py`가 `BaseEdgeOrientationSolver`를 상속하도록 리팩토링.
  - [ ] `mr2s_module/solver/predefined.py`에 `create_robbin_solver()`, `create_ils_solver()` 팩토리 함수 추가 완료 확인 및 유지.
  - [ ] `mr2s_module/solver/__init__.py` 및 `mr2s_module/__init__.py`에 새로 추가된 `BaseEdgeOrientationSolver` 노출.
- [ ] **Step 2: Tests**
  - [ ] `tests/solver/test_base_edge_orientation_solver.py` 작성하여 정렬되지 않은 간선이 있을 때 예외가 발생하는지 테스트.
  - [ ] `tests/solver/test_robbin_mr2s_solver.py` 와 `test_ils_mr2s_solver.py` 테스트 코드가 리팩토링 후에도 여전히 잘 동작하는지 검증.
  - [ ] 기존 `pytest` 통합 테스트를 돌려서 깨지는 부분 없는지 확인.
- [ ] **Step 3: Rollout / Rollback**
  - 기존 SA, QUBO 솔버의 전처리기에 ILS, Robbin 등을 넣어 쓰는 것도 그대로 지원되므로 문제없음.

## Validation
- **Commands to run:**
  - `.venv/bin/python -m pytest`
- **Expected output:**
  - 새로 작성된 테스트를 포함해 전체 테스트 통과.

## Risks & Rollback
- **Risks:**
  - 새로 도입한 `RobbinMR2SSolver`, `IlsMR2SSolver` 내에서 `SampleSet` 생성 시 SA/QUBO의 sample_set 포맷과의 호환성 문제 (예: `dimod.SampleSet`을 예상하고 쓰는 후속 파이프라인에서 에러 발생 가능).
- **Rollback steps:**
  - `git checkout -- mr2s_module/solver/` 를 통해 롤백 진행.

## Open Questions
- ILS나 Robbin처럼 QUBO 기반이 아닌 고전 휴리스틱 솔버가 반환하는 `Solution` 내 `sample_set` 데이터는 어떻게 구성하는 것이 가장 적합할까요? (현재는 빈 `SampleSet` 혹은 가짜 sample을 채워넣는 식으로 대응할 예정입니다)
