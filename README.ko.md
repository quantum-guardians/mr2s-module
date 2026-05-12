# mr2s-module

`mr2s-module`은 평면 그래프에서 간선 방향을 정하는 문제를 풀기 위한 Python 라이브러리입니다.
주어진 무방향 그래프에 대해 각 간선의 방향을 부여할 때, 다음 목표를 최대한 만족하도록 해를 찾습니다.

- 모든 정점 쌍 최단거리 합(APSP sum) 최소화
- 방향 그래프의 강연결성 보장
- 각 정점에서의 흐름 보존 최대화

이 라이브러리는 위 문제를 위해 다음 기능을 제공합니다.

- `FaceCycle` 기반 전처리
- QUBO 다항식 생성
- simulated annealing 기반 해 탐색
- 샘플 랭킹
- 최종 해 평가

현재 파이프라인은 다음처럼 역할을 분리합니다.

- 샘플 랭킹: `ApspSumRanker`
- 최종 평가: `Evaluator`
- 결과 객체: `Solution`

## 요구 사항

- Python `>= 3.11`

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

테스트 및 데모 실행용 의존성까지 함께 설치하려면:

```bash
pip install -e ".[test]"
```

## 핵심 개념

### Graph

`Graph`는 입력 간선 목록을 저장합니다.

```python
from mr2s_module import Edge, Graph

graph = Graph(edges=[
    Edge(0, 1, 1, False),
    Edge(1, 2, 1, False),
    Edge(2, 0, 1, False),
])
```

### Solution

`Solution`은 QUBO solver가 반환하는 핵심 결과 객체입니다.

- `edges`: 선택된 방향 간선 집합
- `graph`: 해를 구성한 원본 그래프
- `sample_set`: annealer가 생성한 raw sample 집합
- `score`: 최종 평가 결과. 평가 이후 채워집니다.

### Score

`Score`는 최종 평가 지표를 담습니다.

- `apsp_sum`: 선택된 방향 그래프에 대한 APSP 기반 점수
- `strong_connect_rate`: 샘플들 중 강연결인 해의 비율
- `flow_score`: 각 정점의 흐름 불균형 제곱합
- `sample_score`: 샘플 해들 중 최소 에너지

## 구조

현재 흐름은 다음과 같습니다.

1. `Graph`를 생성하거나 전처리합니다.
2. 하나 이상의 polynomial generator로 QUBO 항을 만듭니다.
3. `SAQuboSolver`로 QUBO를 풉니다.
4. `ApspSumRanker`로 후보 샘플 중 가장 좋은 해를 고릅니다.
5. `Evaluator`로 최종 `Solution`을 평가합니다.

핵심 분리는 다음과 같습니다.

- `SolutionRankerProtocol`: 샘플 선택용 스칼라 랭킹
- `EvaluatorProtocol`: 최종 `Solution -> Score` 평가

## 주요 구성 요소

### 전처리

- `FaceCycle`

### QUBO 생성기

- `FlowPolyGenerator`
- `NHopPolyGenerator`
- `SmallWorldSpec`
- `NHop`

### Solver

- `SAQuboSolver`
- `QuboMR2SSolver`
- `SAMr2sSolver` (간선 방향 자체를 직접 simulated annealing으로 최적화)

### 랭킹 및 평가

- `ApspSumRanker`
- `Evaluator`

## 사용 예제

```python
from mr2s_module import (
    ApspSumRanker,
    Edge,
    Evaluator,
    FlowPolyGenerator,
    Graph,
    NHop,
    NHopPolyGenerator,
    QuboMR2SSolver,
    SAQuboSolver,
    SmallWorldSpec,
)

graph = Graph(edges=[
    Edge(0, 1, 1, False),
    Edge(1, 2, 1, False),
    Edge(2, 0, 1, False),
])

n_hop_generator = NHopPolyGenerator()
n_hop_generator.small_world_spec = SmallWorldSpec(
    n_hops=[NHop(n=2, weight=1)]
)

solver = QuboMR2SSolver(
    face_cycle=None,
    qubo_solver=SAQuboSolver(ranker=ApspSumRanker()),
    evaluator=Evaluator(),
    poly_generators={FlowPolyGenerator(), n_hop_generator},
)

solution = solver.run(graph)

print(solution.edges)
print(solution.score)
```

## 데모 스크립트

실행 가능한 데모는 아래 파일에 있습니다.

- [tests/run_sa_qubo_solver_demo.py](tests/run_sa_qubo_solver_demo.py)

이 스크립트는 다음을 지원합니다.

- Delaunay triangulation 기반 평면 그래프 생성
- biconnected 성질을 유지하면서 일부 edge 제거
- `FaceCycle` 적용 여부 선택
- `SAQuboSolver` 실행
- 선택된 방향과 최종 점수 출력

예시:

```bash
python tests/run_sa_qubo_solver_demo.py \
  --num-points 20 \
  --num-reads 30 \
  --remove-ratio 0.3 \
  --use-face-cycle \
  --target-k 8
```

인자 설명:

- `--num-points`: 생성할 평면 그래프의 정점 수
- `--seed`: 랜덤 시드
- `--weight`: 모든 간선에 공통으로 부여할 가중치
- `--remove-ratio`: biconnected 성질을 유지하며 제거할 edge 비율
- `--num-reads`: simulated annealing sample 수
- `--use-face-cycle`: `FaceCycle` 전처리 사용 여부
- `--target-k`: `FaceCycle`의 target `k`
