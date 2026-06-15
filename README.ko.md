# mr2s-module

`mr2s-module`은 평면 그래프에서 간선 방향을 정하는 문제를 풀기 위한 Python 라이브러리입니다.
주어진 무방향 그래프에 대해 각 간선의 방향을 부여할 때, 다음 목표를 최대한 만족하도록 해를 찾습니다.

- 모든 정점 쌍 최단거리 합(APSP sum) 최소화
- 방향 그래프의 강연결성 보장
- 각 정점에서의 흐름 보존 최대화

이 라이브러리는 위 문제를 위해 다음 기능을 제공합니다.

- `FaceClusterPartition` 기반 전처리
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
3. `QuboSolver`로 QUBO를 풉니다.
4. `ApspSumRanker`로 후보 샘플 중 가장 좋은 해를 고릅니다.
5. `Evaluator`로 최종 `Solution`을 평가합니다.

핵심 분리는 다음과 같습니다.

- `SolutionRankerProtocol`: 샘플 선택용 스칼라 랭킹
- `EvaluatorProtocol`: 최종 `Solution -> Score` 평가

## 주요 구성 요소

### 전처리

- `FaceClusterPartition`
- `DegreeTwoChainReducer` — 차수-2 정점 체인을 단일 간선으로 축약 ([체인 축약](#체인-축약) 참고)

### QUBO 생성기

- `FlowPolyGenerator`
- `NHopPolyGenerator`
- `SmallWorldSpec`
- `NHop`

### Solver

- `QuboSolver` (통합 QUBO 솔버; 팩토리 메서드로 백엔드 선택)
  - `QuboSolver.create_sa_solver(ranker=...)` — 시뮬레이티드 어닐링 백엔드 (로컬, 자격증명 불필요)
  - `QuboSolver.create_qa_solver(ranker=...)` — D-Wave 양자 어닐러 백엔드; D-Wave API 자격증명 필요 — `DWAVE_API_TOKEN` 환경변수 설정 또는 `~/.config/dwave/dwave.conf` 구성
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
    QuboSolver,
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
    edge_orienter=None,
    qubo_solver=QuboSolver.create_sa_solver(ranker=ApspSumRanker()),
    evaluator=Evaluator(),
    poly_generators={FlowPolyGenerator(), n_hop_generator},
)

solution = solver.run(graph)

print(solution.edges)
print(solution.score)
```

## 체인 축약

MR2S 간선 방향배정에서 차수-2 정점은 두 인접 간선의 방향이 항상 일관되게 강제되므로,
차수-2 정점이 이어진 체인 전체가 단 1비트로 결정됩니다. 체인의 모든 간선을 각각 QUBO 변수로
두는 것은 변수 낭비이며, 더 중요하게는 small-world `NHop` 보상의 시야를 가립니다. n-hop 경로가
분기 없는 "복도"에 들어가면 다음 허브에 닿지 못하고 체인 내부에서 막다른 길로 끝나서, 같은
지평선(horizon) 안에서 허브-허브(매크로) 구조를 보지 못합니다.

`DegreeTwoChainReducer` 는 각 차수-2 체인 `a - x1 - … - xk - b` 를 단일 간선 `a - b` 로 축약하고,
`expand()` 로 방향 해를 원본 그래프 위에 복원합니다. 정점 약 300개의 평면 그래프에서 보통 QUBO
변수 수가 절반으로 줄고, 강연결 성공률이 좋아지며, APSP 품질은 동등 수준을 유지합니다.

> 축약 간선의 가중치는 원본 체인 가중치의 **합** 이라 `NHop`/APSP 항에 체인의 진짜 거리를
> 그대로 전달합니다. 흐름보존은 방향만 셉니다(`(out − in)²`, 간선당 ±1, 가중치 무관)므로
> 합 가중치를 그대로 풀어도 됩니다 — 별도 unit 재가중 불필요, 거리도 정확히 보존됩니다.

### 어댑터

`solve_with_chain_reduction` 은 평범한 `QuboSolver` 호출을
축약 → 풀이 → 복원 으로 감쌉니다. 반환되는 `Solution.edges` 는 **원본** 그래프
위의 방향 간선이라, 직접 푸는 것과 동일한 결과를 줍니다 — `solver.run` 을 직접 호출하는 대신
QUBO 빌더와 솔버를 주입하기만 하면 됩니다.

```python
from mr2s_module import (
    ApspSumRanker, FlowPolyGenerator, Graph, NHop, NHopPolyGenerator,
    QuboSolver, SmallWorldSpec,
)
from mr2s_module.reduction import solve_with_chain_reduction
from mr2s_module.util import add_polys
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm

# 다항식 합성 로직을 콜백으로 주입: (graph) -> QUBO.
def build_qubo(graph: Graph):
    n_hop = NHopPolyGenerator(small_world_spec=SmallWorldSpec(n_hops=[NHop(2, 1)]))
    return map_binary_poly_to_bqm(add_polys(FlowPolyGenerator().run(graph), n_hop.run(graph)))

solver = QuboSolver.create_sa_solver(ranker=ApspSumRanker(), num_reads=80)

# `solver.run(build_qubo(graph), graph)` 와 동일한 결과 + 체인 축약 적용:
solution = solve_with_chain_reduction(graph, build_qubo, solver)

print(solution.edges)   # 원본 그래프 위의 방향 간선
```

차수-2 체인이 없으면 어댑터는 원본 그래프를 그대로 풀어 동작이 완전히 동일합니다. 보조 함수
`expand_solution` 도 커스텀 파이프라인용으로 export 되어 있습니다.

### 벤치마크

n-hop 지평선 맹점을 측정하는 실험(구조 지표 M1/M2/M3 + 실제 SA 비교)은
[tests/run_chain_reduction_benchmark.py](tests/run_chain_reduction_benchmark.py) 에 있습니다.

```bash
python tests/run_chain_reduction_benchmark.py --num-reads 60
```

## 데모 스크립트

실행 가능한 데모는 아래 파일에 있습니다.

- [tests/run_sa_qubo_solver_demo.py](tests/run_sa_qubo_solver_demo.py)

이 스크립트는 다음을 지원합니다.

- Delaunay triangulation 기반 평면 그래프 생성
- biconnected 성질을 유지하면서 일부 edge 제거
- `FaceClusterPartition` 적용 여부 선택
- `QuboSolver` (시뮬레이티드 어닐링 백엔드) 실행
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
- `--use-face-cycle`: `FaceClusterPartition` 전처리 사용 여부
- `--target-k`: `FaceClusterPartition`의 target `k`

## 성능 분석

DnC MR2S solver 실행 로그 분석 보고서는 아래에서 확인할 수 있습니다.

- [docs/DNC_MR2S_SOLVER_PERFORMANCE_ANALYSIS.md](docs/DNC_MR2S_SOLVER_PERFORMANCE_ANALYSIS.md)
