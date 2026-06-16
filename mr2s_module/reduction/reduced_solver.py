"""차수-2 체인 축약을 QUBO 풀이 파이프라인에 끼우는 얇은 어댑터.

평소 흐름은::

    qubo = build_qubo(graph)          # NHop + Flow 등
    solution = solver.run(qubo, graph)

이 어댑터는 그 한 줄을 **축약 → 풀이 → 원본 복원**으로 감싼다::

    solution = solve_with_chain_reduction(graph, build_qubo, solver)

`solution.edges` 는 *원본* 그래프 위의 directed 간선 집합이라, 호출부 입장에서는
축약을 쓰지 않은 것과 동일한 인터페이스다(변수만 줄고 강연결은 더 안정적).

축약 간선의 `collapsed_weight` 는 원본 간선 가중치의 *합* 이라 체인의 거리(APSP/NHop)
정보를 그대로 들고 있다. FlowPolyGenerator 가 흐름보존을 가중치 무관(±1)하게 다루므로
축약 간선을 합 가중치 그대로 풀어도 강연결이 깨지지 않는다 → 별도 unit 재가중 불필요.
"""

from __future__ import annotations

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.protocols import QuboBuilderProtocol, QuboSolverProtocol
from mr2s_module.reduction.degree_two_chain import (
  ChainReductionResult,
  DegreeTwoChainReducer,
)

# build_qubo: 무방향 그래프 → QUBO(BQM). 예) NHop+Flow 다항식을 합성해 BQM 으로 매핑.
# 주입 협력자 컨벤션에 맞춰 Callable 대신 QuboBuilderProtocol 사용(protocols.py).


def expand_solution(
    result: ChainReductionResult,
    reduced_solution: Solution,
) -> set[tuple[int, int]]:
  """축약 그래프의 directed 해를 원본 그래프 위 directed 간선 집합으로 복원.

  reduced_solution.edges 는 (u, v) 튜플이라 가중치가 없으므로, 축약 그래프의 가중치를
  붙여 Edge 로 만든 뒤 `result.expand` 로 펼친다. (축약 간선 가중치는 expand 가
  원본 간선 가중치로 덮으므로 여기 가중치 값 자체는 무방하다.)
  """
  weight_by_id = {e.id: e.weight for e in result.reduced_graph.edges.values()}
  oriented = [
    Edge(s, t, weight_by_id[frozenset({s, t})], True)
    for s, t in reduced_solution.edges
  ]
  return {edge.vertices for edge in result.expand(oriented)}


def solve_with_chain_reduction(
    graph: Graph,
    build_qubo: QuboBuilderProtocol,
    solver: QuboSolverProtocol,
    *,
    min_internal_vertices: int = 2,
) -> Solution:
  """차수-2 체인을 축약해 QUBO 를 풀고, 해를 원본 그래프로 복원해 반환한다.

  매개변수
  --------
  graph: 원본 무방향 그래프.
  build_qubo: (무방향 그래프) -> QUBO. 호출부의 다항식 합성 로직을 그대로 주입.
  solver: .run(qubo, graph) -> Solution 을 제공하는 솔버 (예: QuboSolver).
  min_internal_vertices: 축약 임계값. 제품 기본값 2 유지 권장.

  반환: 원본 그래프 위 directed 간선을 담은 Solution. sample_set 은 축약 그래프
  풀이의 것을 그대로 보존(에너지/통계 추적용).
  """
  result = DegreeTwoChainReducer(min_internal_vertices).reduce(graph)

  # 축약할 체인이 없으면 원본 그대로 풀어 동작을 동일하게 유지(조기 반환).
  if not result.chains:
    return solver.run(build_qubo(graph), graph)

  reduced_graph = result.reduced_graph
  reduced_solution = solver.run(build_qubo(reduced_graph), reduced_graph)

  expanded_edges = expand_solution(result, reduced_solution)
  return Solution(
    edges=expanded_edges,
    graph=graph,
    sample_set=reduced_solution.sample_set,
    score=None,
  )
