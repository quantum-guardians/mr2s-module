"""차수-2 체인 축약을 QUBO 풀이 파이프라인에 끼우는 얇은 어댑터.

평소 흐름은::

    qubo = build_qubo(graph)          # NHop + Flow 등
    solution = solver.run(qubo, graph)

이 어댑터는 그 한 줄을 **축약 → unit 재가중 → 풀이 → 원본 복원**으로 감싼다::

    solution = solve_with_chain_reduction(graph, build_qubo, solver)

`solution.edges` 는 *원본* 그래프 위의 directed 간선 집합이라, 호출부 입장에서는
축약을 쓰지 않은 것과 동일한 인터페이스다(변수만 줄고 강연결은 더 안정적).

⚠️ 가중치 캐비엇: 축약 간선의 `collapsed_weight` 는 원본 간선 가중치의 *합* 이라
Flow/NHop QUBO 를 왜곡해 강연결이 붕괴한다. 따라서 QUBO 를 푸는 그래프에서는 축약
간선을 **unit(1) 가중치**로 재설정한다. 복원(`expand`)은 항상 원본 간선 가중치를 쓰므로
APSP/거리 정보는 보존된다.
"""

from __future__ import annotations

from typing import Callable

from dimod import SampleSet

from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.protocols import QuboMatrix
from mr2s_module.reduction.degree_two_chain import (
  ChainReductionResult,
  DegreeTwoChainReducer,
)

# build_qubo: 무방향 그래프 → QUBO(BQM). 예) NHop+Flow 다항식을 합성해 BQM 으로 매핑.
QuboBuilder = Callable[[Graph], QuboMatrix]

# solver: QuboSolver 처럼 .run(qubo, graph) -> Solution 을 제공하는 모든 솔버.
class _SupportsRun:  # 문서용 프로토콜 힌트 (런타임 강제 아님)
  def run(self, qubo: QuboMatrix, graph: Graph) -> Solution: ...


def reweight_collapsed_to_unit(result: ChainReductionResult) -> Graph:
  """축약 그래프를 QUBO 풀이용으로 정규화: 축약 간선만 가중치를 1 로 고정.

  pass-through(체인이 아니던) 간선은 원본 가중치 그대로 유지한다.
  """
  collapsed_ids = {frozenset(chain.endpoints) for chain in result.chains}
  edges: list[Edge] = []
  for edge in result.reduced_graph.edges.values():
    u, v = edge.endpoints()
    weight = 1 if edge.id in collapsed_ids else edge.weight
    edges.append(Edge(u, v, weight, edge.directed))
  return Graph(edges=edges)


def expand_solution(
    result: ChainReductionResult,
    reduced_solution: Solution,
    solve_graph: Graph,
) -> set[tuple[int, int]]:
  """축약 그래프의 directed 해를 원본 그래프 위 directed 간선 집합으로 복원.

  reduced_solution.edges 는 (u, v) 튜플이라 가중치가 없으므로, solve_graph 의 가중치를
  붙여 Edge 로 만든 뒤 `result.expand` 로 펼친다. (축약 간선 가중치는 expand 가
  원본 간선 가중치로 덮으므로 여기서 unit 이어도 무방하다.)
  """
  weight_by_id = {e.id: e.weight for e in solve_graph.edges.values()}
  oriented = [
    Edge(s, t, weight_by_id[frozenset({s, t})], True)
    for s, t in reduced_solution.edges
  ]
  return {edge.vertices for edge in result.expand(oriented)}


def solve_with_chain_reduction(
    graph: Graph,
    build_qubo: QuboBuilder,
    solver: _SupportsRun,
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

  solve_graph = reweight_collapsed_to_unit(result)
  reduced_solution = solver.run(build_qubo(solve_graph), solve_graph)

  expanded_edges = expand_solution(result, reduced_solution, solve_graph)
  return Solution(
    edges=expanded_edges,
    graph=graph,
    sample_set=_sample_set_of(reduced_solution),
    score=None,
  )


def _sample_set_of(solution: Solution) -> SampleSet:
  return solution.sample_set
