from __future__ import annotations

from dimod import SampleSet

from mr2s_module import (
  ApspSumRanker,
  Edge,
  FlowPolyGenerator,
  Graph,
  NHop,
  NHopPolyGenerator,
  QuboSolver,
  SmallWorldSpec,
)
from mr2s_module.domain import Solution
from mr2s_module.reduction import (
  DegreeTwoChainReducer,
  reweight_collapsed_to_unit,
  solve_with_chain_reduction,
)
from mr2s_module.util import add_polys
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm

from tests.util.graph_fixtures import graph_from_pairs


def _edge_ids(edges) -> set[frozenset[int]]:
  return {edge.id if isinstance(edge, Edge) else frozenset(edge) for edge in edges}


def _ladder_graph_with_chains() -> Graph:
  """허브 10,20,30,40 사이 4개의 차수-2 체인 + 대각 보강 (test_degree_two_chain 와 동일)."""
  return graph_from_pairs([
    (10, 1), (1, 2), (2, 20),
    (20, 3), (3, 4), (4, 30),
    (30, 5), (5, 6), (6, 40),
    (40, 7), (7, 8), (8, 10),
    (10, 30), (20, 40),
  ])


class _RecordingSolver:
  """주입된 graph 의 모든 간선을 정렬된 방향으로 정해 반환하는 결정적 스텁 솔버.

  어댑터가 어떤 그래프로 solver.run 을 호출했는지 기록해 검증에 쓴다.
  """

  def __init__(self) -> None:
    self.seen_graph: Graph | None = None

  def run(self, qubo, graph: Graph) -> Solution:
    self.seen_graph = graph
    edges = {edge.endpoints() for edge in graph.edges.values()}
    return Solution(edges=edges, graph=graph, sample_set=SampleSet.from_samples(
      {}, vartype="BINARY", energy=0.0))


def _build_qubo(graph: Graph):
  n_hop = NHopPolyGenerator(small_world_spec=SmallWorldSpec(n_hops=[NHop(2, 1)]))
  return map_binary_poly_to_bqm(add_polys(FlowPolyGenerator().run(graph), n_hop.run(graph)))


def test_reweight_collapsed_to_unit_sets_only_collapsed_edges_to_one() -> None:
  graph = Graph(edges=[
    Edge(0, 2, 3, False),
    Edge(2, 3, 5, False),
    Edge(3, 1, 7, False),
    Edge(0, 4, 9, False),  # pass-through 간선 (체인 아님)
    Edge(4, 1, 9, False),
    Edge(0, 5, 9, False),
    Edge(5, 1, 9, False),
    Edge(4, 5, 9, False),
  ])
  result = DegreeTwoChainReducer().reduce(graph)

  solve_graph = reweight_collapsed_to_unit(result)

  # 축약 간선 {0,1} 은 sum(15) 대신 unit(1) 로.
  assert solve_graph.edges[frozenset({0, 1})].weight == 1
  # pass-through 간선은 원본 가중치(9) 유지.
  assert solve_graph.edges[frozenset({0, 4})].weight == 9


def test_solve_with_reduction_solves_on_unit_reduced_graph() -> None:
  graph = _ladder_graph_with_chains()
  solver = _RecordingSolver()

  solve_with_chain_reduction(graph, _build_qubo, solver)

  reduced = DegreeTwoChainReducer().reduce(graph).reduced_graph
  # 어댑터는 원본이 아닌 '축약' 그래프로 풀어야 한다 (변수 절감).
  assert len(solver.seen_graph.edges) == len(reduced.edges) < len(graph.edges)
  # 축약 간선은 unit 가중치로 풀린다.
  for chain_endpoint in [frozenset({10, 20}), frozenset({20, 30})]:
    assert solver.seen_graph.edges[chain_endpoint].weight == 1


def test_solve_with_reduction_expands_to_full_original_orientation() -> None:
  graph = _ladder_graph_with_chains()

  solution = solve_with_chain_reduction(graph, _build_qubo, _RecordingSolver())

  # 반환 해는 원본 그래프 위에 모든 무방향 간선을 정확히 한 번씩 방향배정한다.
  assert solution.graph is graph
  assert _edge_ids(solution.edges) == set(graph.edges.keys())
  assert len(solution.edges) == len(graph.edges)


def test_solve_without_chains_runs_on_original_graph() -> None:
  # 차수-2 체인이 없는 삼각형 → 어댑터는 원본을 그대로 푼다.
  graph = graph_from_pairs([(0, 1), (1, 2), (2, 0)])
  solver = _RecordingSolver()

  solution = solve_with_chain_reduction(graph, _build_qubo, solver)

  assert solver.seen_graph is graph
  assert _edge_ids(solution.edges) == set(graph.edges.keys())


def test_solve_with_real_sa_solver_covers_all_original_edges() -> None:
  graph = _ladder_graph_with_chains()
  solver = QuboSolver.create_sa_solver(ranker=ApspSumRanker(), num_reads=20)

  solution = solve_with_chain_reduction(graph, _build_qubo, solver)

  # SA 무작위성과 무관하게 expand 는 항상 원본 간선 전체를 방향배정해 덮는다.
  assert _edge_ids(solution.edges) == set(graph.edges.keys())
  assert len(solution.edges) == len(graph.edges)
