from __future__ import annotations

from mr2s_module.domain import Edge, Graph
from mr2s_module.reduction import DegreeTwoChainReducer
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver

from tests.util.graph_fixtures import graph_from_pairs


def _edge_ids(edges) -> set[frozenset[int]]:
  return {edge.endpoint_key() for edge in edges}


def _undirected_ids(graph: Graph) -> set[frozenset[int]]:
  return {edge.endpoint_key() for edge in graph.edges.values()}


def _chain_graph_with_hubs() -> Graph:
  """a=0, b=1 을 차수 3 으로 만들고 그 사이에 차수-2 체인 0-2-3-1 을 둔 그래프.

  4,5 는 0,1 과 서로 연결되어 차수 3 을 유지하므로 추가 체인을 만들지 않는다.
  """
  return graph_from_pairs([
    (0, 2), (2, 3), (3, 1),                    # chain: 0 - 2 - 3 - 1
    (0, 4), (4, 1), (0, 5), (5, 1), (4, 5),    # 0,1,4,5 클리크로 허브 차수 보강
  ])


def test_reduce_collapses_chain_of_two_into_single_edge() -> None:
  reducer = DegreeTwoChainReducer()

  result = reducer.reduce(_chain_graph_with_hubs())

  assert len(result.chains) == 1
  chain = result.chains[0]
  assert chain.endpoints == (0, 1)
  assert chain.path == (0, 2, 3, 1)
  assert result.reduced_graph.has_endpoint_edge(0, 1)
  # 내부 정점 2, 3 은 축약 그래프에서 사라진다.
  assert result.reduced_graph.get_vertices() == {0, 1, 4, 5}


def test_collapsed_weight_is_sum_of_chain_edge_weights() -> None:
  graph = Graph(edges=[
    Edge(0, 2, 3, False),
    Edge(2, 3, 5, False),
    Edge(3, 1, 7, False),
    Edge(0, 4, 1, False),
    Edge(4, 1, 1, False),
    Edge(0, 5, 1, False),
    Edge(5, 1, 1, False),
    Edge(4, 5, 1, False),
  ])

  result = DegreeTwoChainReducer().reduce(graph)

  collapsed = result.reduced_graph.edge_for_endpoints(0, 1)
  assert collapsed.weight == 3 + 5 + 7
  assert result.chains[0].collapsed_weight == 15


def _single_internal_with_hubs() -> Graph:
  """단일 차수-2 정점 2 가 허브 0,1(차수 3) 사이에 놓인 그래프 (0-2-1)."""
  return graph_from_pairs([
    (0, 2), (2, 1),                  # single internal vertex chain
    (0, 3), (1, 3), (0, 4), (1, 4), (3, 4),   # 0,1,3,4 허브 보강
  ])


def test_reduce_skips_single_internal_when_min_is_two() -> None:
  graph = _single_internal_with_hubs()

  result = DegreeTwoChainReducer(min_internal_vertices=2).reduce(graph)

  assert result.chains == ()
  assert _undirected_ids(result.reduced_graph) == _undirected_ids(graph)


def test_reduce_collapses_single_when_min_is_one() -> None:
  graph = _single_internal_with_hubs()

  result = DegreeTwoChainReducer(min_internal_vertices=1).reduce(graph)

  assert len(result.chains) == 1
  assert result.chains[0].path == (0, 2, 1)
  assert result.reduced_graph.has_endpoint_edge(0, 1)


def test_reduce_skips_isolated_cycle() -> None:
  # 모든 정점이 차수 2 인 사각형 사이클 → 외부 끝점이 없어 축약 불가.
  graph = graph_from_pairs([(0, 1), (1, 2), (2, 3), (3, 0)])

  result = DegreeTwoChainReducer().reduce(graph)

  assert result.chains == ()
  assert _undirected_ids(result.reduced_graph) == _undirected_ids(graph)


def test_reduce_skips_when_direct_edge_exists() -> None:
  # 끝점 0,1 사이에 이미 직접 간선이 있으면 평행간선 충돌을 피해 skip.
  graph = graph_from_pairs([(0, 2), (2, 3), (3, 1), (0, 1), (0, 4), (1, 4)])

  result = DegreeTwoChainReducer().reduce(graph)

  assert result.chains == ()
  assert _undirected_ids(result.reduced_graph) == _undirected_ids(graph)


def test_reduce_skips_parallel_chains_sharing_endpoints() -> None:
  # 0,1 사이 두 평행 체인. 하나만 축약되고 다른 하나는 원본 간선으로 보존된다.
  graph = graph_from_pairs([
    (0, 2), (2, 3), (3, 1),   # chain A: 0-2-3-1
    (0, 4), (4, 5), (5, 1),   # chain B: 0-4-5-1
    (0, 6), (1, 6),           # 0,1 차수 보강
  ])

  result = DegreeTwoChainReducer().reduce(graph)

  assert len(result.chains) == 1
  # 축약 그래프는 단순 그래프로 유지된다 (간선 id 가 중복되지 않음).
  assert len(result.reduced_graph.edges) == len(
    _undirected_ids(result.reduced_graph)
  )
  assert result.reduced_graph.has_endpoint_edge(0, 1)


def test_expand_restores_chain_orientation_forward_and_reverse() -> None:
  result = DegreeTwoChainReducer().reduce(_chain_graph_with_hubs())

  forward = result.expand([Edge(0, 1, 3, True)])
  forward_dirs = {edge.vertices for edge in forward if edge.endpoint_key() != frozenset({0, 1})}
  assert (0, 2) in forward_dirs
  assert (2, 3) in forward_dirs
  assert (3, 1) in forward_dirs

  reverse = result.expand([Edge(1, 0, 3, True)])
  reverse_dirs = {edge.vertices for edge in reverse}
  assert (1, 3) in reverse_dirs
  assert (3, 2) in reverse_dirs
  assert (2, 0) in reverse_dirs


def test_expand_passes_through_non_chain_edges() -> None:
  result = DegreeTwoChainReducer().reduce(_chain_graph_with_hubs())

  # 체인 외 간선 (0,4) 는 그대로 통과해야 한다.
  expanded = result.expand([Edge(0, 1, 3, True), Edge(0, 4, 1, True)])
  ids = _edge_ids(expanded)
  assert frozenset({0, 4}) in ids
  assert all(edge.directed for edge in expanded)


def test_round_trip_preserves_original_edge_set() -> None:
  graph = _chain_graph_with_hubs()
  result = DegreeTwoChainReducer().reduce(graph)

  # 축약 그래프의 모든 간선을 임의 방향으로 정한 뒤 펼친다.
  oriented = [
    Edge(*edge.endpoints(), edge.weight, True)
    for edge in result.reduced_graph.edges.values()
  ]
  expanded = result.expand(oriented)

  assert _edge_ids(expanded) == _undirected_ids(graph)
  assert all(edge.directed for edge in expanded)


def _ladder_graph_with_chains() -> Graph:
  """허브 10,20,30,40 사이 4개의 차수-2 체인 + 대각 보강.

  원본 간선 10개(체인 4×2 + 대각 2), 축약 후 6개(체인 4 + 대각 2)로 줄어든다.
  허브는 모두 차수 3 이상, 2-edge-connected 라 강한 방향배정이 가능하다.
  """
  return graph_from_pairs([
    (10, 1), (1, 2), (2, 20),    # 10 - 20
    (20, 3), (3, 4), (4, 30),    # 20 - 30
    (30, 5), (5, 6), (6, 40),    # 30 - 40
    (40, 7), (7, 8), (8, 10),    # 40 - 10
    (10, 30), (20, 40),          # 대각선
  ])


def test_reduction_reduces_sa_variable_count() -> None:
  graph = _ladder_graph_with_chains()
  result = DegreeTwoChainReducer().reduce(graph)

  original_vars = len(graph.edges)
  reduced_vars = len(result.reduced_graph.edges)

  assert len(result.chains) == 4
  # 체인당 내부 정점 2개(간선 3→1) → 체인당 -2, 총 -8 변수의 유의미한 차이.
  assert reduced_vars == original_vars - 8


def test_sa_solve_on_reduced_graph_expands_to_valid_original_orientation() -> None:
  graph = _ladder_graph_with_chains()
  result = DegreeTwoChainReducer().reduce(graph)

  solver = SAMR2SSolver(num_restarts=1, random_seed=0)
  reduced_solution = solver.run(result.reduced_graph)

  oriented = [
    Edge(edge.vertices[0], edge.vertices[1], edge.weight, True)
    for edge in reduced_solution.edges.values()
  ]
  expanded = result.expand(oriented)

  # 펼친 해는 원본의 모든 무방향 간선을 정확히 한 번씩 방향만 부여해 덮는다.
  assert _edge_ids(expanded) == _undirected_ids(graph)
  assert all(edge.directed for edge in expanded)


def test_sa_solution_quality_is_comparable_with_and_without_reduction() -> None:
  graph = _ladder_graph_with_chains()
  reduced = DegreeTwoChainReducer().reduce(graph).reduced_graph

  baseline = SAMR2SSolver(num_restarts=2, random_seed=0).run(graph)
  on_reduced = SAMR2SSolver(num_restarts=2, random_seed=0).run(reduced)

  # 두 해 모두 강하게 연결된(도달 불가 쌍 없는) 유효 방향배정을 찾아야 한다.
  assert baseline.score is not None
  assert on_reduced.score is not None
  # 축약 그래프는 변수가 적으므로 동일/더 적은 SA 평가로 유효해를 얻는다.
  assert len(reduced.edges) < len(graph.edges)
