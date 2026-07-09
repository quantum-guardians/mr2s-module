"""ISSUE-52: 축약 presolve 솔버 래퍼."""

import networkx as nx

from mr2s_module.domain import Edge, Graph, Score, Solution
from mr2s_module.reduction import ReductionMr2sSolver
from mr2s_module.util.sample_set import empty_binary_sample_set


class _StubSolver:
  """받은 그래프의 모든 간선을 endpoints 순방향으로 배향하는 결정적 스텁."""

  def __init__(self) -> None:
    self.seen_graph: Graph | None = None
    self.call_count = 0

  def run(self, graph: Graph) -> Solution:
    self.seen_graph = graph
    self.call_count += 1
    return Solution(
      edges={eid: e.endpoints() for eid, e in graph.edges.items()},
      graph=graph,
      sample_set=empty_binary_sample_set(),
      score=Score(apsp_sum=1.0, strong_connect_rate=1.0, flow_score=1.0),
    )


def _k4_with_chain() -> Graph:
  body = [
    Edge(0, 1, 1, False), Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  chain = [Edge(0, 10, 1, False), Edge(10, 11, 1, False), Edge(11, 1, 1, False)]
  return Graph(edges=body + chain)


def test_run_solves_contracted_graph_and_lifts_to_original() -> None:
  graph = _k4_with_chain()
  stub = _StubSolver()
  solution = ReductionMr2sSolver(mr2s_solver=stub).run(graph)

  # inner 는 축약 그래프만 본다: 체인 3간선 → super 1개, 원본과 다른 객체
  assert stub.seen_graph is not None
  assert stub.seen_graph is not graph
  assert len(stub.seen_graph.edges) == len(graph.edges) - 3 + 1

  # 반환 Solution 은 원본 그래프 기준
  assert solution.graph is graph
  assert set(solution.edges) == set(graph.edges)
  # inner score 는 축약 그래프 기준이므로 폐기
  assert solution.score is None


def test_run_keeps_original_graph_unmodified() -> None:
  graph = _k4_with_chain()
  ReductionMr2sSolver(mr2s_solver=_StubSolver()).run(graph)

  for edge in graph.edges.values():
    assert edge.directed is False


def test_run_skips_solver_when_graph_reduces_to_nothing() -> None:
  triangle = [Edge(0, 1, 1, False), Edge(1, 2, 1, False), Edge(2, 0, 1, False)]
  graph = Graph(edges=triangle)
  stub = _StubSolver()
  solution = ReductionMr2sSolver(mr2s_solver=stub).run(graph)

  assert stub.call_count == 0  # 변수 0개 → solver 생략
  assert set(solution.edges) == set(graph.edges)
  assert nx.is_strongly_connected(nx.DiGraph(solution.edges.values()))


def test_run_without_chains_still_protects_original() -> None:
  body = [
    Edge(0, 1, 1, False), Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  graph = Graph(edges=body)
  stub = _StubSolver()
  solution = ReductionMr2sSolver(mr2s_solver=stub).run(graph)

  assert stub.seen_graph is not graph  # 체인이 없어도 클론 위에서 solve
  assert set(solution.edges) == set(graph.edges)
  assert solution.graph is graph
