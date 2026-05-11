from dimod import SampleSet
import pytest

from mr2s_module.domain import Edge, Graph, Solution
import mr2s_module.solver.dnc_mr2s_solver as dnc_mr2s_solver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver


def _empty_sample_set() -> SampleSet:
  return SampleSet.from_samples([], vartype="BINARY", energy=[])


class StubMr2sSolver:
  def build_bqm(self, graph: Graph):
    return graph


class StubFaceCycle:
  def __init__(self, sub_graphs: list[Graph]) -> None:
    self.sub_graphs = sub_graphs

  def run(self, graph: Graph):
    return type("Partition", (), {"sub_graphs": self.sub_graphs})()


def test_merge_solutions_combines_solution_edges() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 4, 1, False),
    Edge(4, 1, 1, False),
  ])
  sample_set = _empty_sample_set()
  solver = DnCMr2sSolver(mr2s_solver=object())

  merged = solver.merge_solutions(
    solutions=[
      Solution(
        edges={(1, 2), (2, 3)},
        graph=Graph(edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, False)]),
        sample_set=sample_set,
      ),
      Solution(
        edges={(2, 3), (4, 3)},
        graph=Graph(edges=[Edge(2, 3, 1, False), Edge(3, 4, 1, False)]),
        sample_set=_empty_sample_set(),
      ),
    ],
    graph=graph,
  )

  assert merged.edges == {(1, 2), (2, 3), (4, 3)}
  assert merged.graph is graph
  assert merged.sample_set is sample_set
  assert merged.score is None


def test_divide_graph_keeps_graph_when_embedding_estimate_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_succeeds(_bqm):
    return None

  monkeypatch.setattr(dnc_mr2s_solver, "estimate_required_qubits", estimate_succeeds)
  solver = DnCMr2sSolver(mr2s_solver=StubMr2sSolver())

  sub_graphs = solver.divide_graph(graph)

  assert sub_graphs == [graph]


def test_divide_graph_returns_real_recursive_subgraphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  child = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_fails_for_parent(bqm):
    if len(bqm.edges) > 1:
      raise RuntimeError("too large")
    return None

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_fails_for_parent,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    face_cycle=StubFaceCycle(sub_graphs=[child]),
  )

  sub_graphs = solver.divide_graph(graph)

  assert sub_graphs == [child]
