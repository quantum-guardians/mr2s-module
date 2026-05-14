from mr2s_module.domain import Edge, Graph, GraphPartitionResult
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver


class StubFaceCycle:
  def __init__(self, predefined_edges: set[Edge]) -> None:
    self.predefined_edges = predefined_edges
    self.calls = 0

  def run(self, graph: Graph) -> GraphPartitionResult:
    self.calls += 1
    return GraphPartitionResult(
      sub_graphs=[],
      remaining_edges=list(self.predefined_edges),
    )


def test_run_finds_strongly_connected_triangle_orientation() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(1, 3, 1, False),
  ])
  solver = SAMR2SSolver(
    random_seed=7,
    num_restarts=8,
    sweeps_per_temperature=3,
  )

  solution = solver.run(graph)

  assert len(solution.edges) == 3
  assert {(min(u, v), max(u, v)) for u, v in solution.edges} == {edge.id for edge in graph.edges}
  assert solution.score is not None
  assert solution.score.apsp_sum == 9.0
  assert solution.score.flow_score == 0.0
  assert solution.score.strong_connect_rate == 1.0


def test_run_applies_preprocessing_directed_edges_from_face_cycle() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  predefined_edge = Edge(1, 2, 1, True)
  solver = SAMR2SSolver(
    face_cycle=StubFaceCycle(predefined_edges={predefined_edge}),
    random_seed=3,
  )

  solution = solver.run(graph)

  assert graph.edges[0].directed is True
  assert graph.edges[0].vertices == (1, 2)
  assert (1, 2) in solution.edges
