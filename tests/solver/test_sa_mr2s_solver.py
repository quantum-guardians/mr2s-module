from mr2s_module.domain import Edge, Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver


class StubEdgeOrienter:
  def __init__(self, predefined_edges: set[Edge]) -> None:
    self.predefined_edges = predefined_edges
    self.calls = 0

  def run(self, graph: Graph) -> OrientedEdges:
    self.calls += 1
    return OrientedEdges(edges=list(self.predefined_edges))


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
  assert {edge.endpoint_key() for edge in solution.edges.values()} == {
    e.endpoint_key() for e in graph.edges.values()
  }
  assert solution.score is not None
  assert solution.score.apsp_sum == 9.0
  assert solution.score.flow_score == 0.0
  assert solution.score.strong_connect_rate == 1.0


def test_run_applies_preprocessing_directed_edges_from_edge_orienter() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  predefined_edge = Edge(1, 2, 1, True)
  solver = SAMR2SSolver(
    edge_orienter=StubEdgeOrienter(predefined_edges={predefined_edge}),
    random_seed=3,
  )

  solution = solver.run(graph)

  edge = graph.edge_for_endpoints(1, 2)
  assert edge.directed is True
  assert edge.vertices == (1, 2)
  assert (1, 2) in {edge.vertices for edge in solution.edges.values()}


def test_run_allows_disconnected_soft_solution() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])

  solution = SAMR2SSolver(random_seed=3).run(graph)

  assert solution.score is not None
  assert solution.score.strong_connect_rate == 0.0
  assert solution.score.apsp_sum == float("inf")
  assert len(solution.edges) == 2


def test_run_stops_each_restart_after_consecutive_stale_steps(monkeypatch) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  solver = SAMR2SSolver(
    initial_temperature=8.0,
    final_temperature=0.5,
    cooling_rate=0.5,
    sweeps_per_temperature=1,
    num_restarts=2,
    random_seed=7,
    early_stop_patience=2,
    min_temperature_steps=1,
    early_stop_acceptance_rate=1.0,
  )
  objective_calls = 0
  original_objective = solver._objective

  def count_objective_calls(*args, **kwargs):
    nonlocal objective_calls
    objective_calls += 1
    return original_objective(*args, **kwargs)

  monkeypatch.setattr(solver, "_objective", count_objective_calls)

  solver.run(graph)

  assert objective_calls == 6


def test_run_can_disable_early_stop(monkeypatch) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  solver = SAMR2SSolver(
    initial_temperature=8.0,
    final_temperature=0.5,
    cooling_rate=0.5,
    sweeps_per_temperature=1,
    num_restarts=1,
    random_seed=7,
    early_stop_patience=None,
    min_temperature_steps=1,
    early_stop_acceptance_rate=1.0,
  )
  objective_calls = 0
  original_objective = solver._objective

  def count_objective_calls(*args, **kwargs):
    nonlocal objective_calls
    objective_calls += 1
    return original_objective(*args, **kwargs)

  monkeypatch.setattr(solver, "_objective", count_objective_calls)

  solver.run(graph)

  assert objective_calls == 5
