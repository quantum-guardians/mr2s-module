from mr2s_module.domain import Edge, Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver, SATemperatureTrace


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
  assert {frozenset({u, v}) for u, v in solution.edges} == set(graph.edges.keys())
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

  edge = graph.edges[frozenset({1, 2})]
  assert edge.directed is True
  assert edge.vertices == (1, 2)
  assert (1, 2) in solution.edges


def test_run_reports_metrics_after_each_temperature_step() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(1, 3, 1, False),
  ])
  traces: list[SATemperatureTrace] = []
  solver = SAMR2SSolver(
    initial_temperature=2.0,
    final_temperature=0.5,
    cooling_rate=0.5,
    sweeps_per_temperature=1,
    num_restarts=2,
    random_seed=7,
    trace_callback=traces.append,
    early_stop_patience=None,
  )

  solver.run(graph)

  assert len(traces) == 4
  assert [trace.temperature for trace in traces] == [2.0, 1.0, 2.0, 1.0]
  assert [trace.total_iterations for trace in traces] == [3, 6, 9, 12]
  assert all(0.0 <= trace.acceptance_rate <= 1.0 for trace in traces)
  assert all(trace.best_objective <= trace.objective for trace in traces)


def test_run_stops_each_restart_after_consecutive_stale_steps() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  traces: list[SATemperatureTrace] = []
  solver = SAMR2SSolver(
    initial_temperature=8.0,
    final_temperature=0.5,
    cooling_rate=0.5,
    sweeps_per_temperature=1,
    num_restarts=2,
    random_seed=7,
    trace_callback=traces.append,
    early_stop_patience=2,
    min_temperature_steps=1,
    early_stop_acceptance_rate=1.0,
  )

  solver.run(graph)

  assert len(traces) == 4
  assert [trace.total_iterations for trace in traces] == [1, 2, 3, 4]
  assert [trace.stopped_early for trace in traces] == [False, True, False, True]
  assert [trace.stale_temperature_steps for trace in traces] == [1, 2, 1, 2]


def test_run_can_disable_early_stop() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  traces: list[SATemperatureTrace] = []
  solver = SAMR2SSolver(
    initial_temperature=8.0,
    final_temperature=0.5,
    cooling_rate=0.5,
    sweeps_per_temperature=1,
    num_restarts=1,
    random_seed=7,
    trace_callback=traces.append,
    early_stop_patience=None,
    min_temperature_steps=1,
    early_stop_acceptance_rate=1.0,
  )

  solver.run(graph)

  assert len(traces) == 4
  assert not any(trace.stopped_early for trace in traces)
