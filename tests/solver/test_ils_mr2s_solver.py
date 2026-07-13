from mr2s_module.domain import Edge, Graph
from mr2s_module.solver.ils_mr2s_solver import IlsMR2SSolver
from mr2s_module.solver.predefined import create_ils_solver


def test_ils_solver_triangle_graph() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(1, 3, 1, False),
  ])
  solver = IlsMR2SSolver(max_iter=10, patience=3)
  solution = solver.run(graph)

  assert len(solution.edges) == 3
  assert set(solution.edges) == set(graph.edges)

  # score 검증
  assert solution.score is not None
  assert solution.score.strong_connect_rate == 1.0
  assert solution.score.apsp_sum == 1.5

  # sample_set 검증
  assert solution.sample_set is not None
  samples = list(solution.sample_set.samples())
  assert len(samples) == 1
  for edge in graph.edges.values():
    assert edge.to_key() in samples[0]


def test_create_ils_solver_factory() -> None:
  solver = create_ils_solver()
  assert isinstance(solver, IlsMR2SSolver)
