from mr2s_module.domain import Edge, Graph
from mr2s_module.solver.robbin_mr2s_solver import RobbinMR2SSolver
from mr2s_module.solver.predefined import create_robbin_solver


def test_robbin_solver_triangle_graph() -> None:
  # 삼각형 그래프: 강연결 가능
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(1, 3, 1, False),
  ])
  solver = RobbinMR2SSolver()
  solution = solver.run(graph)

  assert len(solution.edges) == 3
  # 모든 간선에 대해 방향이 결정되어 있어야 함
  assert {frozenset(d) for d in solution.edges.values()} == {e.pair_key() for e in graph.edges.values()}
  
  # score 검증
  assert solution.score is not None
  assert solution.score.strong_connect_rate == 1.0
  assert solution.score.apsp_sum == 1.5  # 사이클 방향화: 정방향 stretch 1, 역방향 2 → 평균 1.5

  # sample_set 검증
  assert solution.sample_set is not None
  samples = list(solution.sample_set.samples())
  assert len(samples) == 1
  for edge in graph.edges.values():
    assert edge.to_key() in samples[0]


def test_create_robbin_solver_factory() -> None:
  solver = create_robbin_solver()
  assert isinstance(solver, RobbinMR2SSolver)
