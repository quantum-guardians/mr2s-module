import pytest
import networkx as nx

from mr2s_module.domain import Edge, EmbeddingEstimate, Graph, GraphPartitionResult, Score, Solution
import mr2s_module.solver.dnc_mr2s_solver as dnc_mr2s_solver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver, DnCSolution
from mr2s_module.util import empty_binary_sample_set


class StubMr2sSolver:
  def build_bqm(self, graph: Graph):
    return graph


class StubFaceCycle:
  def __init__(
      self,
      sub_graphs: list[Graph],
      remaining_edges: list[Edge] | None = None,
  ) -> None:
    self.sub_graphs = sub_graphs
    self.remaining_edges = remaining_edges or []
    self.target_k = 2

  def run(self, graph: Graph):
    return type(
      "Partition",
      (),
      {
        "sub_graphs": self.sub_graphs,
        "remaining_edges": self.remaining_edges,
      },
    )()


class TargetKFaceCycle:
  def __init__(
      self,
      minimum_valid_target_k: int,
      invalid_sub_graphs: list[Graph],
      valid_sub_graphs: list[Graph],
  ) -> None:
    self.target_k = 2
    self.minimum_valid_target_k = minimum_valid_target_k
    self.invalid_sub_graphs = invalid_sub_graphs
    self.valid_sub_graphs = valid_sub_graphs
    self.calls: list[tuple[int, Graph]] = []

  def run(self, graph: Graph) -> GraphPartitionResult:
    self.calls.append((self.target_k, graph))
    if self.target_k >= self.minimum_valid_target_k:
      return GraphPartitionResult(
        sub_graphs=self.valid_sub_graphs,
        remaining_edges=[],
      )
    return GraphPartitionResult(
      sub_graphs=self.invalid_sub_graphs,
      remaining_edges=[],
    )


class StubEvaluator:
  def run(self, solution: Solution) -> Score:
    if solution.score is not None:
      return solution.score
    return Score(apsp_sum=10.0, strong_connect_rate=0.0, flow_score=2.0)


class StubScoringMr2sSolver:
  evaluator = StubEvaluator()


class StubRunningMr2sSolver:
  evaluator = StubEvaluator()

  def __init__(self) -> None:
    self.run_graphs: list[Graph] = []

  def build_bqm(self, graph: Graph):
    return graph

  def run(self, graph: Graph) -> Solution:
    self.run_graphs.append(graph)
    return Solution(
      edges={(edge.vertices[0], edge.vertices[1]) for edge in graph.edges},
      graph=graph,
      sample_set=empty_binary_sample_set(),
    )


def _fake_embedding_estimate(graph: Graph) -> EmbeddingEstimate:
  variables = sorted(graph.get_vertices())
  return EmbeddingEstimate(
    num_logical_variables=len(variables),
    num_quadratic_couplings=len(graph.edges),
    num_physical_qubits=len(variables),
    max_chain_length=1,
    embedding={variable: [variable] for variable in variables},
  )


def test_merge_solutions_combines_solution_edges() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 4, 1, False),
    Edge(4, 1, 1, False),
  ])
  sample_set = empty_binary_sample_set()
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
        sample_set=empty_binary_sample_set(),
      ),
    ],
    graph=graph,
  )

  assert merged.edges == {(1, 2), (2, 3), (4, 3)}
  assert merged.graph is graph
  assert merged.sample_set is sample_set
  assert merged.score is None


def test_merge_solutions_keeps_one_direction_per_input_edge() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  solver = DnCMr2sSolver(mr2s_solver=object())

  merged = solver.merge_solutions(
    solutions=[
      Solution(
        edges={(1, 2), (2, 3)},
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
      Solution(
        edges={(2, 1), (3, 2)},
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
    ],
    graph=graph,
  )

  selected_undirected_edges = {
    (min(source, target), max(source, target))
    for source, target in merged.edges
  }

  assert len(merged.edges) == len(graph.edges)
  assert selected_undirected_edges == {edge.id for edge in graph.edges}


def test_merge_solutions_selects_direction_that_reduces_flow_imbalance() -> None:
  graph = Graph(edges=[
    Edge(1, 3, 2, False),
    Edge(2, 3, 1, False),
  ])
  solver = DnCMr2sSolver(mr2s_solver=object())

  merged = solver.merge_solutions(
    solutions=[
      Solution(
        edges={(1, 3), (2, 3)},
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
      Solution(
        edges={(3, 2)},
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
    ],
    graph=graph,
  )

  assert (1, 3) in merged.edges
  assert (3, 2) in merged.edges


def test_score_merged_solution_multiplies_child_strong_connect_rates() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  merged = Solution(
    edges={(1, 2), (2, 3)},
    graph=graph,
    sample_set=empty_binary_sample_set(),
  )
  child_solutions = [
    Solution(
      edges={(1, 2)},
      graph=Graph(edges=[Edge(1, 2, 1, False)]),
      sample_set=empty_binary_sample_set(),
      score=Score(apsp_sum=1.0, strong_connect_rate=0.8, flow_score=0.0),
    ),
    Solution(
      edges={(2, 3)},
      graph=Graph(edges=[Edge(2, 3, 1, False)]),
      sample_set=empty_binary_sample_set(),
      score=Score(apsp_sum=1.0, strong_connect_rate=0.5, flow_score=0.0),
    ),
  ]
  solver = DnCMr2sSolver(mr2s_solver=StubScoringMr2sSolver())

  score = solver.score_merged_solution(merged, child_solutions)

  assert score.apsp_sum == 10.0
  assert score.flow_score == 2.0
  assert score.strong_connect_rate == pytest.approx(0.4)


def test_subgraph_processes_must_be_positive() -> None:
  with pytest.raises(ValueError, match="subgraph_processes"):
    DnCMr2sSolver(mr2s_solver=StubMr2sSolver(), subgraph_processes=0)


def test_resolve_subgraph_processes_uses_auto_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  solver = DnCMr2sSolver(mr2s_solver=StubMr2sSolver())
  monkeypatch.setattr(dnc_mr2s_solver.os, "process_cpu_count", lambda: 8, raising=False)

  assert solver._resolve_subgraph_processes(3) == 3


def test_resolve_subgraph_processes_caps_configured_count() -> None:
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    subgraph_processes=4,
  )

  assert solver._resolve_subgraph_processes(2) == 2


def test_solve_subgraphs_uses_process_pool_for_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  created_workers: list[int | None] = []
  contexts: list[object] = []

  class FakeProcessPoolExecutor:
    def __init__(self, max_workers: int | None = None, mp_context=None) -> None:
      created_workers.append(max_workers)
      contexts.append(mp_context)

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
      return None

    def map(self, func, iterable):
      return [func(item) for item in iterable]

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "ProcessPoolExecutor",
    FakeProcessPoolExecutor,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubRunningMr2sSolver(),
    subgraph_processes=2,
  )

  solutions = solver._solve_subgraphs([graph_a, graph_b])

  assert created_workers == [2]
  assert contexts == [None]
  assert [solution.graph for solution in solutions] == [graph_a, graph_b]


def test_solve_subgraphs_uses_spawn_context_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  start_methods: list[str] = []

  class FakeProcessPoolExecutor:
    def __init__(self, max_workers: int | None = None, mp_context=None) -> None:
      start_methods.append(mp_context.get_start_method())

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
      return None

    def map(self, func, iterable):
      return [func(item) for item in iterable]

  monkeypatch.setattr(dnc_mr2s_solver.os, "name", "nt")
  monkeypatch.setattr(
    dnc_mr2s_solver,
    "ProcessPoolExecutor",
    FakeProcessPoolExecutor,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubRunningMr2sSolver(),
    subgraph_processes=2,
  )

  solver._solve_subgraphs([graph_a, graph_b])

  assert start_methods == ["spawn"]


def test_solve_subgraphs_falls_back_when_process_pool_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  mr2s_solver = StubRunningMr2sSolver()

  class UnavailableProcessPoolExecutor:
    def __init__(self, max_workers: int | None = None, mp_context=None) -> None:
      raise PermissionError("semaphore unavailable")

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "ProcessPoolExecutor",
    UnavailableProcessPoolExecutor,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=2,
  )

  solutions = solver._solve_subgraphs([graph_a, graph_b])

  assert mr2s_solver.run_graphs == [graph_a, graph_b]
  assert [solution.graph for solution in solutions] == [graph_a, graph_b]


def test_solve_subgraphs_skips_qubo_solver_for_directed_only_graph() -> None:
  sub_graph = Graph(edges=[
    Edge(1, 2, 1, True),
    Edge(2, 3, 1, True),
  ])
  mr2s_solver = StubRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([sub_graph])

  assert mr2s_solver.run_graphs == []
  assert len(solutions) == 1
  assert solutions[0].edges == {(1, 2), (2, 3)}
  assert solutions[0].graph is sub_graph
  assert solutions[0].score is not None


def test_divide_graph_keeps_graph_when_embedding_estimate_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_succeeds(bqm, target_graph=None):
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(dnc_mr2s_solver, "estimate_required_qubits", estimate_succeeds)
  solver = DnCMr2sSolver(mr2s_solver=StubMr2sSolver())

  sub_graphs = solver.divide_graph(graph)

  assert sub_graphs == [graph]


def test_run_delegates_once_when_graph_is_not_divided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_succeeds(bqm, target_graph=None):
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(dnc_mr2s_solver, "estimate_required_qubits", estimate_succeeds)
  mr2s_solver = StubRunningMr2sSolver()
  solver = DnCMr2sSolver(mr2s_solver=mr2s_solver)

  solution = solver.run(graph)

  assert isinstance(solution, DnCSolution)
  assert mr2s_solver.run_graphs == [graph]
  assert solution.edges == {(1, 2)}
  assert solution.sub_graphs == [graph]
  assert len(solution.embedding_estimates) == 1


def test_divide_graph_returns_binary_search_subgraphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  child = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_fails_for_parent(bqm, target_graph=None):
    if len(bqm.edges) > 1:
      raise RuntimeError("too large")
    return _fake_embedding_estimate(bqm)

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


def test_divide_graph_raises_when_no_embeddable_partition_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])

  def estimate_always_fails(_bqm, target_graph=None):
    raise RuntimeError("too large")

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_always_fails,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    face_cycle=StubFaceCycle(sub_graphs=[]),
  )

  with pytest.raises(RuntimeError, match="no embeddable subgraph partition"):
    solver.divide_graph(graph)


def test_divide_graph_finds_target_k_with_binary_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 4, 1, False),
    Edge(4, 5, 1, False),
    Edge(5, 6, 1, False),
    Edge(6, 7, 1, False),
    Edge(7, 8, 1, False),
    Edge(8, 9, 1, False),
  ])
  invalid_child = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 4, 1, False),
  ])
  valid_sub_graphs = [
    Graph(edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, False)]),
    Graph(edges=[Edge(4, 5, 1, False), Edge(5, 6, 1, False)]),
  ]

  def estimate_fails_for_large_graphs(bqm, target_graph=None):
    if len(bqm.edges) > 2:
      raise RuntimeError("too large")
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_fails_for_large_graphs,
  )
  face_cycle = TargetKFaceCycle(
    minimum_valid_target_k=5,
    invalid_sub_graphs=[invalid_child],
    valid_sub_graphs=valid_sub_graphs,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    face_cycle=face_cycle,
  )

  sub_graphs = solver.divide_graph(graph)

  assert sub_graphs == valid_sub_graphs
  assert [target_k for target_k, _ in face_cycle.calls] == [5, 3, 4]
  assert all(called_graph is graph for _, called_graph in face_cycle.calls)
  assert face_cycle.target_k == 2


def test_run_solves_full_graph_after_applying_merged_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  child = Graph(edges=[Edge(1, 2, 1, False)])
  remaining = Edge(2, 3, 1, False)

  def estimate_fails_for_parent(bqm, target_graph=None):
    if len(bqm.edges) > 1:
      raise RuntimeError("too large")
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_fails_for_parent,
  )
  mr2s_solver = StubRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    face_cycle=StubFaceCycle(
      sub_graphs=[child],
      remaining_edges=[remaining],
    ),
  )

  solution = solver.run(graph)

  assert isinstance(solution, DnCSolution)
  assert mr2s_solver.run_graphs == [child, graph]
  assert graph.edges[0].directed is True
  assert graph.edges[0].vertices == (1, 2)
  assert graph.edges[1].id == remaining.id
  assert graph.edges[1].directed is False
  assert solution.edges == {(1, 2), (2, 3)}
  assert solution.sub_graphs == [child]
  assert len(solution.embedding_estimates) == 1


def test_embedding_estimate_returns_none_without_calling_estimator_when_edge_count_exceeds_target_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 4, 1, False),
  ])
  target_graph = nx.path_graph(2)
  called = False

  def estimate_required_qubits_should_not_be_called(_bqm, target_graph=None):
    nonlocal called
    called = True
    raise AssertionError("estimate_required_qubits should not be called")

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_required_qubits_should_not_be_called,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is None
  assert called is False


def test_embedding_estimate_passes_solver_target_graph_to_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  target_graph = nx.path_graph(3)
  passed_target_graph = None

  def estimate_required_qubits_with_target_graph(bqm, target_graph=None):
    nonlocal passed_target_graph
    passed_target_graph = target_graph
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_required_qubits_with_target_graph,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is not None
  assert passed_target_graph is target_graph


def test_embedding_estimate_calls_estimator_when_edge_count_matches_target_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  target_graph = nx.path_graph(2)
  called = False

  def estimate_required_qubits_called(_bqm, target_graph=None):
    nonlocal called
    called = True
    return _fake_embedding_estimate(_bqm)

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "estimate_required_qubits",
    estimate_required_qubits_called,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is not None
  assert called is True
