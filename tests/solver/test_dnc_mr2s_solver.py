import pytest
import networkx as nx

from mr2s_module.domain import (
  Edge,
  EmbeddableGraphPartition,
  EmbeddingEstimate,
  Graph,
  GraphPartitionResult,
  Score,
  Solution,
)
import mr2s_module.solver.dnc_mr2s_solver as dnc_mr2s_solver
import mr2s_module.solver.partition.embedding_aware as embedding_aware
from mr2s_module.qubo import InvalidEmbeddingError
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver, DnCSolution
from mr2s_module.solver.solve_context import QuboSolveContext
from mr2s_module.solver.partition import (
  DegeneracyPruningFaceCyclePartitionStrategy,
  EmbeddingAwareFaceCyclePartitionStrategy,
)
from mr2s_module.util import empty_binary_sample_set


class StubMr2sSolver:
  def build_bqm(self, graph: Graph):
    return StubBqm(
      variables=sorted(graph.get_vertices()),
      edges=list(graph.edges.values()),
    )


class StubBqm:
  def __init__(self, variables: list[int], edges: list[Edge] | None = None) -> None:
    self.variables = variables
    self.edges = edges or []


class StubBqmMr2sSolver:
  def __init__(self, bqm: StubBqm) -> None:
    self._bqm = bqm

  def build_bqm(self, graph: Graph):
    return self._bqm


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


class StubPartitionStrategy:
  def __init__(self, partition: EmbeddableGraphPartition) -> None:
    self.partition = partition

  def run(self, graph: Graph) -> EmbeddableGraphPartition:
    return self.partition


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
    return StubBqm(
      variables=sorted(graph.get_vertices()),
      edges=list(graph.edges.values()),
    )

  def run(self, graph: Graph) -> Solution:
    self.run_graphs.append(graph)
    return Solution(
      edges={(edge.vertices[0], edge.vertices[1]) for edge in graph.edges.values()},
      graph=graph,
      sample_set=empty_binary_sample_set(),
    )


class StubEmbeddingAwareRunningMr2sSolver(StubRunningMr2sSolver):
  def __init__(self) -> None:
    super().__init__()
    self.run_with_embedding_calls: list[tuple[Graph, EmbeddingEstimate]] = []

  def run_with_embedding(
      self,
      graph: Graph,
      embedding_estimate: EmbeddingEstimate,
  ) -> Solution:
    self.run_with_embedding_calls.append((graph, embedding_estimate))
    return Solution(
      edges={(edge.vertices[1], edge.vertices[0]) for edge in graph.edges.values()},
      graph=graph,
      sample_set=empty_binary_sample_set(),
      score=Score(apsp_sum=1.0, strong_connect_rate=1.0, flow_score=0.0),
    )


class StubContextFailingMr2sSolver(StubRunningMr2sSolver):
  def __init__(self) -> None:
    super().__init__()
    self.run_with_context_calls: list[QuboSolveContext] = []

  def run_with_context(self, context: QuboSolveContext) -> Solution:
    self.run_with_context_calls.append(context)
    raise InvalidEmbeddingError("invalid reused embedding")


class StubValueErrorEmbeddingAwareRunningMr2sSolver(StubRunningMr2sSolver):
  def __init__(self) -> None:
    super().__init__()
    self.run_with_embedding_calls: list[tuple[Graph, EmbeddingEstimate]] = []

  def run_with_embedding(
      self,
      graph: Graph,
      embedding_estimate: EmbeddingEstimate,
  ) -> Solution:
    self.run_with_embedding_calls.append((graph, embedding_estimate))
    raise ValueError("invalid reused embedding")


class StubValueErrorContextRunningMr2sSolver(StubRunningMr2sSolver):
  def __init__(self) -> None:
    super().__init__()
    self.run_with_context_calls: list[QuboSolveContext] = []

  def run_with_context(self, context: QuboSolveContext) -> Solution:
    self.run_with_context_calls.append(context)
    raise ValueError("invalid reused context embedding")


def _fake_embedding_estimate(bqm_or_graph) -> EmbeddingEstimate:
  if hasattr(bqm_or_graph, "variables"):
    variables = sorted(bqm_or_graph.variables)
  else:
    variables = sorted(bqm_or_graph.get_vertices())
  if hasattr(bqm_or_graph, "edges"):
    edge_count = len(bqm_or_graph.edges)
  else:
    edge_count = 0
  return EmbeddingEstimate(
    num_logical_variables=len(variables),
    num_quadratic_couplings=edge_count,
    num_physical_qubits=len(variables),
    max_chain_length=1,
    embedding={variable: [variable] for variable in variables},
  )


def _embedding_aware_dnc_solver(
    mr2s_solver,
    face_cycle=None,
    target_graph=None,
) -> DnCMr2sSolver:
  face_cycle = face_cycle or StubFaceCycle(sub_graphs=[])
  target_graph = target_graph or nx.path_graph(100)
  return DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    face_cycle=face_cycle,
    target_graph=target_graph,
    graph_partition_strategy=EmbeddingAwareFaceCyclePartitionStrategy(
      mr2s_solver=mr2s_solver,
      face_cycle=face_cycle,
      target_graph=target_graph,
      embedding_estimator=embedding_aware.estimate_required_qubits,
    ),
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
    frozenset({source, target})
    for source, target in merged.edges
  }

  assert len(merged.edges) == len(graph.edges)
  assert selected_undirected_edges == set(graph.edges.keys())


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


def test_subgraph_start_method_must_be_spawn_or_fork() -> None:
  with pytest.raises(ValueError, match="subgraph_start_method"):
    DnCMr2sSolver(
      mr2s_solver=StubMr2sSolver(),
      subgraph_start_method="forkserver",
    )


def test_resolve_subgraph_processes_defaults_to_one_worker() -> None:
  solver = _embedding_aware_dnc_solver(StubMr2sSolver())

  assert solver._resolve_subgraph_processes(3) == 1


def test_resolve_subgraph_processes_caps_configured_count() -> None:
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    subgraph_processes=4,
  )

  assert solver._resolve_subgraph_processes(2) == 2


def test_solve_subgraphs_uses_process_runner_for_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  created_workers: list[int | None] = []
  start_methods: list[str | None] = []

  class FakeProcessRunner:
    def __init__(self, max_workers: int, start_method=None) -> None:
      created_workers.append(max_workers)
      start_methods.append(start_method)

    def map(self, func, iterable):
      return [func(item) for item in iterable]

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "ProcessRunner",
    FakeProcessRunner,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubRunningMr2sSolver(),
    subgraph_processes=2,
  )

  solutions = solver._solve_subgraphs([graph_a, graph_b])

  assert created_workers == [2]
  assert start_methods == [None]
  assert [solution.graph for solution in solutions] == [graph_a, graph_b]


def test_solve_subgraphs_passes_configured_start_method_to_process_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  start_methods: list[str] = []

  class FakeProcessRunner:
    def __init__(self, max_workers: int, start_method=None) -> None:
      start_methods.append(start_method)

    def map(self, func, iterable):
      return [func(item) for item in iterable]

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "ProcessRunner",
    FakeProcessRunner,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubRunningMr2sSolver(),
    subgraph_processes=2,
    subgraph_start_method="spawn",
  )

  solver._solve_subgraphs([graph_a, graph_b])

  assert start_methods == ["spawn"]


def test_solve_subgraphs_falls_back_when_process_runner_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  mr2s_solver = StubRunningMr2sSolver()

  class UnavailableProcessRunner:
    def __init__(self, max_workers: int, start_method=None) -> None:
      raise PermissionError("semaphore unavailable")

  monkeypatch.setattr(
    dnc_mr2s_solver,
    "ProcessRunner",
    UnavailableProcessRunner,
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


def test_solve_subgraphs_reuses_matching_physical_embedding() -> None:
  graph_a = Graph(edges=[Edge(1, 2, 1, False)])
  graph_b = Graph(edges=[Edge(3, 4, 1, False)])
  estimate_a = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: ["a"], 2: ["b"]},
  )
  estimate_b = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={3: ["c"], 4: ["d"]},
  )
  mr2s_solver = StubEmbeddingAwareRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([graph_a, graph_b], [estimate_a, estimate_b])

  assert mr2s_solver.run_graphs == []
  assert mr2s_solver.run_with_embedding_calls == [
    (graph_a, estimate_a),
    (graph_b, estimate_b),
  ]
  assert [solution.edges for solution in solutions] == [{(2, 1)}, {(4, 3)}]


def test_solve_subgraphs_falls_back_for_placeholder_embedding() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  placeholder_estimate = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: [], 2: []},
  )
  mr2s_solver = StubEmbeddingAwareRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([graph], [placeholder_estimate])

  assert mr2s_solver.run_with_embedding_calls == []
  assert mr2s_solver.run_graphs == [graph]
  assert solutions[0].edges == {(1, 2)}


def test_solve_subgraphs_keeps_directed_only_skip_before_embedding_reuse() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, True)])
  estimate = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: ["a"], 2: ["b"]},
  )
  mr2s_solver = StubEmbeddingAwareRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([graph], [estimate])

  assert mr2s_solver.run_with_embedding_calls == []
  assert mr2s_solver.run_graphs == []
  assert solutions[0].edges == {(1, 2)}


def test_solve_subgraphs_falls_back_when_context_embedding_is_invalid() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  estimate = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: ["a"], 2: ["b"]},
  )
  context = QuboSolveContext(
    graph=graph,
    bqm=StubBqm(variables=[1, 2]),
    target_graph=nx.path_graph(["a", "b"]),
    embedding_estimate=estimate,
  )
  mr2s_solver = StubContextFailingMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([graph], [estimate], [context])

  assert mr2s_solver.run_with_context_calls == [context]
  assert mr2s_solver.run_graphs == [graph]
  assert solutions[0].edges == {(1, 2)}


def test_solve_subgraphs_falls_back_when_reused_embedding_raises_value_error() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  estimate = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: ["a"], 2: ["b"]},
  )
  mr2s_solver = StubValueErrorEmbeddingAwareRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([graph], [estimate])

  assert mr2s_solver.run_with_embedding_calls == [(graph, estimate)]
  assert mr2s_solver.run_graphs == [graph]
  assert solutions[0].edges == {(1, 2)}


def test_solve_subgraphs_falls_back_when_reused_context_raises_value_error() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  estimate = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: ["a"], 2: ["b"]},
  )
  context = QuboSolveContext(
    graph=graph,
    bqm=StubBqm(variables=[1, 2]),
    target_graph=nx.path_graph(["a", "b"]),
    embedding_estimate=estimate,
  )
  mr2s_solver = StubValueErrorContextRunningMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    subgraph_processes=1,
  )

  solutions = solver._solve_subgraphs([graph], [estimate], [context])

  assert mr2s_solver.run_with_context_calls == [context]
  assert mr2s_solver.run_graphs == [graph]
  assert solutions[0].edges == {(1, 2)}


def test_run_direct_partition_falls_back_when_context_embedding_is_invalid() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  estimate = EmbeddingEstimate(
    num_logical_variables=2,
    num_quadratic_couplings=1,
    num_physical_qubits=2,
    max_chain_length=1,
    embedding={1: ["a"], 2: ["b"]},
  )
  context = QuboSolveContext(
    graph=graph,
    bqm=StubBqm(variables=[1, 2]),
    target_graph=nx.path_graph(["a", "b"]),
    embedding_estimate=estimate,
  )
  partition = EmbeddableGraphPartition(
    sub_graphs=[graph],
    embedding_estimates=[estimate],
    solve_contexts=[context],
  )
  mr2s_solver = StubContextFailingMr2sSolver()
  solver = DnCMr2sSolver(
    mr2s_solver=mr2s_solver,
    graph_partition_strategy=StubPartitionStrategy(partition),
  )

  solution = solver.run(graph)

  assert mr2s_solver.run_with_context_calls == [context]
  assert mr2s_solver.run_graphs == [graph]
  assert solution.edges == {(1, 2)}
  assert solution.solve_contexts == [context]


def test_divide_graph_keeps_graph_when_embedding_estimate_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_succeeds(bqm, target_graph=None):
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(embedding_aware, "estimate_required_qubits", estimate_succeeds)
  solver = DnCMr2sSolver(mr2s_solver=StubMr2sSolver())

  sub_graphs = solver.divide_graph(graph)

  assert sub_graphs == [graph]


def test_run_delegates_once_when_graph_is_not_divided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  def estimate_succeeds(bqm, target_graph=None):
    return _fake_embedding_estimate(bqm)

  monkeypatch.setattr(embedding_aware, "estimate_required_qubits", estimate_succeeds)
  mr2s_solver = StubRunningMr2sSolver()
  solver = _embedding_aware_dnc_solver(mr2s_solver)

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
    embedding_aware,
    "estimate_required_qubits",
    estimate_fails_for_parent,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
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
    embedding_aware,
    "estimate_required_qubits",
    estimate_always_fails,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
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
    embedding_aware,
    "estimate_required_qubits",
    estimate_fails_for_large_graphs,
  )
  face_cycle = TargetKFaceCycle(
    minimum_valid_target_k=5,
    invalid_sub_graphs=[invalid_child],
    valid_sub_graphs=valid_sub_graphs,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
    face_cycle=face_cycle,
  )

  sub_graphs = solver.divide_graph(graph)

  assert sub_graphs == valid_sub_graphs
  assert [target_k for target_k, _ in face_cycle.calls] == [5, 3, 4]
  assert all(called_graph is graph for _, called_graph in face_cycle.calls)
  assert face_cycle.target_k == 2


def test_degeneracy_pruning_partition_strategy_does_not_call_embedding_estimator() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(1, 3, 1, False),
  ])
  child = Graph(edges=[Edge(1, 2, 1, False)])

  def fail_if_called(*_args, **_kwargs):
    raise AssertionError("embedding estimator should not be called")

  face_cycle = StubFaceCycle(sub_graphs=[child])
  strategy = DegeneracyPruningFaceCyclePartitionStrategy(
    mr2s_solver=StubMr2sSolver(),
    face_cycle=face_cycle,
    target_graph=nx.path_graph(10),
    embedding_estimator=fail_if_called,
    max_degeneracy=1,
  )
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    graph_partition_strategy=strategy,
  )

  partition = solver._divide_graph_with_embeddings(graph)

  assert partition.sub_graphs == [child]
  assert len(partition.embedding_estimates) == 1
  assert partition.embedding_estimates[0].num_logical_variables == 2
  assert partition.embedding_estimates[0].max_chain_length == 1
  assert len(partition.embedding_estimates[0].embedding) == 2


def test_replacing_default_partition_strategy_after_init_does_not_sync_or_raise() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  replacement_partition = EmbeddableGraphPartition(
    sub_graphs=[graph],
    embedding_estimates=[],
    target_k=7,
  )
  solver = DnCMr2sSolver(mr2s_solver=StubMr2sSolver())
  solver.graph_partition_strategy = StubPartitionStrategy(replacement_partition)

  partition = solver._divide_graph_with_embeddings(graph)

  assert partition is replacement_partition
  assert solver._owns_graph_partition_strategy is False


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
    embedding_aware,
    "estimate_required_qubits",
    estimate_fails_for_parent,
  )
  mr2s_solver = StubRunningMr2sSolver()
  face_cycle = StubFaceCycle(
      sub_graphs=[child],
      remaining_edges=[remaining],
  )
  solver = _embedding_aware_dnc_solver(
    mr2s_solver,
    face_cycle=face_cycle,
  )

  solution = solver.run(graph)

  assert isinstance(solution, DnCSolution)
  assert mr2s_solver.run_graphs == [child, graph]
  child_edge = graph.edges[frozenset({1, 2})]
  assert child_edge.directed is True
  assert child_edge.vertices == (1, 2)
  remaining_edge_in_graph = graph.edges[remaining.id]
  assert remaining_edge_in_graph.directed is False
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
    embedding_aware,
    "estimate_required_qubits",
    estimate_required_qubits_should_not_be_called,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is None
  assert called is False


def test_embedding_estimate_uses_mutated_target_graph() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 4, 1, False),
  ])
  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    target_graph=nx.path_graph(10),
  )
  solver.target_graph = nx.path_graph(2)

  assert solver._embedding_estimate(graph) is None


def test_divide_graph_uses_mutated_face_cycle() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  old_child = Graph(edges=[Edge(1, 2, 1, False)])
  new_child = Graph(edges=[Edge(2, 3, 1, False)])

  solver = DnCMr2sSolver(
    mr2s_solver=StubMr2sSolver(),
    face_cycle=StubFaceCycle(sub_graphs=[old_child]),
    target_graph=nx.path_graph(2),
  )
  solver.face_cycle = StubFaceCycle(sub_graphs=[new_child])

  assert solver.divide_graph(graph) == [new_child]


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
    embedding_aware,
    "estimate_required_qubits",
    estimate_required_qubits_with_target_graph,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is not None
  assert passed_target_graph is target_graph


def test_embedding_estimate_calls_estimator_when_edge_count_and_bqm_variables_match_target_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 1, 1, False),
  ])
  target_graph = nx.path_graph(3)
  called = False

  def estimate_required_qubits_called(_bqm, target_graph=None):
    nonlocal called
    called = True
    return EmbeddingEstimate(
      num_logical_variables=0,
      num_quadratic_couplings=0,
      num_physical_qubits=0,
      max_chain_length=1,
      embedding={},
    )

  monkeypatch.setattr(
    embedding_aware,
    "estimate_required_qubits",
    estimate_required_qubits_called,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is not None
  assert called is True


def test_embedding_estimate_counts_only_undirected_edges_for_prefilter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, True),
    Edge(2, 1, 1, True),
    Edge(1, 2, 1, True),
  ])
  target_graph = nx.path_graph(2)
  called = False

  def estimate_required_qubits_called(_bqm, target_graph=None):
    nonlocal called
    called = True
    return EmbeddingEstimate(
      num_logical_variables=0,
      num_quadratic_couplings=0,
      num_physical_qubits=0,
      max_chain_length=1,
      embedding={},
    )

  monkeypatch.setattr(
    embedding_aware,
    "estimate_required_qubits",
    estimate_required_qubits_called,
  )
  solver = _embedding_aware_dnc_solver(
    StubMr2sSolver(),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is not None
  assert called is True


def test_embedding_estimate_skips_estimator_when_bqm_variables_exceed_target_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  target_graph = nx.path_graph(2)
  called = False

  def estimate_required_qubits_should_not_be_called(_bqm, target_graph=None):
    nonlocal called
    called = True
    raise AssertionError("estimate_required_qubits should not be called")

  monkeypatch.setattr(
    embedding_aware,
    "estimate_required_qubits",
    estimate_required_qubits_should_not_be_called,
  )
  solver = _embedding_aware_dnc_solver(
    StubBqmMr2sSolver(StubBqm(variables=[1, 2, 3])),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is None
  assert called is False


def test_embedding_estimate_calls_estimator_when_bqm_variables_within_target_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  target_graph = nx.path_graph(2)
  called = False

  def estimate_required_qubits_called(_bqm, target_graph=None):
    nonlocal called
    called = True
    return EmbeddingEstimate(
      num_logical_variables=2,
      num_quadratic_couplings=0,
      num_physical_qubits=2,
      max_chain_length=1,
      embedding={1: [1], 2: [2]},
    )

  monkeypatch.setattr(
    embedding_aware,
    "estimate_required_qubits",
    estimate_required_qubits_called,
  )
  solver = _embedding_aware_dnc_solver(
    StubBqmMr2sSolver(StubBqm(variables=[1, 2])),
    target_graph=target_graph,
  )

  assert solver._embedding_estimate(graph) is not None
  assert called is True
