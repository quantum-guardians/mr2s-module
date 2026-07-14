from typing import cast

import pytest
import networkx as nx

from mr2s_module.cycle import FaceClusterPartition
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
from mr2s_module.evaluator import Evaluator
from mr2s_module.protocols import EvaluatorProtocol, QuboMatrix
from mr2s_module.qubo import InvalidEmbeddingError
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver, DnCSolution
from mr2s_module.solver.process_runner import ProcessStartMethod
from mr2s_module.solver.solve_context import QuboSolveContext
from mr2s_module.solver.partition import (
  DegeneracyPruningFaceCyclePartitionStrategy,
  EmbeddingAwareFaceCyclePartitionStrategy,
)
from mr2s_module.util import empty_binary_sample_set


def _edge_id_for_endpoints(graph: Graph, u: int, v: int) -> int:
  endpoints = (u, v) if u <= v else (v, u)
  ids = [
    edge.id
    for edge in graph.edges.values()
    if edge.endpoints() == endpoints
  ]
  assert len(ids) == 1
  return ids[0]


def _edge_for_endpoints(graph: Graph, u: int, v: int) -> Edge:
  return graph.edges[_edge_id_for_endpoints(graph, u, v)]


class StubEvaluator:
  def run(self, solution: Solution) -> Score:
    if solution.score is not None:
      return solution.score
    return Score(apsp_sum=10.0, strong_connect_rate=0.0, flow_score=2.0)


class StubMr2sSolver:
  evaluator = StubEvaluator()

  def build_bqm(self, graph: Graph) -> QuboMatrix:
    # StubBqm 은 dimod BQM 의 variables/edges 표면만 흉내내는 테스트 대역이다.
    return cast(QuboMatrix, StubBqm(
      variables=sorted(graph.get_vertices()),
      edges=list(graph.edges.values()),
    ))

  def run(self, graph: Graph) -> Solution:
    raise NotImplementedError("StubMr2sSolver.run must not be called")


class UnusedMr2sSolver:
  """mr2s_solver 가 사용되지 않는 경로를 증명하는 stub — 모든 접근이 실패한다."""

  @property
  def evaluator(self) -> EvaluatorProtocol:
    raise NotImplementedError("mr2s_solver must not be used")

  def run(self, graph: Graph) -> Solution:
    raise NotImplementedError("mr2s_solver must not be used")

  def build_bqm(self, graph: Graph) -> QuboMatrix:
    raise NotImplementedError("mr2s_solver must not be used")


class StubBqm:
  def __init__(self, variables: list[int], edges: list[Edge] | None = None) -> None:
    self.variables = variables
    self.edges = edges or []


class StubBqmMr2sSolver:
  evaluator = StubEvaluator()

  def __init__(self, bqm: StubBqm) -> None:
    self._bqm = bqm

  def build_bqm(self, graph: Graph) -> QuboMatrix:
    return cast(QuboMatrix, self._bqm)

  def run(self, graph: Graph) -> Solution:
    raise NotImplementedError("StubBqmMr2sSolver.run must not be called")


class StubFaceCycle:
  def __init__(
      self,
      sub_graphs: list[Graph],
      remaining_edges: list[Edge] | None = None,
  ) -> None:
    self.sub_graphs = sub_graphs
    self.remaining_edges = remaining_edges or []
    self.target_k = 2

  def run(self, graph: Graph) -> GraphPartitionResult:
    return GraphPartitionResult(
      sub_graphs=self.sub_graphs,
      remaining_edges=self.remaining_edges,
    )


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


class StubScoringMr2sSolver:
  evaluator = StubEvaluator()

  def run(self, graph: Graph) -> Solution:
    raise NotImplementedError("StubScoringMr2sSolver.run must not be called")

  def build_bqm(self, graph: Graph) -> QuboMatrix:
    raise NotImplementedError("StubScoringMr2sSolver.build_bqm must not be called")


class StubRunningMr2sSolver:
  evaluator = StubEvaluator()

  def __init__(self) -> None:
    self.run_graphs: list[Graph] = []

  def build_bqm(self, graph: Graph) -> QuboMatrix:
    return cast(QuboMatrix, StubBqm(
      variables=sorted(graph.get_vertices()),
      edges=list(graph.edges.values()),
    ))

  def run(self, graph: Graph) -> Solution:
    self.run_graphs.append(graph)
    return Solution(
      edges={edge.id: edge.vertices for edge in graph.edges.values()},
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
      edges={edge.id: (edge.vertices[1], edge.vertices[0]) for edge in graph.edges.values()},
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
    face_cycle=cast(FaceClusterPartition, face_cycle),
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
  solver = DnCMr2sSolver(mr2s_solver=UnusedMr2sSolver())

  merged = solver.merge_solutions(
    solutions=[
      Solution(
        edges={
          _edge_id_for_endpoints(graph, 1, 2): (1, 2),
          _edge_id_for_endpoints(graph, 2, 3): (2, 3),
        },
        graph=Graph(edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, False)]),
        sample_set=sample_set,
      ),
      Solution(
        edges={
          _edge_id_for_endpoints(graph, 2, 3): (2, 3),
          _edge_id_for_endpoints(graph, 3, 4): (4, 3),
        },
        graph=Graph(edges=[Edge(2, 3, 1, False), Edge(3, 4, 1, False)]),
        sample_set=empty_binary_sample_set(),
      ),
    ],
    graph=graph,
  )

  assert set(merged.edges.values()) == {(1, 2), (2, 3), (4, 3)}
  assert merged.graph is graph
  # 병합 해는 자식 sample 을 물려받지 않는다. 자식 sample 의 변수는 부모 간선을
  # 부분적으로만 덮어서 Evaluator 가 merged edges 대신 기본 정방향을 채점하게 된다.
  assert merged.sample_set is not sample_set
  assert len(merged.sample_set) == 0
  assert merged.score is None


def test_merge_solutions_keeps_one_direction_per_input_edge() -> None:
  graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
  ])
  solver = DnCMr2sSolver(mr2s_solver=UnusedMr2sSolver())

  merged = solver.merge_solutions(
    solutions=[
      Solution(
        edges={
          _edge_id_for_endpoints(graph, 1, 2): (1, 2),
          _edge_id_for_endpoints(graph, 2, 3): (2, 3),
        },
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
      Solution(
        edges={
          _edge_id_for_endpoints(graph, 1, 2): (2, 1),
          _edge_id_for_endpoints(graph, 2, 3): (3, 2),
        },
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
    ],
    graph=graph,
  )

  selected_ids = set(merged.edges)

  assert len(merged.edges) == len(graph.edges)
  assert selected_ids == set(graph.edges)


def test_merge_solutions_selects_direction_that_reduces_flow_imbalance() -> None:
  graph = Graph(edges=[
    Edge(1, 3, 2, False),
    Edge(2, 3, 1, False),
  ])
  solver = DnCMr2sSolver(mr2s_solver=UnusedMr2sSolver())

  merged = solver.merge_solutions(
    solutions=[
      Solution(
        edges={
          _edge_id_for_endpoints(graph, 1, 3): (1, 3),
          _edge_id_for_endpoints(graph, 2, 3): (2, 3),
        },
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
      Solution(
        edges={_edge_id_for_endpoints(graph, 2, 3): (3, 2)},
        graph=graph,
        sample_set=empty_binary_sample_set(),
      ),
    ],
    graph=graph,
  )

  assert (1, 3) in merged.edges.values()
  assert (3, 2) in merged.edges.values()


class RealEvaluatorMr2sSolver:
  """Evaluator 실측값을 그대로 쓰는 stub — DnC 채점 경로 검증용."""

  evaluator = Evaluator()

  def run(self, graph: Graph) -> Solution:
    raise NotImplementedError("RealEvaluatorMr2sSolver.run must not be called")

  def build_bqm(self, graph: Graph) -> QuboMatrix:
    raise NotImplementedError("RealEvaluatorMr2sSolver.build_bqm must not be called")


def _child_solution(u: int, v: int, strong_connect_rate: float) -> Solution:
  child_graph = Graph(edges=[Edge(u, v, 1, False)])
  return Solution(
    edges={_edge_id_for_endpoints(child_graph, u, v): (u, v)},
    graph=child_graph,
    sample_set=empty_binary_sample_set(),
    score=Score(
      apsp_sum=1.0,
      strong_connect_rate=strong_connect_rate,
      flow_score=0.0,
    ),
  )


def test_score_merged_solution_measures_strong_connectivity_on_merged_orientation() -> None:
  """병합 해의 strong_connect_rate 는 실측 전역 강연결이다 (자식 rate 의 곱이 아니다).

  자식 rate 의 곱은 전역 강연결과 다른 값이다: 자식이 각자 강연결이어도 병합
  결과가 강연결이 아닐 수 있고, 그 반대(아래 삼각형)도 가능하다.
  """
  # 경로 1→2→3: 자식은 각자 (단일 간선 그래프라) 강연결이지만 병합 결과는 아니다.
  path_graph = Graph(edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, False)])
  path_merged = Solution(
    edges={
      _edge_id_for_endpoints(path_graph, 1, 2): (1, 2),
      _edge_id_for_endpoints(path_graph, 2, 3): (2, 3),
    },
    graph=path_graph,
    sample_set=empty_binary_sample_set(),
  )
  solver = DnCMr2sSolver(mr2s_solver=RealEvaluatorMr2sSolver())

  path_score = solver.score_merged_solution(
    path_merged,
    [_child_solution(1, 2, 1.0), _child_solution(2, 3, 1.0)],
  )

  assert path_score.strong_connect_rate == 0.0

  # 삼각형 1→2→3→1: 병합 결과는 강연결. 자식 rate 곱(0.5*0.5*0.5=0.125)과 무관하다.
  cycle_graph = Graph(edges=[
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 1, 1, False),
  ])
  cycle_merged = Solution(
    edges={
      _edge_id_for_endpoints(cycle_graph, 1, 2): (1, 2),
      _edge_id_for_endpoints(cycle_graph, 2, 3): (2, 3),
      _edge_id_for_endpoints(cycle_graph, 3, 1): (3, 1),
    },
    graph=cycle_graph,
    sample_set=empty_binary_sample_set(),
  )

  cycle_score = solver.score_merged_solution(
    cycle_merged,
    [
      _child_solution(1, 2, 0.5),
      _child_solution(2, 3, 0.5),
      _child_solution(3, 1, 0.5),
    ],
  )

  assert cycle_score.strong_connect_rate == 1.0


def test_score_merged_solution_fills_missing_child_scores() -> None:
  graph = Graph(edges=[Edge(1, 2, 1, False)])
  merged = Solution(
    edges={_edge_id_for_endpoints(graph, 1, 2): (1, 2)},
    graph=graph,
    sample_set=empty_binary_sample_set(),
  )
  child_graph = Graph(edges=[Edge(1, 2, 1, False)])
  child = Solution(
    edges={_edge_id_for_endpoints(child_graph, 1, 2): (1, 2)},
    graph=child_graph,
    sample_set=empty_binary_sample_set(),
  )
  solver = DnCMr2sSolver(mr2s_solver=StubScoringMr2sSolver())

  score = solver.score_merged_solution(merged, [child])

  assert score.apsp_sum == 10.0
  assert score.flow_score == 2.0
  assert child.score is not None


def test_subgraph_processes_must_be_positive() -> None:
  with pytest.raises(ValueError, match="subgraph_processes"):
    DnCMr2sSolver(mr2s_solver=StubMr2sSolver(), subgraph_processes=0)


def test_subgraph_start_method_must_be_spawn_or_fork() -> None:
  with pytest.raises(ValueError, match="subgraph_start_method"):
    DnCMr2sSolver(
      mr2s_solver=StubMr2sSolver(),
      subgraph_start_method=cast(ProcessStartMethod, "forkserver"),
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
  start_methods: list[str | None] = []

  class FakeProcessRunner:
    def __init__(self, max_workers: int, start_method: str | None = None) -> None:
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
  assert set(solutions[0].edges.values()) == {(1, 2), (2, 3)}
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
  assert [set(solution.edges.values()) for solution in solutions] == [{(2, 1)}, {(4, 3)}]


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
  assert set(solutions[0].edges.values()) == {(1, 2)}


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
  assert set(solutions[0].edges.values()) == {(1, 2)}


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
  assert set(solutions[0].edges.values()) == {(1, 2)}


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
  assert set(solutions[0].edges.values()) == {(1, 2)}


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
  assert set(solutions[0].edges.values()) == {(1, 2)}


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
  assert set(solution.edges.values()) == {(1, 2)}
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
  assert set(solution.edges.values()) == {(1, 2)}
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
  estimate = partition.embedding_estimates[0]
  assert estimate is not None
  assert estimate.num_logical_variables == 2
  assert estimate.max_chain_length == 1
  assert len(estimate.embedding) == 2


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
  # 실제 face_cluster 처럼 서브그래프/remaining 은 부모 Edge 정체성(id)을 공유한다.
  child = Graph(edges=[_edge_for_endpoints(graph, 1, 2)])
  remaining = _edge_for_endpoints(graph, 2, 3)

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
  child_edge = _edge_for_endpoints(graph, 1, 2)
  assert child_edge.directed is True
  assert child_edge.vertices == (1, 2)
  remaining_edge_in_graph = _edge_for_endpoints(graph, 2, 3)
  assert remaining_edge_in_graph.directed is False
  assert set(solution.edges.values()) == {(1, 2), (2, 3)}
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
    face_cycle=cast(FaceClusterPartition, StubFaceCycle(sub_graphs=[old_child])),
    target_graph=nx.path_graph(2),
  )
  solver.face_cycle = cast(FaceClusterPartition, StubFaceCycle(sub_graphs=[new_child]))

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
