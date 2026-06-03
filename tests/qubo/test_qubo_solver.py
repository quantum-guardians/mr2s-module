from dimod import BinaryQuadraticModel, SampleSet
import networkx as nx
import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.evaluator import ApspSumRanker
from mr2s_module.qubo import InvalidEmbeddingError, QuboSolver
from mr2s_module.qubo.solution_processing import select_best_sample
from mr2s_module.util import empty_binary_sample_set
import mr2s_module.qubo.qubo_solver as qubo_solver_module


class EmptySampler:
  def sample(self, qubo, **kwargs):
    sample_set = empty_binary_sample_set()
    sample_set.info["reason"] = "empty"
    return sample_set


class FailingSampler:
  def sample(self, qubo, **kwargs):
    raise AssertionError("sampler should not be called")


class OneSampleSampler:
  def __init__(self):
    self.calls = 0

  def sample(self, qubo, **kwargs):
    self.calls += 1
    return SampleSet.from_samples({"e_1_2": 0}, vartype="BINARY", energy=0.0)


class TopologySampler(OneSampleSampler):
  def __init__(self, nodelist, edgelist):
    super().__init__()
    self.nodelist = nodelist
    self.edgelist = edgelist


def test_sample_logs_info_for_empty_sample_set(caplog) -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=EmptySampler())
  qubo = BinaryQuadraticModel({}, {}, 0.0, "BINARY")

  sample_set = solver._sample(solver.sampler, qubo)

  assert len(sample_set) == 0
  output = caplog.text
  assert "SampleSet is empty" in output
  assert "sample_set.info={'reason': 'empty'}" in output


def test_sample_logs_debug_context_for_empty_sample_set(caplog) -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=EmptySampler(), num_reads=5)
  qubo = BinaryQuadraticModel({"e_1_2": 1.0}, {("e_1_2", "e_2_3"): -1.0}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, True)])

  sample_set = solver._sample(solver.sampler, qubo, graph)

  assert len(sample_set) == 0
  output = caplog.text
  assert "sampler=EmptySampler" in output
  assert "num_reads=5" in output
  assert "qubo_variables=2" in output
  assert "qubo_interactions=1" in output
  assert "graph_vertices=3" in output
  assert "graph_edges=2" in output
  assert "directed_edges=1" in output
  assert "undirected_edges=1" in output


def test_run_returns_directed_solution_without_sampling_when_no_undirected_edges() -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=FailingSampler())
  qubo = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(2, 1, 1, True), Edge(2, 3, 1, True)])

  solution = solver.run(qubo, graph)

  assert solution.edges == {(2, 1), (2, 3)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 0


def test_run_returns_default_solution_without_sampling_when_qubo_has_no_variables() -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=FailingSampler())
  qubo = BinaryQuadraticModel({}, {}, 7.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  solution = solver.run(qubo, graph)

  assert solution.edges == {(1, 2)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 1
  assert list(solution.sample_set.samples()) == [{}]
  assert solution.sample_set.record.energy.tolist() == [7.0]
  assert solution.sample_set.info["fallback_reason"] == "zero_variable_qubo"


def test_run_returns_default_solution_when_sampler_returns_empty_sample_set() -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=EmptySampler())
  qubo = BinaryQuadraticModel({"e_1_2": 1.0}, {}, 4.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  solution = solver.run(qubo, graph)

  assert solution.edges == {(1, 2)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 1
  assert list(solution.sample_set.samples()) == [{}]
  assert solution.sample_set.record.energy.tolist() == [4.0]
  assert solution.sample_set.info["fallback_reason"] == "empty_sample_set"


def test_select_best_sample_returns_default_solution_for_empty_sample_set() -> None:
  sample_set = empty_binary_sample_set()
  edges = [Edge(1, 2, 1, False), Edge(2, 3, 1, True)]

  solution_edges = select_best_sample(sample_set, edges, ApspSumRanker())

  assert solution_edges == {(1, 2), (2, 3)}


def test_run_with_embedding_returns_directed_solution_without_sampling_when_no_undirected_edges() -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=FailingSampler())
  qubo = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, True)])

  solution = solver.run_with_embedding(qubo, graph, embedding={})

  assert solution.edges == {(1, 2)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 0


def test_run_with_embedding_returns_default_solution_when_qubo_has_no_variables() -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=FailingSampler())
  qubo = BinaryQuadraticModel({}, {}, -3.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  solution = solver.run_with_embedding(qubo, graph, embedding={})

  assert solution.edges == {(1, 2)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 1
  assert list(solution.sample_set.samples()) == [{}]
  assert solution.sample_set.record.energy.tolist() == [-3.0]
  assert solution.sample_set.info["fallback_reason"] == "zero_variable_qubo"


def test_run_with_embedding_returns_default_solution_when_sampler_returns_empty_sample_set(monkeypatch) -> None:
  child_sampler = EmptySampler()

  class FakeFixedEmbeddingComposite:
    def __init__(self, child, embedding):
      self.child = child
      self.embedding = embedding

    def sample(self, qubo, **kwargs):
      return self.child.sample(qubo, **kwargs)

  monkeypatch.setattr(
    qubo_solver_module,
    "FixedEmbeddingComposite",
    FakeFixedEmbeddingComposite,
  )
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )
  qubo = BinaryQuadraticModel({"e_1_2": 1.0}, {}, 4.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  solution = solver.run_with_embedding(qubo, graph, embedding={"e_1_2": ["q1"]})

  assert solution.edges == {(1, 2)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 1
  assert list(solution.sample_set.samples()) == [{}]
  assert solution.sample_set.info["fallback_reason"] == "empty_sample_set"


def test_run_with_embedding_stores_fixed_embedding_composite(monkeypatch) -> None:
  child_sampler = OneSampleSampler()
  embedding = {"e_1_2": ["q1"]}

  class FakeFixedEmbeddingComposite:
    def __init__(self, child, embedding):
      self.child = child
      self.embedding = embedding

    def sample(self, qubo, **kwargs):
      return self.child.sample(qubo, **kwargs)

  monkeypatch.setattr(
    qubo_solver_module,
    "FixedEmbeddingComposite",
    FakeFixedEmbeddingComposite,
  )
  monkeypatch.setattr(
    qubo_solver_module,
    "select_best_sample",
    lambda sample_set, edges, ranker: {(1, 2)},
  )
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )
  qubo = BinaryQuadraticModel({"e_1_2": 0.0}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  solution = solver.run_with_embedding(qubo, graph, embedding)

  assert isinstance(solver.fixed_sampler, FakeFixedEmbeddingComposite)
  assert solver.fixed_sampler.child is child_sampler
  assert solver.fixed_sampler.embedding == embedding
  assert child_sampler.calls == 1
  assert solution.edges == {(1, 2)}


def test_fixed_embedding_target_graph_uses_child_sampler_topology() -> None:
  child_sampler = TopologySampler(["q1", "q2"], [("q1", "q2")])
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )

  target_graph = solver.fixed_embedding_target_graph()

  assert target_graph is not None
  assert set(target_graph.nodes) == {"q1", "q2"}
  assert set(target_graph.edges) == {("q1", "q2")}


def test_fixed_embedding_target_graph_prefers_to_networkx_graph() -> None:
  target = nx.path_graph(["q1", "q2", "q3"])

  class NetworkxTopologySampler(OneSampleSampler):
    def to_networkx_graph(self):
      return target

  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=NetworkxTopologySampler(),
  )

  target_graph = solver.fixed_embedding_target_graph()

  assert target_graph is not target
  assert set(target_graph.edges) == {("q1", "q2"), ("q2", "q3")}


def test_run_with_embedding_rejects_disconnected_chain_before_sampling() -> None:
  child_sampler = TopologySampler(["q1", "q2"], [])
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )
  qubo = BinaryQuadraticModel({"x": 1.0}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  with pytest.raises(InvalidEmbeddingError, match="not connected"):
    solver.run_with_embedding(qubo, graph, {"x": ["q1", "q2"]})

  assert child_sampler.calls == 0


def test_run_with_embedding_rejects_missing_bqm_variable_chain() -> None:
  child_sampler = TopologySampler(["q1"], [])
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )
  qubo = BinaryQuadraticModel({"x": 1.0}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  with pytest.raises(InvalidEmbeddingError, match="missing non-empty chains"):
    solver.run_with_embedding(qubo, graph, {})

  assert child_sampler.calls == 0


def test_run_with_embedding_rejects_overlapping_chains() -> None:
  child_sampler = TopologySampler(["q1"], [])
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )
  qubo = BinaryQuadraticModel({"x": 1.0, "y": 1.0}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  with pytest.raises(InvalidEmbeddingError, match="overlap"):
    solver.run_with_embedding(qubo, graph, {"x": ["q1"], "y": ["q1"]})

  assert child_sampler.calls == 0


def test_run_with_embedding_rejects_unrepresented_interaction() -> None:
  child_sampler = TopologySampler(["q1", "q2"], [])
  solver = QuboSolver(
    ranker=ApspSumRanker(),
    sampler=FailingSampler(),
    fixed_embedding_child_sampler=child_sampler,
  )
  qubo = BinaryQuadraticModel(
    {"x": 1.0, "y": 1.0},
    {("x", "y"): -1.0},
    0.0,
    "BINARY",
  )
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  with pytest.raises(InvalidEmbeddingError, match="not represented"):
    solver.run_with_embedding(qubo, graph, {"x": ["q1"], "y": ["q2"]})

  assert child_sampler.calls == 0
