from dimod import BinaryQuadraticModel, SampleSet

from mr2s_module.domain import Edge, Graph
from mr2s_module.evaluator import ApspSumRanker
from mr2s_module.qubo import QuboSolver
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


def test_sample_prints_info_for_empty_sample_set(capsys) -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=EmptySampler())
  qubo = BinaryQuadraticModel({}, {}, 0.0, "BINARY")

  sample_set = solver._sample(solver.sampler, qubo)

  assert len(sample_set) == 0
  output = capsys.readouterr().out
  assert "SampleSet is empty" in output
  assert "sample_set.info={'reason': 'empty'}" in output


def test_sample_prints_debug_context_for_empty_sample_set(capsys) -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=EmptySampler(), num_reads=5)
  qubo = BinaryQuadraticModel({"e_1_2": 1.0}, {("e_1_2", "e_2_3"): -1.0}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False), Edge(2, 3, 1, True)])

  sample_set = solver._sample(solver.sampler, qubo, graph)

  assert len(sample_set) == 0
  output = capsys.readouterr().out
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


def test_run_with_embedding_returns_directed_solution_without_sampling_when_no_undirected_edges() -> None:
  solver = QuboSolver(ranker=ApspSumRanker(), sampler=FailingSampler())
  qubo = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, True)])

  solution = solver.run_with_embedding(qubo, graph, embedding={})

  assert solution.edges == {(1, 2)}
  assert solution.graph is graph
  assert len(solution.sample_set) == 0


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
  qubo = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
  graph = Graph(edges=[Edge(1, 2, 1, False)])

  solution = solver.run_with_embedding(qubo, graph, embedding)

  assert isinstance(solver.fixed_sampler, FakeFixedEmbeddingComposite)
  assert solver.fixed_sampler.child is child_sampler
  assert solver.fixed_sampler.embedding == embedding
  assert child_sampler.calls == 1
  assert solution.edges == {(1, 2)}
