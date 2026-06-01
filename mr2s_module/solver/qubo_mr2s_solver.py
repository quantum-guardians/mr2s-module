from dimod import BinaryPolynomial, Vartype, BinaryQuadraticModel

from mr2s_module.evaluator import ApspSumRanker, Evaluator
from mr2s_module.domain import EmbeddingEstimate
from mr2s_module.protocols import (
  QuboSolverProtocol,
  EdgeOrientationProtocol,
  EvaluatorProtocol,
  Graph,
  PolyGeneratorProtocol, Solution
)
from mr2s_module.qubo import (
  FlowPolyGenerator,
  NHop,
  NHopPolyGenerator,
  QuboSolver,
  SmallWorldSpec,
)
from mr2s_module.util import add_polys
from mr2s_module.util.embedding_util import estimate_required_qubits
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm


class QuboMR2SSolver:
  def __init__(
      self,
      edge_orienter: EdgeOrientationProtocol | None = None,
      qubo_solver: QuboSolverProtocol = QuboSolver.create_sa_solver(ranker=ApspSumRanker()),
      evaluator: EvaluatorProtocol = Evaluator(),
      poly_generators: list[PolyGeneratorProtocol] | set[PolyGeneratorProtocol] | None = None,
  ) -> None:
    if poly_generators is None:
      poly_generators = [
        FlowPolyGenerator(),
        NHopPolyGenerator(small_world_spec=SmallWorldSpec(n_hops=[NHop(2, 1), NHop(3, 1)]))
      ]
    self.edge_orienter = edge_orienter
    self.qubo_solver = qubo_solver
    self.evaluator = evaluator
    self.poly_generators = poly_generators

  def _build_polynomial(
      self, graph: Graph
  ) -> BinaryPolynomial:
    terms = BinaryPolynomial({}, Vartype.BINARY)
    for poly_generator in self.poly_generators:
      temp = poly_generator.run(graph)
      terms = add_polys(terms, temp)
    return terms

  def build_bqm(self, graph) -> BinaryQuadraticModel:
    if self.edge_orienter is not None:
      graph.define_edge_direction(set(self.edge_orienter.run(graph).get_edges()))

    # build qubo
    binary_polynomial = self._build_polynomial(graph)
    return map_binary_poly_to_bqm(binary_polynomial)

  def estimate_embedding(self, graph: Graph) -> EmbeddingEstimate:
    return estimate_required_qubits(self.build_bqm(graph))

  def run(self, graph: Graph) -> Solution:
    bqm = self.build_bqm(graph)

    # solve
    solution = self.qubo_solver.run(bqm, graph)

    # evaluate
    score = self.evaluator.run(solution)

    solution.score = score

    return solution

  def run_with_embedding(
      self,
      graph: Graph,
      embedding_estimate: EmbeddingEstimate,
  ) -> Solution:
    if not embedding_estimate.has_physical_embedding:
      raise ValueError("embedding_estimate does not contain a physical embedding")

    bqm = self.build_bqm(graph)
    run_with_embedding = getattr(self.qubo_solver, "run_with_embedding", None)
    if run_with_embedding is None:
      raise NotImplementedError("QUBO solver does not support embedding reuse")

    solution = run_with_embedding(bqm, graph, embedding_estimate.embedding)
    score = self.evaluator.run(solution)
    solution.score = score
    return solution
