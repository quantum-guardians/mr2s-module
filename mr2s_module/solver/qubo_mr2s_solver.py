from dimod import BinaryPolynomial, Vartype, BinaryQuadraticModel

from mr2s_module.evaluator import ApspSumRanker, Evaluator
from mr2s_module.domain import EmbeddingEstimate
from mr2s_module.protocols import (
  QuboSolverProtocol,
  FaceCycleProtocol,
  EvaluatorProtocol,
  Graph,
  PolyGeneratorProtocol, Solution
)
from mr2s_module.qubo import (
  FlowPolyGenerator,
  NHop,
  NHopPolyGenerator,
  SAQuboSolver,
  SmallWorldSpec,
)
from mr2s_module.util import add_polys
from mr2s_module.util.embedding_util import estimate_required_qubits
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm


class QuboMR2SSolver:
  def __init__(
      self,
      face_cycle: FaceCycleProtocol | None = None,
      qubo_solver: QuboSolverProtocol = SAQuboSolver(ranker=ApspSumRanker()),
      evaluator: EvaluatorProtocol = Evaluator(),
      poly_generators: list[PolyGeneratorProtocol] | set[PolyGeneratorProtocol] | None = None,
  ) -> None:
    if poly_generators is None:
      poly_generators = [
        FlowPolyGenerator(),
        NHopPolyGenerator(small_world_spec=SmallWorldSpec(n_hops=[NHop(2, 1), NHop(3, 1)]))
      ]
    self.face_cycle = face_cycle
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
    if self.face_cycle is not None:
      partition = self.face_cycle.run(graph)
      graph.define_edge_direction(set(partition.directed_edges()))

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
