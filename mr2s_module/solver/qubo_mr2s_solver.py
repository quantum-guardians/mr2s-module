from dimod import BinaryPolynomial, Vartype

from mr2s_module.protocols import (
  QuboSolverProtocol,
  FaceCycleProtocol,
  EvaluatorProtocol,
  Graph,
  Score, PolyGeneratorProtocol, Solution
)
from mr2s_module.util import add_polys
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm


class QuboMR2SSolver:
  def __init__(
      self,
      face_cycle: FaceCycleProtocol | None,
      qubo_solver: QuboSolverProtocol,
      evaluator: EvaluatorProtocol,
      poly_generators: set[PolyGeneratorProtocol],
  ) -> None:
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

  def run(self, graph: Graph) -> Solution:
    if self.face_cycle is not None:
      partition = self.face_cycle.run(graph)
      # TODO: partition.sub_graphs 를 활용하는 방향은 아직 미합의.
      # 우선은 기존 동작을 유지하기 위해 directed remaining_edges 만 predefined 로 사용한다.
      predefined_edges = {edge for edge in partition.remaining_edges if edge.directed}
      graph.define_edge_direction(predefined_edges)

    # build qubo
    binary_polynomial = self._build_polynomial(graph)
    bqm = map_binary_poly_to_bqm(binary_polynomial)

    # solve
    solution = self.qubo_solver.run(bqm, graph)

    # evaluate
    score = self.evaluator.run(solution)

    solution.score = score

    return solution
