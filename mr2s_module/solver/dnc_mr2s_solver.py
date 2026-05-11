from dataclasses import dataclass, field
from typing import Iterable

from dimod import SampleSet

from mr2s_module import estimate_required_qubits
from mr2s_module.cycle import FaceCycle
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Graph, Solution
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver


@dataclass
class DnCMr2sSolver:
  mr2s_solver: QuboMR2SSolver
  face_cycle: FaceCycle = field(default_factory=lambda: FaceCycle(target_k=2, clusterer=KMeansFaceClusterer()))

  @staticmethod
  def _empty_sample_set() -> SampleSet:
    return SampleSet.from_samples([], vartype="BINARY", energy=[])

  def merge_solutions(
      self,
      solutions: Iterable[Solution],
      graph: Graph
  ) -> Solution:
    solution_list = list(solutions)
    merged_edges: set[tuple[int, int]] = set()

    for solution in solution_list:
      merged_edges.update(solution.edges)

    sample_set = (
      solution_list[0].sample_set
      if solution_list
      else self._empty_sample_set()
    )

    return Solution(
      edges=merged_edges,
      graph=graph,
      sample_set=sample_set,
      score=None,
    )

  @staticmethod
  def _can_recurse(parent: Graph, sub_graphs: list[Graph]) -> bool:
    if not sub_graphs:
      return False

    parent_edge_count = len(parent.edges)
    return all(
      0 < len(sub_graph.edges) < parent_edge_count
      for sub_graph in sub_graphs
    )

  def divide_graph(self, graph: Graph) -> list[Graph]:
    try:
      estimate_required_qubits(self.mr2s_solver.build_bqm(graph))
      return [graph]
    except RuntimeError:
      result = self.face_cycle.run(graph)
      sub_graphs = result.sub_graphs
      if not self._can_recurse(graph, sub_graphs):
        return [graph]

      leaf_graphs = []
      for sub_graph in sub_graphs:
        leaf_graphs += self.divide_graph(sub_graph)

    return leaf_graphs

  def run(self, graph: Graph) -> Solution:
    sub_graphs = self.divide_graph(graph)
    solutions = []
    for sub_graph in sub_graphs:
      solution = self.mr2s_solver.run(sub_graph)
      solutions.append(solution)

    merged_solution = self.merge_solutions(
      solutions=solutions,
      graph=graph
    )
    merged_solution.score = self.mr2s_solver.evaluator.run(merged_solution)
    return merged_solution
