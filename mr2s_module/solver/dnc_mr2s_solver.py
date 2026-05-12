from dataclasses import dataclass, field
from typing import Iterable

from dimod import SampleSet

from mr2s_module import estimate_required_qubits
from mr2s_module.cycle import FaceCycle
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Edge, Graph, GraphPartitionResult, Score, Solution
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
    candidate_edges: dict[tuple[int, int], tuple[int, int]] = {}

    for solution in solution_list:
      for source, target in solution.edges:
        edge_id = (min(source, target), max(source, target))
        direction = (source, target)
        previous_direction = candidate_edges.get(edge_id)
        if previous_direction is not None and previous_direction != direction:
          raise ValueError(
            f"Conflicting directions for edge {edge_id}: "
            f"{previous_direction} and {direction}"
          )
        candidate_edges[edge_id] = direction

    merged_edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
      direction = candidate_edges.get(edge.id)
      if direction is None:
        continue
      if edge.directed:
        direction = edge.vertices

      merged_edges.add(direction)

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

  def _can_embed(self, graph: Graph) -> bool:
    try:
      estimate_required_qubits(self.mr2s_solver.build_bqm(graph))
      return True
    except RuntimeError:
      return False

  def _partition_with_target_k(
      self,
      graph: Graph,
      target_k: int,
  ) -> GraphPartitionResult:
    previous_target_k = self.face_cycle.target_k
    self.face_cycle.target_k = target_k
    try:
      return self.face_cycle.run(graph)
    finally:
      self.face_cycle.target_k = previous_target_k

  def _find_partition_by_target_k(self, graph: Graph) -> list[Graph]:
    left = 2
    right = max(2, len(graph.edges))
    best_sub_graphs: list[Graph] | None = None

    while left <= right:
      target_k = (left + right) // 2
      result = self._partition_with_target_k(graph, target_k)
      sub_graphs = result.sub_graphs

      if self._can_recurse(graph, sub_graphs) and all(
          self._can_embed(sub_graph)
          for sub_graph in sub_graphs
      ):
        best_sub_graphs = sub_graphs
        right = target_k - 1
      else:
        left = target_k + 1

    return best_sub_graphs or []

  def divide_graph(self, graph: Graph) -> list[Graph]:
    if self._can_embed(graph):
      return [graph]

    sub_graphs = self._find_partition_by_target_k(graph)
    if not sub_graphs:
      return [graph]
    return sub_graphs

  @staticmethod
  def _apply_merged_directions(graph: Graph, solution: Solution) -> None:
    weights_by_edge = {
      edge.id: edge.weight
      for edge in graph.edges
    }
    predefined_edges = {
      Edge(
        source,
        target,
        weights_by_edge[(min(source, target), max(source, target))],
        True,
      )
      for source, target in solution.edges
    }
    graph.define_edge_direction(predefined_edges)

  def score_merged_solution(
      self,
      merged_solution: Solution,
      child_solutions: Iterable[Solution],
  ) -> Score:
    score = self.mr2s_solver.evaluator.run(merged_solution)
    strong_connect_rate = 1.0

    for child_solution in child_solutions:
      if child_solution.score is None:
        child_solution.score = self.mr2s_solver.evaluator.run(child_solution)
      strong_connect_rate *= child_solution.score.strong_connect_rate

    score.strong_connect_rate = strong_connect_rate
    return score

  def run(self, graph: Graph) -> Solution:
    sub_graphs = self.divide_graph(graph)
    if len(sub_graphs) == 1 and sub_graphs[0] is graph:
      return self.mr2s_solver.run(graph)

    solutions = []
    for sub_graph in sub_graphs:
      solution = self.mr2s_solver.run(sub_graph)
      solutions.append(solution)

    merged_solution = self.merge_solutions(
      solutions=solutions,
      graph=graph
    )
    self._apply_merged_directions(graph, merged_solution)
    final_solution = self.mr2s_solver.run(graph)
    final_solution.score = self.score_merged_solution(final_solution, solutions)
    return final_solution
