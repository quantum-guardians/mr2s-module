from dataclasses import dataclass, field
from typing import Iterable

from dimod import SampleSet

from mr2s_module import estimate_required_qubits
from mr2s_module.cycle import FaceCycle
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Graph, Score, Solution
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
    candidate_edges: dict[tuple[int, int], set[tuple[int, int]]] = {}
    balance: dict[int, float] = {}

    for solution in solution_list:
      for source, target in solution.edges:
        edge_id = (min(source, target), max(source, target))
        candidate_edges.setdefault(edge_id, set()).add((source, target))

    merged_edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
      candidates = candidate_edges.get(edge.id, set())
      if not candidates:
        continue
      if edge.directed:
        direction = edge.vertices
      elif len(candidates) == 1:
        direction = next(iter(candidates))
      elif len(candidates) > 1:
        direction = self._select_merge_direction(candidates, edge.weight, balance)

      merged_edges.add(direction)
      self._apply_flow_balance(direction, edge.weight, balance)

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
  def _apply_flow_balance(
      direction: tuple[int, int],
      weight: float,
      balance: dict[int, float],
  ) -> None:
    source, target = direction
    balance[source] = balance.get(source, 0.0) - weight
    balance[target] = balance.get(target, 0.0) + weight

  @classmethod
  def _select_merge_direction(
      cls,
      candidates: set[tuple[int, int]],
      weight: float,
      balance: dict[int, float],
  ) -> tuple[int, int]:
    def flow_penalty(direction: tuple[int, int]) -> float:
      source, target = direction
      source_balance = balance.get(source, 0.0)
      target_balance = balance.get(target, 0.0)
      return (
        (source_balance - weight) ** 2
        + (target_balance + weight) ** 2
        - source_balance ** 2
        - target_balance ** 2
      )

    return min(candidates, key=lambda direction: (flow_penalty(direction), direction))

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
    solutions = []
    for sub_graph in sub_graphs:
      solution = self.mr2s_solver.run(sub_graph)
      solutions.append(solution)

    merged_solution = self.merge_solutions(
      solutions=solutions,
      graph=graph
    )
    merged_solution.score = self.score_merged_solution(merged_solution, solutions)
    return merged_solution
