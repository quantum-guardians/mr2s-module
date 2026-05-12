from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from typing import Iterable

from dimod import SampleSet

from mr2s_module import estimate_required_qubits
from mr2s_module.cycle import FaceCycle
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Edge, Graph, GraphPartitionResult, Score, Solution
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.util import empty_binary_sample_set


def _run_subgraph_solution(args: tuple[QuboMR2SSolver, Graph, SampleSet]) -> Solution:
  mr2s_solver, sub_graph, empty_sample_set = args
  return _solve_subgraph(mr2s_solver, sub_graph, empty_sample_set)


def _solve_subgraph(
    mr2s_solver: QuboMR2SSolver,
    sub_graph: Graph,
    empty_sample_set: SampleSet,
) -> Solution:
  if sub_graph.edges and all(edge.directed for edge in sub_graph.edges):
    solution = Solution(
      edges={edge.vertices for edge in sub_graph.edges},
      graph=sub_graph,
      sample_set=empty_sample_set,
    )
    solution.score = mr2s_solver.evaluator.run(solution)
    return solution

  return mr2s_solver.run(sub_graph)


@dataclass
class DnCMr2sSolver:
  mr2s_solver: QuboMR2SSolver
  face_cycle: FaceCycle = field(default_factory=lambda: FaceCycle(target_k=2, clusterer=KMeansFaceClusterer()))
  subgraph_processes: int | None = None

  def __post_init__(self) -> None:
    if self.subgraph_processes is not None and self.subgraph_processes < 1:
      raise ValueError("subgraph_processes must be None or at least 1")

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
      else:
        direction = self._select_merge_direction(candidates, edge.weight, balance)

      merged_edges.add(direction)
      self._apply_flow_balance(direction, edge.weight, balance)

    sample_set = (
      solution_list[0].sample_set
      if solution_list
      else empty_binary_sample_set()
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

  def _resolve_subgraph_processes(self, subgraph_count: int) -> int:
    if subgraph_count < 1:
      return 1
    if self.subgraph_processes is not None:
      return min(self.subgraph_processes, subgraph_count)

    process_cpu_count = getattr(os, "process_cpu_count", None)
    cpu_count = (
      process_cpu_count()
      if process_cpu_count is not None
      else os.cpu_count()
    ) or 1
    return max(1, min(cpu_count, subgraph_count))

  def _solve_subgraphs(self, sub_graphs: list[Graph]) -> list[Solution]:
    process_count = self._resolve_subgraph_processes(len(sub_graphs))
    empty_sample_set = empty_binary_sample_set()
    if process_count == 1:
      return [
        _solve_subgraph(self.mr2s_solver, sub_graph, empty_sample_set)
        for sub_graph in sub_graphs
      ]

    mp_context = (
      multiprocessing.get_context("spawn")
      if os.name == "nt"
      else None
    )
    try:
      with ProcessPoolExecutor(
          max_workers=process_count,
          mp_context=mp_context,
      ) as executor:
        return list(executor.map(
          _run_subgraph_solution,
          [
            (self.mr2s_solver, sub_graph, empty_sample_set)
            for sub_graph in sub_graphs
          ],
        ))
    except (NotImplementedError, OSError, PermissionError):
      return [
        _solve_subgraph(self.mr2s_solver, sub_graph, empty_sample_set)
        for sub_graph in sub_graphs
      ]

  def run(self, graph: Graph) -> Solution:
    sub_graphs = self.divide_graph(graph)
    if len(sub_graphs) == 1 and sub_graphs[0] is graph:
      return self.mr2s_solver.run(graph)

    solutions = self._solve_subgraphs(sub_graphs)

    merged_solution = self.merge_solutions(
      solutions=solutions,
      graph=graph,
    )
    self._apply_merged_directions(graph, merged_solution)
    final_solution = self.mr2s_solver.run(graph)
    final_solution.score = self.score_merged_solution(final_solution, solutions)
    return final_solution
