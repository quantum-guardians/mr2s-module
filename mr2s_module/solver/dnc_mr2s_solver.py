from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
import os
from time import perf_counter
from typing import Iterable

import dwave_networkx as dnx
import networkx as nx
from dimod import SampleSet

from mr2s_module import estimate_required_qubits
from mr2s_module.cycle import FaceCycle
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Edge, EmbeddingEstimate, Graph, GraphPartitionResult, Score, Solution
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.util import empty_binary_sample_set


logger = logging.getLogger(__name__)


def _elapsed_ms(start_time: float) -> float:
  return (perf_counter() - start_time) * 1000


def _graph_log_context(graph: Graph) -> dict[str, int]:
  return {
    "vertices": len(graph.get_vertices()),
    "edges": len(graph.edges),
    "directed_edges": sum(1 for edge in graph.edges if edge.directed),
  }


def _run_subgraph_solution(args: tuple[QuboMR2SSolver, Graph, SampleSet]) -> Solution:
  mr2s_solver, sub_graph, empty_sample_set = args
  return _solve_subgraph(mr2s_solver, sub_graph, empty_sample_set)


def _solve_subgraph(
    mr2s_solver: QuboMR2SSolver,
    sub_graph: Graph,
    empty_sample_set: SampleSet,
) -> Solution:
  started_at = perf_counter()
  graph_context = _graph_log_context(sub_graph)
  logger.info(
    "DnC solve subgraph started vertices=%d edges=%d directed_edges=%d",
    graph_context["vertices"],
    graph_context["edges"],
    graph_context["directed_edges"],
  )
  if sub_graph.edges and all(edge.directed for edge in sub_graph.edges):
    solution = Solution(
      edges={edge.vertices for edge in sub_graph.edges},
      graph=sub_graph,
      sample_set=empty_sample_set,
    )
    solution.score = mr2s_solver.evaluator.run(solution)
    logger.info(
      "DnC solve subgraph skipped QUBO for directed-only graph elapsed_ms=%.3f",
      _elapsed_ms(started_at),
    )
    return solution

  solution = mr2s_solver.run(sub_graph)
  logger.info(
    "DnC solve subgraph finished elapsed_ms=%.3f solution_edges=%d",
    _elapsed_ms(started_at),
    len(solution.edges),
  )
  return solution


@dataclass
class _EmbeddablePartition:
  sub_graphs: list[Graph]
  embedding_estimates: list[EmbeddingEstimate]


@dataclass
class DnCSolution(Solution):
  sub_graphs: list[Graph] = field(default_factory=list)
  embedding_estimates: list[EmbeddingEstimate] = field(default_factory=list)


@dataclass
class DnCMr2sSolver:
  mr2s_solver: QuboMR2SSolver
  face_cycle: FaceCycle = field(default_factory=lambda: FaceCycle(target_k=2, clusterer=KMeansFaceClusterer()))
  subgraph_processes: int | None = None
  target_graph: nx.Graph = field(default_factory=lambda: dnx.pegasus_graph(16))

  def __post_init__(self) -> None:
    if self.subgraph_processes is not None and self.subgraph_processes < 1:
      raise ValueError("subgraph_processes must be None or at least 1")

  def merge_solutions(
      self,
      solutions: Iterable[Solution],
      graph: Graph
  ) -> Solution:
    started_at = perf_counter()
    solution_list = list(solutions)
    logger.info(
      "DnC merge solutions started child_solutions=%d graph_edges=%d",
      len(solution_list),
      len(graph.edges),
    )
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

    merged_solution = Solution(
      edges=merged_edges,
      graph=graph,
      sample_set=sample_set,
      score=None,
    )
    logger.info(
      "DnC merge solutions finished elapsed_ms=%.3f merged_edges=%d",
      _elapsed_ms(started_at),
      len(merged_edges),
    )
    return merged_solution

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
  def _is_progressing_partition(parent: Graph, sub_graphs: list[Graph]) -> bool:
    if not sub_graphs:
      return False

    parent_edge_count = len(parent.edges)
    return all(
      0 < len(sub_graph.edges) < parent_edge_count
      for sub_graph in sub_graphs
    )

  def _embedding_estimate(self, graph: Graph) -> EmbeddingEstimate | None:
    started_at = perf_counter()
    graph_context = _graph_log_context(graph)
    logger.info(
      "DnC embedding estimate started vertices=%d edges=%d directed_edges=%d",
      graph_context["vertices"],
      graph_context["edges"],
      graph_context["directed_edges"],
    )
    target_node_count = self.target_graph.number_of_nodes()
    undirected_edge_count = sum(
      1
      for edge in graph.edges
      if not edge.directed
    )
    if undirected_edge_count > target_node_count:
      logger.info(
        "DnC embedding estimate skipped by edge prefilter "
        "elapsed_ms=%.3f undirected_edges=%d target_nodes=%d",
        _elapsed_ms(started_at),
        undirected_edge_count,
        target_node_count,
      )
      return None

    try:
      build_started_at = perf_counter()
      bqm = self.mr2s_solver.build_bqm(graph)
      logger.info(
        "DnC embedding estimate built BQM elapsed_ms=%.3f variables=%d",
        _elapsed_ms(build_started_at),
        len(bqm.variables),
      )
      if len(bqm.variables) > target_node_count:
        logger.info(
          "DnC embedding estimate skipped by variable prefilter "
          "elapsed_ms=%.3f variables=%d target_nodes=%d",
          _elapsed_ms(started_at),
          len(bqm.variables),
          target_node_count,
        )
        return None
      estimate_started_at = perf_counter()
      estimate = estimate_required_qubits(
        bqm,
        target_graph=self.target_graph,
      )
      logger.info(
        "DnC embedding estimate finished elapsed_ms=%.3f "
        "estimator_elapsed_ms=%.3f physical_qubits=%d max_chain_length=%d",
        _elapsed_ms(started_at),
        _elapsed_ms(estimate_started_at),
        estimate.num_physical_qubits,
        estimate.max_chain_length,
      )
      return estimate
    except RuntimeError:
      logger.info(
        "DnC embedding estimate failed elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return None

  def _can_embed(self, graph: Graph) -> bool:
    return self._embedding_estimate(graph) is not None

  def _estimate_partition(
      self,
      sub_graphs: list[Graph],
  ) -> list[EmbeddingEstimate] | None:
    estimates: list[EmbeddingEstimate] = []
    for sub_graph in sub_graphs:
      estimate = self._embedding_estimate(sub_graph)
      if estimate is None:
        return None
      estimates.append(estimate)
    return estimates

  def _single_graph_partition(self, graph: Graph) -> _EmbeddablePartition | None:
    started_at = perf_counter()
    estimate = self._embedding_estimate(graph)
    if estimate is None:
      logger.info(
        "DnC single-graph partition unavailable elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return None
    logger.info(
      "DnC single-graph partition selected elapsed_ms=%.3f",
      _elapsed_ms(started_at),
    )
    return _EmbeddablePartition(
      sub_graphs=[graph],
      embedding_estimates=[estimate],
    )

  def _raise_partition_failed(self, graph: Graph) -> None:
    raise RuntimeError(
      "DnC partition failed: input graph is not embeddable and no "
      "embeddable subgraph partition was found "
      f"(vertices={len(graph.get_vertices())}, edges={len(graph.edges)})"
    )

  def _divide_graph_with_embeddings(self, graph: Graph) -> _EmbeddablePartition:
    started_at = perf_counter()
    graph_context = _graph_log_context(graph)
    logger.info(
      "DnC divide graph started vertices=%d edges=%d directed_edges=%d",
      graph_context["vertices"],
      graph_context["edges"],
      graph_context["directed_edges"],
    )
    partition = self._single_graph_partition(graph)
    if partition is not None:
      logger.info(
        "DnC divide graph finished with single graph elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return partition

    partition = self._find_partition_by_target_k(graph)
    if partition is None:
      logger.info(
        "DnC divide graph failed elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      self._raise_partition_failed(graph)

    logger.info(
      "DnC divide graph finished elapsed_ms=%.3f subgraphs=%d",
      _elapsed_ms(started_at),
      len(partition.sub_graphs),
    )
    return partition

  def _attach_partition_metadata(
      self,
      solution: Solution,
      partition: _EmbeddablePartition,
  ) -> DnCSolution:
    return DnCSolution(
      edges=solution.edges,
      graph=solution.graph,
      sample_set=solution.sample_set,
      score=solution.score,
      sub_graphs=partition.sub_graphs,
      embedding_estimates=partition.embedding_estimates,
    )

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

  def _find_partition_by_target_k(self, graph: Graph) -> _EmbeddablePartition | None:
    started_at = perf_counter()
    left = 2
    right = max(2, len(graph.edges))
    best_partition: _EmbeddablePartition | None = None
    logger.info(
      "DnC target_k search started left=%d right=%d graph_edges=%d",
      left,
      right,
      len(graph.edges),
    )

    while left <= right:
      target_k = (left + right) // 2
      attempt_started_at = perf_counter()
      logger.info(
        "DnC target_k attempt started target_k=%d left=%d right=%d",
        target_k,
        left,
        right,
      )
      result = self._partition_with_target_k(graph, target_k)
      partition_elapsed_ms = _elapsed_ms(attempt_started_at)
      sub_graphs = result.sub_graphs
      estimate_started_at = perf_counter()
      embedding_estimates = (
        self._estimate_partition(sub_graphs)
        if self._is_progressing_partition(graph, sub_graphs)
        else None
      )
      estimate_elapsed_ms = _elapsed_ms(estimate_started_at)

      if embedding_estimates is not None:
        best_partition = _EmbeddablePartition(
          sub_graphs=sub_graphs,
          embedding_estimates=embedding_estimates,
        )
        logger.info(
          "DnC target_k attempt succeeded target_k=%d subgraphs=%d "
          "partition_elapsed_ms=%.3f estimate_elapsed_ms=%.3f",
          target_k,
          len(sub_graphs),
          partition_elapsed_ms,
          estimate_elapsed_ms,
        )
        right = target_k - 1
      else:
        logger.info(
          "DnC target_k attempt failed target_k=%d subgraphs=%d "
          "partition_elapsed_ms=%.3f estimate_elapsed_ms=%.3f",
          target_k,
          len(sub_graphs),
          partition_elapsed_ms,
          estimate_elapsed_ms,
        )
        left = target_k + 1

    logger.info(
      "DnC target_k search finished elapsed_ms=%.3f found=%s",
      _elapsed_ms(started_at),
      best_partition is not None,
    )
    return best_partition

  def divide_graph(self, graph: Graph) -> list[Graph]:
    return self._divide_graph_with_embeddings(graph).sub_graphs

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
    started_at = perf_counter()
    child_solution_list = list(child_solutions)
    logger.info(
      "DnC score merged solution started child_solutions=%d",
      len(child_solution_list),
    )
    score = self.mr2s_solver.evaluator.run(merged_solution)
    strong_connect_rate = 1.0

    for child_solution in child_solution_list:
      if child_solution.score is None:
        child_solution.score = self.mr2s_solver.evaluator.run(child_solution)
      strong_connect_rate *= child_solution.score.strong_connect_rate

    score.strong_connect_rate = strong_connect_rate
    logger.info(
      "DnC score merged solution finished elapsed_ms=%.3f "
      "strong_connect_rate=%.6f",
      _elapsed_ms(started_at),
      score.strong_connect_rate,
    )
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
    started_at = perf_counter()
    process_count = self._resolve_subgraph_processes(len(sub_graphs))
    empty_sample_set = empty_binary_sample_set()
    logger.info(
      "DnC solve subgraphs started subgraphs=%d processes=%d",
      len(sub_graphs),
      process_count,
    )
    if process_count == 1:
      solutions = [
        _solve_subgraph(self.mr2s_solver, sub_graph, empty_sample_set)
        for sub_graph in sub_graphs
      ]
      logger.info(
        "DnC solve subgraphs finished sequential elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return solutions

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
        solutions = list(executor.map(
          _run_subgraph_solution,
          [
            (self.mr2s_solver, sub_graph, empty_sample_set)
            for sub_graph in sub_graphs
          ],
        ))
        logger.info(
          "DnC solve subgraphs finished parallel elapsed_ms=%.3f",
          _elapsed_ms(started_at),
        )
        return solutions
    except (NotImplementedError, OSError, PermissionError):
      logger.info(
        "DnC solve subgraphs falling back to sequential processes=%d",
        process_count,
      )
      solutions = [
        _solve_subgraph(self.mr2s_solver, sub_graph, empty_sample_set)
        for sub_graph in sub_graphs
      ]
      logger.info(
        "DnC solve subgraphs finished fallback elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return solutions

  def run(self, graph: Graph) -> DnCSolution:
    started_at = perf_counter()
    logger.info("DnC solver run started")
    partition_started_at = perf_counter()
    partition = self._divide_graph_with_embeddings(graph)
    logger.info(
      "DnC solver partition phase finished elapsed_ms=%.3f subgraphs=%d",
      _elapsed_ms(partition_started_at),
      len(partition.sub_graphs),
    )
    sub_graphs = partition.sub_graphs
    if len(sub_graphs) == 1 and sub_graphs[0] is graph:
      direct_started_at = perf_counter()
      solution = self._attach_partition_metadata(
        self.mr2s_solver.run(graph),
        partition,
      )
      logger.info(
        "DnC solver direct solve finished elapsed_ms=%.3f total_elapsed_ms=%.3f",
        _elapsed_ms(direct_started_at),
        _elapsed_ms(started_at),
      )
      return solution

    solve_started_at = perf_counter()
    solutions = self._solve_subgraphs(sub_graphs)
    logger.info(
      "DnC solver subgraph solve phase finished elapsed_ms=%.3f",
      _elapsed_ms(solve_started_at),
    )

    merge_started_at = perf_counter()
    merged_solution = self.merge_solutions(
      solutions=solutions,
      graph=graph,
    )
    logger.info(
      "DnC solver merge phase finished elapsed_ms=%.3f",
      _elapsed_ms(merge_started_at),
    )
    apply_started_at = perf_counter()
    self._apply_merged_directions(graph, merged_solution)
    logger.info(
      "DnC solver apply directions phase finished elapsed_ms=%.3f",
      _elapsed_ms(apply_started_at),
    )
    final_solve_started_at = perf_counter()
    final_solution = self.mr2s_solver.run(graph)
    logger.info(
      "DnC solver final solve phase finished elapsed_ms=%.3f",
      _elapsed_ms(final_solve_started_at),
    )
    score_started_at = perf_counter()
    final_solution.score = self.score_merged_solution(final_solution, solutions)
    logger.info(
      "DnC solver score phase finished elapsed_ms=%.3f",
      _elapsed_ms(score_started_at),
    )
    solution = self._attach_partition_metadata(final_solution, partition)
    logger.info(
      "DnC solver run finished total_elapsed_ms=%.3f",
      _elapsed_ms(started_at),
    )
    return solution
