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

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import (
  Edge,
  EmbeddableGraphPartition,
  EmbeddingEstimate,
  Graph,
  GraphPartitionResult,
  Score,
  Solution,
)
from mr2s_module.protocols import DnCGraphPartitionStrategyProtocol
from mr2s_module.solver.partition import (
  DegeneracyPruningFaceCyclePartitionStrategy,
  EmbeddingAwareFaceCyclePartitionStrategy,
)
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.util import empty_binary_sample_set


logger = logging.getLogger(__name__)


def _elapsed_ms(start_time: float) -> float:
  return (perf_counter() - start_time) * 1000


def _graph_log_context(graph: Graph) -> dict[str, int]:
  return {
    "vertices": len(graph.get_vertices()),
    "edges": len(graph.edges),
    "directed_edges": sum(1 for edge in graph.edges.values() if edge.directed),
  }


def _run_subgraph_solution(
    args: tuple[QuboMR2SSolver, Graph, EmbeddingEstimate | None, SampleSet],
) -> Solution:
  mr2s_solver, sub_graph, embedding_estimate, empty_sample_set = args
  return _solve_subgraph(
    mr2s_solver,
    sub_graph,
    empty_sample_set,
    embedding_estimate,
  )


def _solve_with_reused_embedding(
    mr2s_solver: QuboMR2SSolver,
    graph: Graph,
    embedding_estimate: EmbeddingEstimate | None,
) -> tuple[Solution, bool] | None:
  if embedding_estimate is None or not embedding_estimate.has_physical_embedding:
    return None

  run_with_embedding = getattr(mr2s_solver, "run_with_embedding", None)
  if run_with_embedding is None:
    return None

  try:
    return run_with_embedding(graph, embedding_estimate), True
  except (NotImplementedError, ValueError):
    return None


def _solve_subgraph(
    mr2s_solver: QuboMR2SSolver,
    sub_graph: Graph,
    empty_sample_set: SampleSet,
    embedding_estimate: EmbeddingEstimate | None = None,
) -> Solution:
  started_at = perf_counter()
  graph_context = _graph_log_context(sub_graph)
  logger.info(
    "DnC solve subgraph started vertices=%d edges=%d directed_edges=%d",
    graph_context["vertices"],
    graph_context["edges"],
    graph_context["directed_edges"],
  )
  if sub_graph.edges and all(edge.directed for edge in sub_graph.edges.values()):
    solution = Solution(
      edges={edge.vertices for edge in sub_graph.edges.values()},
      graph=sub_graph,
      sample_set=empty_sample_set,
    )
    solution.score = mr2s_solver.evaluator.run(solution)
    logger.info(
      "DnC solve subgraph skipped QUBO for directed-only graph elapsed_ms=%.3f",
      _elapsed_ms(started_at),
    )
    return solution

  reused_solution = _solve_with_reused_embedding(
    mr2s_solver,
    sub_graph,
    embedding_estimate,
  )
  if reused_solution is not None:
    solution, _ = reused_solution
    logger.info(
      "DnC solve subgraph reused embedding elapsed_ms=%.3f solution_edges=%d",
      _elapsed_ms(started_at),
      len(solution.edges),
    )
    return solution

  if embedding_estimate is not None:
    logger.info(
      "DnC solve subgraph falling back without embedding reuse "
      "has_physical_embedding=%s",
      embedding_estimate.has_physical_embedding,
    )

  solution = mr2s_solver.run(sub_graph)
  logger.info(
    "DnC solve subgraph finished elapsed_ms=%.3f solution_edges=%d",
    _elapsed_ms(started_at),
    len(solution.edges),
  )
  return solution


_EmbeddablePartition = EmbeddableGraphPartition


@dataclass
class DnCSolution(Solution):
  sub_graphs: list[Graph] = field(default_factory=list)
  embedding_estimates: list[EmbeddingEstimate] = field(default_factory=list)
  partition_target_k: int | None = None


@dataclass
class DnCMr2sSolver:
  mr2s_solver: QuboMR2SSolver
  face_cycle: FaceClusterPartition = field(
    default_factory=lambda: FaceClusterPartition(
      target_k=2,
      clusterer=KMeansFaceClusterer(),
    )
  )
  subgraph_processes: int | None = None
  target_graph: nx.Graph = field(default_factory=lambda: dnx.pegasus_graph(16))
  graph_partition_strategy: DnCGraphPartitionStrategyProtocol | None = None
  _owns_graph_partition_strategy: bool = field(default=False, init=False)
  _owned_graph_partition_strategy: DnCGraphPartitionStrategyProtocol | None = field(
    default=None,
    init=False,
  )

  def __post_init__(self) -> None:
    if self.subgraph_processes is not None and self.subgraph_processes < 1:
      raise ValueError("subgraph_processes must be None or at least 1")
    if self.graph_partition_strategy is None:
      self._owns_graph_partition_strategy = True
      self.graph_partition_strategy = DegeneracyPruningFaceCyclePartitionStrategy(
        mr2s_solver=self.mr2s_solver,
        face_cycle=self.face_cycle,
        target_graph=self.target_graph,
      )
      self._owned_graph_partition_strategy = self.graph_partition_strategy

  def _sync_default_partition_strategy(self) -> None:
    if not self._owns_graph_partition_strategy:
      return
    if self.graph_partition_strategy is not self._owned_graph_partition_strategy:
      self._owns_graph_partition_strategy = False
      self._owned_graph_partition_strategy = None
      return
    if not isinstance(
        self.graph_partition_strategy,
        EmbeddingAwareFaceCyclePartitionStrategy,
    ):
      self._owns_graph_partition_strategy = False
      self._owned_graph_partition_strategy = None
      return
    strategy = self.graph_partition_strategy
    strategy.mr2s_solver = self.mr2s_solver
    strategy.face_cycle = self.face_cycle
    strategy.target_graph = self.target_graph
    if hasattr(strategy, "_resolved_target_degeneracy"):
      strategy._resolved_target_degeneracy = None

  def _default_partition_strategy(
      self,
      sync: bool = True,
  ) -> EmbeddingAwareFaceCyclePartitionStrategy:
    if not isinstance(
        self.graph_partition_strategy,
        EmbeddingAwareFaceCyclePartitionStrategy,
    ):
      raise TypeError(
        "This compatibility method requires "
        "EmbeddingAwareFaceCyclePartitionStrategy"
      )
    if sync:
      self._sync_default_partition_strategy()
    return self.graph_partition_strategy

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
    candidate_edges: dict[frozenset[int], set[tuple[int, int]]] = {}
    balance: dict[int, float] = {}

    for solution in solution_list:
      for source, target in solution.edges:
        edge_id = frozenset({source, target})
        candidate_edges.setdefault(edge_id, set()).add((source, target))

    merged_edges: set[tuple[int, int]] = set()
    for edge in graph.edges.values():
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
    return EmbeddingAwareFaceCyclePartitionStrategy._is_progressing_partition(
      parent,
      sub_graphs,
    )

  def _embedding_estimate(self, graph: Graph) -> EmbeddingEstimate | None:
    return self._default_partition_strategy()._embedding_estimate(graph)

  def _can_embed(self, graph: Graph) -> bool:
    return self._embedding_estimate(graph) is not None

  def _estimate_partition(
      self,
      sub_graphs: list[Graph],
  ) -> list[EmbeddingEstimate] | None:
    return self._default_partition_strategy()._estimate_partition(sub_graphs)

  def _single_graph_partition(self, graph: Graph) -> _EmbeddablePartition | None:
    return self._default_partition_strategy()._single_graph_partition(graph)

  def _raise_partition_failed(self, graph: Graph) -> None:
    self._default_partition_strategy()._raise_partition_failed(graph)

  def _divide_graph_with_embeddings(self, graph: Graph) -> _EmbeddablePartition:
    if self.graph_partition_strategy is None:
      raise RuntimeError("graph_partition_strategy is not configured")
    self._sync_default_partition_strategy()
    return self.graph_partition_strategy.run(graph)

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
      partition_target_k=partition.target_k,
    )

  def _partition_with_target_k(
      self,
      graph: Graph,
      target_k: int,
  ) -> GraphPartitionResult:
    return self._default_partition_strategy()._partition_with_target_k(
      graph,
      target_k,
    )

  def _find_partition_by_target_k(self, graph: Graph) -> _EmbeddablePartition | None:
    return self._default_partition_strategy()._find_partition_by_target_k(graph)

  def divide_graph(self, graph: Graph) -> list[Graph]:
    return self._divide_graph_with_embeddings(graph).sub_graphs

  @staticmethod
  def _apply_merged_directions(graph: Graph, solution: Solution) -> None:
    weights_by_edge = {
      edge.id: edge.weight
      for edge in graph.edges.values()
    }
    predefined_edges = {
      Edge(
        source,
        target,
        weights_by_edge[frozenset({source, target})],
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

  def _solve_subgraphs(
      self,
      sub_graphs: list[Graph],
      embedding_estimates: list[EmbeddingEstimate] | None = None,
  ) -> list[Solution]:
    started_at = perf_counter()
    if embedding_estimates is not None and len(embedding_estimates) != len(sub_graphs):
      raise ValueError("embedding_estimates must align with sub_graphs")

    estimates = embedding_estimates or [None] * len(sub_graphs)
    process_count = self._resolve_subgraph_processes(len(sub_graphs))
    empty_sample_set = empty_binary_sample_set()
    logger.info(
      "DnC solve subgraphs started subgraphs=%d processes=%d",
      len(sub_graphs),
      process_count,
    )
    if process_count == 1:
      solutions = [
        _solve_subgraph(
          self.mr2s_solver,
          sub_graph,
          empty_sample_set,
          embedding_estimate,
        )
        for sub_graph, embedding_estimate in zip(sub_graphs, estimates)
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
            (self.mr2s_solver, sub_graph, embedding_estimate, empty_sample_set)
            for sub_graph, embedding_estimate in zip(sub_graphs, estimates)
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
        _solve_subgraph(
          self.mr2s_solver,
          sub_graph,
          empty_sample_set,
          embedding_estimate,
        )
        for sub_graph, embedding_estimate in zip(sub_graphs, estimates)
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
      reused_solution = _solve_with_reused_embedding(
        self.mr2s_solver,
        graph,
        partition.embedding_estimates[0] if partition.embedding_estimates else None,
      )
      direct_solution = (
        reused_solution[0]
        if reused_solution is not None
        else self.mr2s_solver.run(graph)
      )
      solution = self._attach_partition_metadata(
        direct_solution,
        partition,
      )
      logger.info(
        "DnC solver direct solve finished elapsed_ms=%.3f total_elapsed_ms=%.3f",
        _elapsed_ms(direct_started_at),
        _elapsed_ms(started_at),
      )
      return solution

    solve_started_at = perf_counter()
    solutions = self._solve_subgraphs(sub_graphs, partition.embedding_estimates)
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
