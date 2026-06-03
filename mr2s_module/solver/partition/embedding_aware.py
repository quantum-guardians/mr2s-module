import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

import networkx as nx
import dwave_networkx as dnx

from mr2s_module.domain import (
  EmbeddableGraphPartition,
  EmbeddingEstimate,
  Graph,
  GraphPartitionResult,
)
from mr2s_module.protocols import FaceCycleProtocol
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver
from mr2s_module.solver.solve_context import QuboSolveContext
from mr2s_module.util import estimate_required_qubits


logger = logging.getLogger(__name__)

EmbeddingEstimator = Callable[..., EmbeddingEstimate]


def _elapsed_ms(start_time: float) -> float:
  return (perf_counter() - start_time) * 1000


def _graph_log_context(graph: Graph) -> dict[str, int]:
  return {
    "vertices": len(graph.get_vertices()),
    "edges": len(graph.edges),
    "directed_edges": sum(1 for edge in graph.edges.values() if edge.directed),
  }


@dataclass
class EmbeddingAwareFaceCyclePartitionStrategy:
  mr2s_solver: QuboMR2SSolver
  face_cycle: FaceCycleProtocol
  target_graph: nx.Graph | None
  embedding_estimator: EmbeddingEstimator = estimate_required_qubits
  _fallback_target_graph_cache: nx.Graph | None = field(default=None, init=False)

  @staticmethod
  def _is_progressing_partition(parent: Graph, sub_graphs: list[Graph]) -> bool:
    if not sub_graphs:
      return False

    parent_edge_count = len(parent.edges)
    return all(
      0 < len(sub_graph.edges) < parent_edge_count
      for sub_graph in sub_graphs
    )

  def _fallback_target_graph(self) -> nx.Graph:
    if self._fallback_target_graph_cache is None:
      self._fallback_target_graph_cache = dnx.pegasus_graph(16)
    return self._fallback_target_graph_cache

  def _build_solve_context(self, graph: Graph) -> QuboSolveContext:
    build_solve_context = getattr(self.mr2s_solver, "build_solve_context", None)
    if build_solve_context is not None:
      context = build_solve_context(graph, target_graph=self.target_graph)
      if context.target_graph is None:
        context.target_graph = self._fallback_target_graph()
      return context

    return QuboSolveContext(
      graph=graph,
      bqm=self.mr2s_solver.build_bqm(graph),
      target_graph=self.target_graph or self._fallback_target_graph(),
    )

  def _embedding_context(self, graph: Graph) -> QuboSolveContext | None:
    started_at = perf_counter()
    graph_context = _graph_log_context(graph)
    logger.info(
      "DnC embedding estimate started vertices=%d edges=%d directed_edges=%d",
      graph_context["vertices"],
      graph_context["edges"],
      graph_context["directed_edges"],
    )
    try:
      build_started_at = perf_counter()
      context = self._build_solve_context(graph)
      bqm = context.bqm
      target_graph = context.target_graph or self._fallback_target_graph()
      context.target_graph = target_graph
      logger.info(
        "DnC embedding estimate built BQM elapsed_ms=%.3f variables=%d",
        _elapsed_ms(build_started_at),
        len(bqm.variables),
      )
    except RuntimeError:
      logger.info(
        "DnC embedding estimate failed elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return None

    target_node_count = target_graph.number_of_nodes()
    undirected_edge_count = sum(
      1
      for edge in graph.edges.values()
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
      estimate = self.embedding_estimator(
        bqm,
        target_graph=target_graph,
      )
      context.embedding_estimate = estimate
      logger.info(
        "DnC embedding estimate finished elapsed_ms=%.3f "
        "estimator_elapsed_ms=%.3f physical_qubits=%d max_chain_length=%d",
        _elapsed_ms(started_at),
        _elapsed_ms(estimate_started_at),
        estimate.num_physical_qubits,
        estimate.max_chain_length,
      )
      return context
    except RuntimeError:
      logger.info(
        "DnC embedding estimate failed elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return None

  def _embedding_estimate(self, graph: Graph) -> EmbeddingEstimate | None:
    context = self._embedding_context(graph)
    if context is None:
      return None
    return context.embedding_estimate

  def can_embed(self, graph: Graph) -> bool:
    return self._embedding_estimate(graph) is not None

  def _estimate_partition(
      self,
      sub_graphs: list[Graph],
  ) -> list[EmbeddingEstimate] | None:
    contexts = self._estimate_partition_contexts(sub_graphs)
    if contexts is None:
      return None
    return [context.embedding_estimate for context in contexts]

  def _estimate_partition_contexts(
      self,
      sub_graphs: list[Graph],
  ) -> list[QuboSolveContext] | None:
    contexts: list[QuboSolveContext] = []
    for sub_graph in sub_graphs:
      context = self._embedding_context(sub_graph)
      if context is None or context.embedding_estimate is None:
        return None
      contexts.append(context)
    return contexts

  def _single_graph_partition(self, graph: Graph) -> EmbeddableGraphPartition | None:
    started_at = perf_counter()
    context = self._embedding_context(graph)
    if context is None or context.embedding_estimate is None:
      logger.info(
        "DnC single-graph partition unavailable elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return None
    logger.info(
      "DnC single-graph partition selected elapsed_ms=%.3f",
      _elapsed_ms(started_at),
    )
    return EmbeddableGraphPartition(
      sub_graphs=[graph],
      embedding_estimates=[context.embedding_estimate],
      target_k=None,
      solve_contexts=[context],
    )

  @staticmethod
  def _raise_partition_failed(graph: Graph) -> None:
    raise RuntimeError(
      "DnC partition failed: input graph is not embeddable and no "
      "embeddable subgraph partition was found "
      f"(vertices={len(graph.get_vertices())}, edges={len(graph.edges)})"
    )

  def run(self, graph: Graph) -> EmbeddableGraphPartition:
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

  def _find_partition_by_target_k(
      self,
      graph: Graph,
  ) -> EmbeddableGraphPartition | None:
    started_at = perf_counter()
    left = 2
    right = max(2, len(graph.edges))
    best_partition: EmbeddableGraphPartition | None = None
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
      solve_contexts = (
        self._estimate_partition_contexts(sub_graphs)
        if self._is_progressing_partition(graph, sub_graphs)
        else None
      )
      estimate_elapsed_ms = _elapsed_ms(estimate_started_at)

      if solve_contexts is not None:
        best_partition = EmbeddableGraphPartition(
          sub_graphs=sub_graphs,
          embedding_estimates=[context.embedding_estimate for context in solve_contexts],
          target_k=target_k,
          solve_contexts=solve_contexts,
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
