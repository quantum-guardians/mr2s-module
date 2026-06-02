import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import networkx as nx

from mr2s_module.domain import EmbeddingEstimate, Graph
from mr2s_module.solver.partition.embedding_aware import (
  EmbeddingAwareFaceCyclePartitionStrategy,
  _elapsed_ms,
  _graph_log_context,
)
from mr2s_module.solver.solve_context import QuboSolveContext


logger = logging.getLogger(__name__)


@dataclass
class DegeneracyPruningFaceCyclePartitionStrategy(
    EmbeddingAwareFaceCyclePartitionStrategy,
):
  max_degeneracy: int | None = None
  _resolved_target_degeneracy: int | None = field(default=None, init=False)

  def _embedding_context(self, graph: Graph) -> QuboSolveContext | None:
    started_at = perf_counter()
    graph_context = _graph_log_context(graph)
    logger.info(
      "DnC pruning estimate started vertices=%d edges=%d directed_edges=%d",
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
      build_elapsed_ms = _elapsed_ms(build_started_at)
    except RuntimeError:
      logger.info(
        "DnC pruning estimate failed to build BQM elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return None

    target_node_count = target_graph.number_of_nodes()
    undirected_edge_count = sum(1 for edge in graph.edges.values() if not edge.directed)
    if undirected_edge_count > target_node_count:
      logger.info(
        "DnC pruning estimate skipped by edge prefilter "
        "elapsed_ms=%.3f undirected_edges=%d target_nodes=%d",
        _elapsed_ms(started_at),
        undirected_edge_count,
        target_node_count,
      )
      return None

    variable_count = len(bqm.variables)
    if variable_count > target_node_count:
      logger.info(
        "DnC pruning estimate skipped by variable prefilter "
        "elapsed_ms=%.3f variables=%d target_nodes=%d",
        _elapsed_ms(started_at),
        variable_count,
        target_node_count,
      )
      return None

    interaction_graph = self._build_interaction_graph(graph, bqm)
    quadratic_count = interaction_graph.number_of_edges()
    target_edge_count = target_graph.number_of_edges()
    if quadratic_count > target_edge_count:
      logger.info(
        "DnC pruning estimate skipped by coupling prefilter "
        "elapsed_ms=%.3f couplings=%d target_edges=%d",
        _elapsed_ms(started_at),
        quadratic_count,
        target_edge_count,
      )
      return None

    degeneracy_started_at = perf_counter()
    degeneracy = self._estimate_degeneracy(interaction_graph)
    max_degeneracy = self._target_degeneracy()
    if degeneracy > max_degeneracy:
      logger.info(
        "DnC pruning estimate skipped by degeneracy prefilter "
        "elapsed_ms=%.3f degeneracy_elapsed_ms=%.3f "
        "degeneracy=%d max_degeneracy=%d",
        _elapsed_ms(started_at),
        _elapsed_ms(degeneracy_started_at),
        degeneracy,
        max_degeneracy,
      )
      return None

    logger.info(
      "DnC pruning estimate accepted elapsed_ms=%.3f build_bqm_ms=%.3f "
      "degeneracy_elapsed_ms=%.3f variables=%d couplings=%d "
      "degeneracy=%d max_degeneracy=%d",
      _elapsed_ms(started_at),
      build_elapsed_ms,
      _elapsed_ms(degeneracy_started_at),
      variable_count,
      quadratic_count,
      degeneracy,
      max_degeneracy,
    )
    context.embedding_estimate = EmbeddingEstimate(
      num_logical_variables=variable_count,
      num_quadratic_couplings=quadratic_count,
      num_physical_qubits=variable_count,
      max_chain_length=degeneracy,
      embedding={variable: [] for variable in bqm.variables},
    )
    return context

  def _embedding_estimate(self, graph: Graph) -> EmbeddingEstimate | None:
    context = self._embedding_context(graph)
    if context is None:
      return None
    return context.embedding_estimate

  @staticmethod
  def _build_interaction_graph(graph: Graph, bqm: Any) -> nx.Graph:
    interaction_graph = nx.Graph()
    interaction_graph.add_nodes_from(bqm.variables)

    quadratic = getattr(bqm, "quadratic", None)
    if quadratic is not None:
      interaction_graph.add_edges_from(quadratic)
      return interaction_graph

    bqm_edges = getattr(bqm, "edges", None)
    edges = bqm_edges if bqm_edges is not None else list(graph.edges.values())
    for edge in edges:
      if hasattr(edge, "endpoints"):
        interaction_graph.add_edge(*edge.endpoints())
      else:
        source, target = edge
        interaction_graph.add_edge(source, target)
    return interaction_graph

  @staticmethod
  def _estimate_degeneracy(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
      return 0
    core_numbers = nx.core_number(graph)
    return max(core_numbers.values(), default=0)

  def _target_degeneracy(self) -> int:
    if self.max_degeneracy is not None:
      return self.max_degeneracy
    if self._resolved_target_degeneracy is None:
      self._resolved_target_degeneracy = self._estimate_degeneracy(
        nx.Graph(self.target_graph or self._fallback_target_graph())
      )
    return self._resolved_target_degeneracy
