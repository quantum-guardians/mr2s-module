import logging
from dataclasses import dataclass
from time import perf_counter

from mr2s_module.domain import EmbeddableGraphPartition, Graph
from mr2s_module.protocols import TunableFaceCycleProtocol

logger = logging.getLogger(__name__)


def _elapsed_ms(started_at: float) -> float:
  return (perf_counter() - started_at) * 1000


@dataclass
class VertexCountPartitionStrategy:
  face_cycle: TunableFaceCycleProtocol
  max_vertices: int = 100

  def run(self, graph: Graph) -> EmbeddableGraphPartition:
    started_at = perf_counter()
    vertices_count = len(graph.get_vertices())
    logger.info(
      "VertexCount partition started vertices=%d max_vertices=%d",
      vertices_count,
      self.max_vertices,
    )

    if vertices_count <= self.max_vertices:
      logger.info(
        "VertexCount partition finished with single graph elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      return EmbeddableGraphPartition(
        sub_graphs=[graph],
        embedding_estimates=[None],
        target_k=None,
        solve_contexts=[None],
      )

    left = 2
    right = max(2, len(graph.edges))
    best_partition: EmbeddableGraphPartition | None = None

    while left <= right:
      target_k = (left + right) // 2
      attempt_started_at = perf_counter()
      logger.info(
        "VertexCount target_k attempt started target_k=%d left=%d right=%d",
        target_k,
        left,
        right,
      )

      previous_target_k = self.face_cycle.target_k
      self.face_cycle.target_k = target_k

      try:
        result = self.face_cycle.run(graph)
      finally:
        self.face_cycle.target_k = previous_target_k

      sub_graphs = result.sub_graphs
      parent_edge_count = len(graph.edges)

      is_valid = False
      if sub_graphs:
        is_progressing = all(
          0 < len(sub_graph.edges) < parent_edge_count
          for sub_graph in sub_graphs
        )
        if is_progressing:
          is_valid = all(
            len(sub_graph.get_vertices()) <= self.max_vertices
            for sub_graph in sub_graphs
          )

      if is_valid:
        best_partition = EmbeddableGraphPartition(
          sub_graphs=sub_graphs,
          embedding_estimates=[None] * len(sub_graphs),
          target_k=target_k,
          solve_contexts=[None] * len(sub_graphs),
        )
        logger.info(
          "VertexCount target_k attempt succeeded target_k=%d subgraphs=%d elapsed_ms=%.3f",
          target_k,
          len(sub_graphs),
          _elapsed_ms(attempt_started_at),
        )
        right = target_k - 1
      else:
        logger.info(
          "VertexCount target_k attempt failed target_k=%d subgraphs=%d elapsed_ms=%.3f",
          target_k,
          len(sub_graphs),
          _elapsed_ms(attempt_started_at),
        )
        left = target_k + 1

    if best_partition is None:
      logger.info(
        "VertexCount partition failed elapsed_ms=%.3f",
        _elapsed_ms(started_at),
      )
      raise RuntimeError(
        "DnC partition failed: could not divide graph to have <= "
        f"{self.max_vertices} vertices (vertices={vertices_count})"
      )

    logger.info(
      "VertexCount partition finished elapsed_ms=%.3f subgraphs=%d",
      _elapsed_ms(started_at),
      len(best_partition.sub_graphs),
    )
    return best_partition
