from collections import defaultdict
from dataclasses import dataclass

from mr2s_module.domain import AdjEntry
from mr2s_module.protocols import Edge


@dataclass
class Graph:
  edges: list[Edge]

  def define_edge_direction(self, predefined_edges: set[Edge]):
    edge_map = {edge.id: edge for edge in self.edges}

    for p_edge in predefined_edges:
      if p_edge.directed:
        edge_map[p_edge.id] = p_edge

    self.edges[:] = list(edge_map.values())

  def is_empty(self):
    return len(self.edges) == 0

  def get_vertices(self) -> set[int]:
    return {v for edge in self.edges for v in edge.vertices}

  def get_adjacency_dict(self) -> dict[int, list[AdjEntry]]:
    adj = defaultdict(list)
    for edge in self.edges:
      if edge.directed:
        adj[edge.vertices[0]].append(AdjEntry(edge.vertices[1], edge.weight, True))
      else:
        adj[edge.vertices[0]].append(AdjEntry(edge.vertices[1], edge.weight, False))
        adj[edge.vertices[1]].append(AdjEntry(edge.vertices[0], edge.weight, False))
    return dict(adj)
