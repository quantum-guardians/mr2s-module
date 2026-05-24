from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from mr2s_module.domain.adj_entry import AdjEntry
from mr2s_module.domain.edge import Edge



@dataclass
class Graph:
  edges: dict[frozenset[int], Edge] = field(default_factory=dict)

  def __post_init__(self):
    if isinstance(self.edges, dict):
      return
    if isinstance(self.edges, Iterable):
      self.edges = {edge.id: edge for edge in self.edges}
      return
    raise TypeError(
      f"Graph.edges must be a dict or an iterable of Edge, got {type(self.edges)!r}"
    )

  def define_edge_direction(self, predefined_edges: Iterable[Edge]):
    for p_edge in predefined_edges:
      if p_edge.directed:
        self.edges[p_edge.id] = p_edge

  def is_empty(self):
    return len(self.edges) == 0

  def get_vertices(self) -> set[int]:
    return {v for edge in self.edges.values() for v in edge.vertices}

  def get_adjacency_dict(self) -> dict[int, list[AdjEntry]]:
    adj = defaultdict(list)
    for edge in self.edges.values():
      if edge.directed:
        adj[edge.vertices[0]].append(AdjEntry(edge.vertices[1], edge.weight, True))
      else:
        adj[edge.vertices[0]].append(AdjEntry(edge.vertices[1], edge.weight, False))
        adj[edge.vertices[1]].append(AdjEntry(edge.vertices[0], edge.weight, False))
    return dict(adj)
