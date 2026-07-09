from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from mr2s_module.domain.adj_entry import AdjEntry
from mr2s_module.domain.edge import Edge


@dataclass
class Graph:
  edges: dict[int, Edge] = field(default_factory=dict)

  def __post_init__(self):
    if isinstance(self.edges, dict):
      return
    if isinstance(self.edges, Iterable) and not isinstance(self.edges, str):
      self.edges = {edge.id: edge for edge in self.edges}
      return
    raise TypeError(
      f"Graph.edges must be a dict or an iterable of Edge, got {type(self.edges)!r}"
    )

  def define_edge_direction(self, predefined_edges: Iterable[Edge]):
    """외부 orienter 결과를 id 로 찾아 방향을 in-place 로 박는다."""
    for p_edge in predefined_edges:
      if not p_edge.directed:
        continue
      tail, head = p_edge.vertices
      target = self.edges.get(p_edge.id)
      if target is None:
        raise ValueError(f"Directed edge id {p_edge.id} is not in this graph.")
      target.set_direction(tail, head)

  def is_empty(self):
    return len(self.edges) == 0

  def get_vertices(self) -> set[int]:
    return {v for edge in self.edges.values() for v in edge.vertices}

  def get_adjacency_dict(self) -> dict[int, list[AdjEntry]]:
    adj = defaultdict(list)
    for edge in self.edges.values():
      if edge.directed:
        adj[edge.vertices[0]].append(
          AdjEntry(edge.vertices[1], edge.weight, True, edge.id)
        )
      else:
        adj[edge.vertices[0]].append(
          AdjEntry(edge.vertices[1], edge.weight, False, edge.id)
        )
        adj[edge.vertices[1]].append(
          AdjEntry(edge.vertices[0], edge.weight, False, edge.id)
        )
    return dict(adj)
