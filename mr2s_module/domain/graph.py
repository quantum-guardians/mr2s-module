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

  def edge_by_pair(self, u: int, v: int) -> Edge | None:
    """끝점 (u,v) 로 간선 룩업(단순그래프 뷰). 평행 간선이면 첫 매치."""
    key = frozenset({u, v})
    for edge in self.edges.values():
      if edge.pair_key() == key:
        return edge
    return None

  def edges_by_pair(self, u: int, v: int) -> list[Edge]:
    """끝점 (u,v) 의 모든 간선(평행 간선 포함)."""
    key = frozenset({u, v})
    return [edge for edge in self.edges.values() if edge.pair_key() == key]

  def define_edge_direction(self, predefined_edges: Iterable[Edge]):
    """외부 orienter 결과(독립 directed Edge)를 끝점으로 찾아 방향을 in-place 로 박는다.

    predefined 는 nx 기반 orienter 가 (u,v) 만 알고 만든 별개 Edge 라 자체 id 는 무관.
    원본 graph 간선을 끝점으로 찾아(set_direction) 정체성(id)을 유지한다.
    """
    for p_edge in predefined_edges:
      if not p_edge.directed:
        continue
      tail, head = p_edge.vertices
      target = self.edge_by_pair(tail, head)
      if target is None:
        continue
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
