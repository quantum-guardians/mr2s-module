from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from mr2s_module.domain.adj_entry import AdjEntry
from mr2s_module.domain.edge import Edge



@dataclass
class Graph:
  edges: dict[int, Edge] = field(default_factory=dict)

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
    # Edge.id 가 인스턴스 고유 int 라, predefined directed 간선을 그냥 추가하면 기존 무방향
    # 간선을 대체하지 못하고 평행 추가된다. 같은 양끝점의 기존 간선을 제거한 뒤 삽입한다.
    # 양끝점→id 인덱스를 1회만 만들어 predefined 마다 O(E) 스캔하지 않는다(O(P·E)→O(E)).
    stale_by_key: dict[frozenset[int], list[int]] = {}
    for eid, edge in self.edges.items():
      stale_by_key.setdefault(edge.endpoint_key(), []).append(eid)

    for p_edge in predefined_edges:
      if not p_edge.directed:
        continue
      key = p_edge.endpoint_key()
      for eid in stale_by_key.get(key, []):
        self.edges.pop(eid, None)
      # 같은 양끝점의 다음 predefined 가 방금 삽입한 간선을 덮도록 인덱스를 갱신(last-wins).
      stale_by_key[key] = [p_edge.id]
      self.edges[p_edge.id] = p_edge

  def edge_for_endpoints(self, u: int, v: int) -> Edge | None:
    """양끝점 {u, v} 를 잇는 간선 하나를 반환 (없으면 None).

    평행간선이 없다고 가정하는 지점(체인 축약·euler 전개 등)에서 frozenset 키 조회를 대체.
    """
    key = frozenset({u, v})
    for edge in self.edges.values():
      if edge.endpoint_key() == key:
        return edge
    return None

  def has_endpoint_edge(self, u: int, v: int) -> bool:
    key = frozenset({u, v})
    return any(edge.endpoint_key() == key for edge in self.edges.values())

  def endpoint_index(self) -> dict[frozenset[int], Edge]:
    """양끝점 키 → Edge 인덱스 (1회 빌드, O(E)). 평행간선은 last-wins.

    int id 키잉 이후 양끝점 조회는 선형스캔(edge_for_endpoints)이라, 같은 그래프에
    양끝점 조회를 반복하는 핫 경로(euler orientation·체인 축약)는 이 인덱스를 1회만
    만들어 O(1) 조회로 O(E²)→O(E) 로 낮춘다. Graph 가 가변이므로 영속 캐시는 두지
    않고 호출부가 필요 시점에 1회 빌드한다(staleness 회피).
    """
    return {edge.endpoint_key(): edge for edge in self.edges.values()}

  def is_empty(self):
    return len(self.edges) == 0

  def get_vertices(self) -> set[int]:
    return {v for edge in self.edges.values() for v in edge.vertices}

  def simple_adjacency(self) -> dict[int, set[int]]:
    """무방향 단순 인접 집합 (self-loop·중복 무시). 차수 = 이웃 수."""
    neighbors: dict[int, set[int]] = {}
    for edge in self.edges.values():
      u, v = edge.endpoints()
      if u == v:
        continue
      neighbors.setdefault(u, set()).add(v)
      neighbors.setdefault(v, set()).add(u)
    return neighbors

  def get_adjacency_dict(self) -> dict[int, list[AdjEntry]]:
    adj = defaultdict(list)
    for edge in self.edges.values():
      key = edge.to_key()
      if edge.directed:
        adj[edge.vertices[0]].append(AdjEntry(edge.vertices[1], edge.weight, True, key))
      else:
        adj[edge.vertices[0]].append(AdjEntry(edge.vertices[1], edge.weight, False, key))
        adj[edge.vertices[1]].append(AdjEntry(edge.vertices[0], edge.weight, False, key))
    return dict(adj)
