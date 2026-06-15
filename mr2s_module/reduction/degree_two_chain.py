"""차수-2 정점 체인을 단일 간선으로 축약하고, 해를 원본으로 복원하는 전처리.

MR2S edge-orientation 에서 차수가 2 인 정점은 강연결+흐름보존 제약 때문에 두 간선의
방향이 항상 일관되게 강제된다 (a→x→b 또는 b→x→a). 이런 정점이 연속된 체인 K 는 1비트로
방향이 결정되므로, QUBO 변수만 늘리는 낭비다. 본 모듈은 그런 체인을 양 끝 정점 a, b 사이의
단일 무방향 간선(가중치 = 체인 간선 가중치 합)으로 축약하고, 축약 그래프의 해(directed)를
다시 체인 전체로 펼쳐 원본 그래프의 방향 결과를 복원한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mr2s_module.domain import Edge, Graph


@dataclass(frozen=True)
class CollapsedChain:
  """축약된 차수-2 정점 체인 하나의 기록.

  - endpoints: 양 끝 정점 (a, b), a < b 로 정규화.
  - path: 체인 전체 정점 순서 [a, x1, …, xk, b].
  - original_edges: path 를 따라가는 원본 무방향 간선들 (순서 보존).
  - collapsed_weight: 축약 간선 가중치 = 원본 간선 가중치 합 (APSP 거리 보존).
  """

  endpoints: tuple[int, int]
  path: tuple[int, ...]
  original_edges: tuple[Edge, ...]
  collapsed_weight: int

  @property
  def length(self) -> int:
    """체인의 간선 수 (= 내부 정점 수 + 1)."""
    return len(self.original_edges)


@dataclass
class ChainReductionResult:
  """`DegreeTwoChainReducer.reduce` 의 결과.

  - reduced_graph: 체인이 단일 간선으로 축약된 무방향 그래프.
  - chains: 축약된 체인 기록들. `{a,b}` id 로 축약 간선과 매칭된다.
  """

  reduced_graph: Graph
  chains: tuple[CollapsedChain, ...]

  def expand(self, oriented_edges: Iterable[Edge]) -> list[Edge]:
    """축약 그래프의 directed 해를 원본 그래프 위의 directed 간선들로 펼친다.

    축약 간선(`{a,b}`)의 방향에 맞춰 체인을 a→x1→…→b (또는 역방향)로 전개하고,
    체인에 속하지 않은 directed 간선은 그대로 통과시킨다.
    """
    chain_by_id = {
      frozenset(chain.endpoints): chain for chain in self.chains
    }
    expanded: list[Edge] = []
    for edge in oriented_edges:
      chain = chain_by_id.get(edge.id)
      if chain is None:
        expanded.append(edge)
        continue
      expanded.extend(_orient_chain(chain, edge))
    return expanded


def _orient_chain(chain: CollapsedChain, collapsed_edge: Edge) -> list[Edge]:
  """축약 간선의 방향(tail→head)에 맞춰 체인 path 를 directed 간선들로 전개."""
  tail, head = collapsed_edge.vertices
  a, b = chain.endpoints
  # path 는 [a, …, b] 순. tail 이 b 이면 역방향으로 전개한다.
  vertices = chain.path if tail == a else tuple(reversed(chain.path))
  weight_by_id = {edge.id: edge.weight for edge in chain.original_edges}
  oriented: list[Edge] = []
  for u, v in zip(vertices, vertices[1:]):
    weight = weight_by_id[frozenset({u, v})]
    oriented.append(Edge(u, v, weight, True))
  return oriented


class DegreeTwoChainReducer:
  """차수-2 정점 체인을 단일 간선으로 축약하는 전처리기.

  min_internal_vertices: 축약 대상으로 삼을 체인의 최소 내부(차수-2) 정점 수. 기본 2
    (스펙: 차수-2 정점이 2개 이상 연결된 경우만 축약). 1 로 두면 단일 차수-2 정점도 축약.
  """

  def __init__(self, min_internal_vertices: int = 2):
    if min_internal_vertices < 1:
      raise ValueError("min_internal_vertices must be >= 1")
    self.min_internal_vertices = min_internal_vertices

  def reduce(self, graph: Graph) -> ChainReductionResult:
    neighbors = _build_simple_adjacency(graph)
    degree_two = {v for v, nbrs in neighbors.items() if len(nbrs) == 2}

    chains: list[CollapsedChain] = []
    collapsed_edge_ids: set[frozenset[int]] = set()
    visited_internal: set[int] = set()
    # 이미 채택된 체인이 점유한 끝점 쌍. 평행 체인(같은 a,b)이 단일 간선 키를 덮어쓰는 것을 방지.
    reserved_endpoint_ids: set[frozenset[int]] = set()

    for start in sorted(degree_two):
      if start in visited_internal:
        continue
      internal = _walk_chain(start, neighbors, degree_two)
      visited_internal.update(internal)

      chain = self._build_chain(internal, neighbors, graph)
      if chain is None:
        continue
      endpoint_id = frozenset(chain.endpoints)
      if endpoint_id in reserved_endpoint_ids:
        continue  # 같은 끝점을 가진 체인이 이미 축약됨 → 원본 간선 유지.
      reserved_endpoint_ids.add(endpoint_id)
      chains.append(chain)
      collapsed_edge_ids.update(edge.id for edge in chain.original_edges)

    reduced_edges = [
      edge for edge in graph.edges.values() if edge.id not in collapsed_edge_ids
    ]
    for chain in chains:
      a, b = chain.endpoints
      reduced_edges.append(Edge(a, b, chain.collapsed_weight, False))

    return ChainReductionResult(
      reduced_graph=Graph(edges=reduced_edges),
      chains=tuple(chains),
    )

  def _build_chain(
    self,
    internal: list[int],
    neighbors: dict[int, set[int]],
    graph: Graph,
  ) -> CollapsedChain | None:
    """차수-2 정점들의 한 경로 성분으로부터 축약 가능한 체인을 구성. 불가하면 None."""
    if len(internal) < self.min_internal_vertices:
      return None

    internal_set = set(internal)
    # 경로의 외부 연결점: (내부 끝 정점, 외부 끝점) 쌍. 단순 경로는 정확히 2개여야 한다
    # (단일 내부 정점은 한 정점이 외부 이웃 2개를 가져 2개 쌍을 만든다).
    attachments = [
      (v, n) for v in internal for n in neighbors[v] if n not in internal_set
    ]
    if len(attachments) != 2:
      # 고립 사이클(외부 이웃 없음) 등 단순 경로가 아닌 경우 → 축약 불가.
      return None

    (head, a), (tail, b) = attachments
    if a == b:
      return None  # 양 끝이 같은 정점 → self-loop 회피.
    if frozenset({a, b}) in graph.edges:
      return None  # 직접 간선 존재 → 평행간선/dict 키 충돌 회피.

    ordered_internal = _order_path(head, tail, neighbors, internal_set)
    path = (a, *ordered_internal, b)
    original_edges = tuple(
      graph.edges[frozenset({u, v})] for u, v in zip(path, path[1:])
    )
    collapsed_weight = sum(edge.weight for edge in original_edges)
    endpoints = (a, b) if a < b else (b, a)
    if endpoints != (a, b):
      path = tuple(reversed(path))
      original_edges = tuple(reversed(original_edges))
    return CollapsedChain(
      endpoints=endpoints,
      path=path,
      original_edges=original_edges,
      collapsed_weight=collapsed_weight,
    )


def _build_simple_adjacency(graph: Graph) -> dict[int, set[int]]:
  """무방향 단순 인접 집합 (self-loop·중복 무시). 차수 = 이웃 수."""
  neighbors: dict[int, set[int]] = {}
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    if u == v:
      continue
    neighbors.setdefault(u, set()).add(v)
    neighbors.setdefault(v, set()).add(u)
  return neighbors


def _walk_chain(
  start: int,
  neighbors: dict[int, set[int]],
  degree_two: set[int],
) -> list[int]:
  """start 가 속한 차수-2 정점들의 연결 성분(경로/사이클)을 모은다."""
  component: list[int] = []
  seen: set[int] = set()
  stack = [start]
  while stack:
    v = stack.pop()
    if v in seen:
      continue
    seen.add(v)
    component.append(v)
    for n in neighbors[v]:
      if n in degree_two and n not in seen:
        stack.append(n)
  return component


def _order_path(
  head: int,
  tail: int,
  neighbors: dict[int, set[int]],
  internal_set: set[int],
) -> list[int]:
  """경로의 한 끝(head)에서 다른 끝(tail)까지 내부 정점을 순서대로 나열."""
  ordered = [head]
  prev: int | None = None
  current = head
  while current != tail:
    nxt = next(
      n for n in neighbors[current]
      if n in internal_set and n != prev
    )
    ordered.append(nxt)
    prev, current = current, nxt
  return ordered
