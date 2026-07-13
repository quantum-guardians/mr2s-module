"""차수-2 정점 체인 탐지 (ISSUE-52 간선 축약 1단계).

축약 가능한 정보를 만들어서 넘깁니다.

Graph adj matrix를 사용해서 이웃만 느끼도록 합니다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from mr2s_module.domain import Edge, Graph


class ChainKind(StrEnum):
  """체인 종류. StrEnum 이라 문자열 비교/직렬화와 하위호환된다."""

  PATH = "path"      # 일반 체인
  CYCLE = "cycle"    # 매달린/전체 사이클
  FORCED = "forced"  # directed 간선 포함, 체인 전체 방향이 이미 강제됨


@dataclass(frozen=True)
class Chain:
  """탐지된 체인 하나. interior_vertices / edge_ids 는 walk 순서(a → b).

  - endpoints: 양 끝 정점 (a, b). 매달린 사이클이면 a == b (부착점).
  - kind: ChainKind (PATH | CYCLE | FORCED).
  """

  endpoints: tuple[int, int]
  interior_vertices: tuple[int, ...]
  edge_ids: tuple[int, ...]
  kind: ChainKind

  @property
  def length(self) -> int:
    """체인의 간선 수."""
    return len(self.edge_ids)


@dataclass(frozen=True)
class ChainReport:
  """탐지 결과 요약. saved_vars 는 단일 패스 기준이다 — 사이클 제거로 부착점이

  새로 차수-2 가 되는 고정점 반복은 반영하지 않는다(과소 추정 쪽으로 보수적).
  """

  chains: tuple[Chain, ...]
  total_edges: int                    # 현재 QUBO 변수 수 상한 (무방향 간선이 전부 변수)
  contractible_edges: int             # 체인 소속 간선 수 합
  saved_vars: int                     # path: k-1, cycle: k(전부 고정), forced: 무방향 수
  equal_endpoint_weight_chains: int   # 첫·끝 간선 가중치가 같은 path 수 (축약 정확도 자료)


class DegreeTwoChainDetector:
  """차수-2 체인 탐지기. 그래프를 변형하지 않는다(read-only)."""

  def run(self, graph: Graph, min_internal_vertices: int = 1) -> ChainReport:
    if min_internal_vertices < 1:
      raise ValueError("min_internal_vertices must be >= 1")

    incident = _build_incident_edges(graph)
    interior: set[int] = {
      vertex
      for vertex, edges in incident.items()
      if len(edges) == 2 and edges[0].id != edges[1].id  # self-loop 정점 제외
    }

    # interior 의 자료형은 어떻게 되는 거지?

    chains: list[Chain] = []
    visited: set[int] = set()
    for start in sorted(interior): 
      if start in visited:
        continue
      chain = _walk_chain(start, incident, interior)
      visited.update(chain.interior_vertices)
      visited.add(chain.endpoints[0])  # 전체-사이클이면 시작점도 소비됨
      if len(chain.interior_vertices) >= min_internal_vertices:
        chains.append(chain)

    return _build_report(graph, tuple(chains))


def _build_incident_edges(graph: Graph) -> dict[int, list[Edge]]:
  """정점별 incident 간선 목록. self-loop 는 같은 정점에 2회(멀티그래프 차수 관례)."""
  incident: dict[int, list[Edge]] = defaultdict(list)
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    incident[u].append(edge)
    incident[v].append(edge)
  return incident


def _walk_chain(
    start: int,
    incident: dict[int, list[Edge]],
    interior: set[int],
) -> Chain:
  """내부 정점 start 에서 양방향으로 edge id 를 따라 확장해 최대 체인을 만든다."""
  first_edge, second_edge = incident[start]

  fwd_edges, fwd_passed, fwd_terminal = _traverse(start, first_edge, incident, interior)
  if fwd_terminal == start:
    # 전체 컴포넌트가 사이클: 한 바퀴 돌아 시작점 복귀. start 가 끝점 역할.
    return _make_chain((start, start), tuple(fwd_passed), fwd_edges)

  bwd_edges, bwd_passed, bwd_terminal = _traverse(start, second_edge, incident, interior)

  # fwd 는 start→a 방향이므로 뒤집어 a→start→b 순서로 잇는다.
  edges = list(reversed(fwd_edges)) + bwd_edges
  interior_vertices = tuple(reversed(fwd_passed)) + (start,) + tuple(bwd_passed)
  return _make_chain((fwd_terminal, bwd_terminal), interior_vertices, edges)


def _traverse(
    start: int,
    first_edge: Edge,
    incident: dict[int, list[Edge]],
    interior: set[int],
) -> tuple[list[Edge], list[int], int]:
  """start 에서 first_edge 방향으로 내부 정점을 통과하며 진행.

  (지나간 간선들, 경유한 내부 정점들, 종착 정점) 반환. 종착 정점이 start 면 사이클.
  """
  edges = [first_edge]
  passed: list[int] = []
  prev_edge = first_edge
  current = first_edge.other_vertex(start)

  while current in interior and current != start:
    passed.append(current)
    edge_a, edge_b = incident[current]
    next_edge = edge_b if edge_a.id == prev_edge.id else edge_a
    edges.append(next_edge)
    prev_edge = next_edge
    current = next_edge.other_vertex(current)

  return edges, passed, current


def _make_chain(
    endpoints: tuple[int, int],
    interior_vertices: tuple[int, ...],
    edges: list[Edge],
) -> Chain:
  if any(edge.directed for edge in edges):
    kind = ChainKind.FORCED
  elif endpoints[0] == endpoints[1]:
    kind = ChainKind.CYCLE
  else:
    kind = ChainKind.PATH
  return Chain(
    endpoints=endpoints,
    interior_vertices=interior_vertices,
    edge_ids=tuple(edge.id for edge in edges),
    kind=kind,
  )


def _saved_vars(chain: Chain, graph: Graph) -> int:
  """체인 축약 시 사라지는 QUBO 변수 수.

  - path: k 간선 → super edge 1변수, k-1 절감.
  - cycle: 방향을 lift 시점에 고정하고 QUBO 에서 통째로 제외, k 전부 절감.
  - forced: 방향이 이미 강제 → 무방향 간선(현재 변수인 것) 전부 절감.
  """
  if chain.kind == ChainKind.CYCLE:
    return chain.length
  if chain.kind == ChainKind.FORCED:
    return sum(1 for edge_id in chain.edge_ids if not graph.edges[edge_id].directed)
  return chain.length - 1


def _build_report(graph: Graph, chains: tuple[Chain, ...]) -> ChainReport:
  equal_weight = sum(
    1
    for chain in chains
    if chain.kind == ChainKind.PATH
    and graph.edges[chain.edge_ids[0]].weight == graph.edges[chain.edge_ids[-1]].weight
  )
  return ChainReport(
    chains=chains,
    total_edges=len(graph.edges),
    contractible_edges=sum(chain.length for chain in chains),
    saved_vars=sum(_saved_vars(chain, graph) for chain in chains),
    equal_endpoint_weight_chains=equal_weight,
  )
