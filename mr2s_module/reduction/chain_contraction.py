"""degree-2 체인 축약 + lift (ISSUE-52 간선 축약 2단계, presolve 변환).

path 체인 a-x1-…-xk-b 는 강연결 제약상 전체가 1비트 결정이므로 super edge 1개
(QUBO 변수 1개)로 축약한다. 
축약 시에 시작점과 끝점이 동일한 경우 self-loop 가 생길 수 있다.
이때 해당 간선은 평가와 강연결에 영향을 주지 않으므로 제거해서 평가한다.
복원 시점에서 점수 계산만 참여하도록 한다.

self-loop 삭제를 하면 해당 vertex의 차수가 2가 줄게 된다.
이 영향으로 새롭게 축약이 되는 후보가 생길 수 있으므로, 반복해서 축약을 찾게 된다.
단순 축약은 반복을 야기하지 않는다.
이렇게 구현할 시 겹치는 super edge 가 있을 수 있으므로 재귀적으로 풀어야 한다.

super edge 가중치 기본값은 "harmonic" — W = 1/Σ(1/wᵢ), 1 미만은 1로 설정
APSP 거리(1/w)를 보존해 solve 내부(n-hop·랭킹·병합) 정확도를 높인다. 평가는
복원 후 원본 그래프에서 하므로 최종 점수 계산은 모드와 무관하게 무손실.

축약 그래프의 통과 간선은 전부 클론(원본 id 유지)이다 — DnC 파이프라인이
간선 방향을 in-place 로 고정하므로 원본 그래프 오염을 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mr2s_module.domain import Edge, Graph
from mr2s_module.reduction.degree_two_chain import (
  Chain,
  ChainKind,
  DegreeTwoChainDetector,
)


class SuperEdgeWeight(StrEnum):
  """super edge 가중치 정책. StrEnum 이라 문자열 비교/직렬화와 하위호환된다."""

  HARMONIC = "harmonic"  # 1/Σ(1/wᵢ) — APSP 거리 보존 (기본)
  UNIT = "unit"          # 1 고정 (비교 실험용)


@dataclass(frozen=True)
class ContractionResult:
  """축약 결과. contracted_graph 는 원본과 간선 객체를 공유하지 않는다."""

  contracted_graph: Graph
  chain_by_super_id: dict[int, Chain]  # super edge id → 원본(또는 이전 라운드) 체인
  cycle_chains: tuple[Chain, ...]      # 제거된 매달린 사이클 (lift 때 균일 배향)

  @property
  def has_chains(self) -> bool:
    return bool(self.chain_by_super_id) or bool(self.cycle_chains)


def _super_edge_weight(mode: SuperEdgeWeight, chain: Chain, graph: Graph) -> float:
  """super edge 가중치 정책.

  - HARMONIC: 1/Σ(1/wᵢ) — 체인 거리 Σ(1/wᵢ) == 1/W 로 APSP 거리 보존.
    W < 1 이면 1 로 클램프 (가중치는 1 이상이라는 도메인 규칙).
  - UNIT: 1 고정 (비교 실험용).
  """
  if mode == SuperEdgeWeight.HARMONIC:
    inverse_sum = sum(1 / graph.edges[edge_id].weight for edge_id in chain.edge_ids)
    return max(1.0, 1.0 / inverse_sum) if inverse_sum > 0 else 1.0
  if mode == SuperEdgeWeight.UNIT:
    return 1
  raise ValueError(f"unknown super_edge_weight mode: {mode!r}")


def _clone_edges(graph: Graph, exclude_ids: set[int]) -> list[Edge]:
  """exclude 를 뺀 간선 클론 목록. 원본 id 를 유지해 lift 매칭을 보장한다."""
  clones: list[Edge] = []
  for edge_id, edge in graph.edges.items():
    if edge_id in exclude_ids:
      continue
    clone = Edge(edge.vertices[0], edge.vertices[1], edge.weight, edge.directed)
    clone.id = edge.id
    clones.append(clone)
  return clones


def contract_chains(
    graph: Graph,
    min_internal_vertices: int = 1,
    super_edge_weight: SuperEdgeWeight = SuperEdgeWeight.HARMONIC,
) -> ContractionResult:
  """체인을 축약한 새 Graph 를 만든다. 원본 비변형."""
  detector = DegreeTwoChainDetector()
  chain_by_super_id: dict[int, Chain] = {}
  cycle_chains: list[Chain] = []
  current = graph
  contracted_once = False

  while True:
    report = detector.run(current, min_internal_vertices=min_internal_vertices)
    path_chains = [c for c in report.chains if c.kind == ChainKind.PATH]
    cycles = [c for c in report.chains if c.kind == ChainKind.CYCLE]
    if not path_chains and not cycles: # 전부 비어있는 경우
      break

    removed_ids = {
      edge_id
      for chain in (*path_chains, *cycles)
      for edge_id in chain.edge_ids
    }
    new_edges = _clone_edges(current, removed_ids)
    for chain in path_chains:
      a, b = chain.endpoints
      weight = _super_edge_weight(super_edge_weight, chain, current)
      super_edge = Edge(a, b, weight, False)
      chain_by_super_id[super_edge.id] = chain
      new_edges.append(super_edge)
    cycle_chains.extend(cycles)

    current = Graph(edges={edge.id: edge for edge in new_edges})
    contracted_once = True
    if not cycles:
      break  # path 축약은 차수 불변 → 새 체인이 생길 수 없다

  if not contracted_once:
    # 체인이 없어도 클론을 돌려준다 — solver 의 in-place 방향 고정으로부터
    # 원본을 항상 보호해 호출부 분기를 없앤다.
    current = Graph(edges={edge.id: edge for edge in _clone_edges(current, set())})

  return ContractionResult(
    contracted_graph=current,
    chain_by_super_id=chain_by_super_id,
    cycle_chains=tuple(cycle_chains),
  )


def _chain_directions(
    chain: Chain, forward: bool
) -> list[tuple[int, tuple[int, int]]]:
  """체인 간선들의 (edge_id, (tail, head)) 목록. forward=False 면 역방향."""
  a, b = chain.endpoints
  vertices = [a, *chain.interior_vertices, b]
  edge_ids = list(chain.edge_ids)
  if not forward:
    vertices.reverse()
    edge_ids.reverse()
  return [
    (edge_id, (tail, head))
    for edge_id, tail, head in zip(edge_ids, vertices, vertices[1:])
  ]


def lift_solution_edges(
    solved_edges: dict[int, tuple[int, int]],
    result: ContractionResult,
) -> dict[int, tuple[int, int]]:
  """축약 그래프의 방향 결과를 원본 간선 방향으로 전개한다.

  - super edge → 체인 간선들에 균일 전개 (중첩 super edge 도 끝까지).
  - 제거된 cycle 체인 → walk 순서로 균일 배향해 주입 (평가 전용, 변수 없음).
  - solver 결과에 super edge 가 누락되면 해당 체인이 통째로 미배향되므로
    조용히 넘기지 않고 예외를 던진다.
  """
  pending = list(solved_edges.items())
  for chain in result.cycle_chains:
    pending.extend(_chain_directions(chain, forward=True))

  lifted: dict[int, tuple[int, int]] = {}
  expanded: set[int] = set()
  while pending:
    edge_id, direction = pending.pop()
    chain = result.chain_by_super_id.get(edge_id)
    if chain is None:
      lifted[edge_id] = direction
      continue

    expanded.add(edge_id)
    a, b = chain.endpoints
    if direction == (a, b):
      forward = True
    elif direction == (b, a):
      forward = False
    else:
      raise ValueError(
        f"super edge {edge_id} direction {direction} does not match endpoints {(a, b)}"
      )
    pending.extend(_chain_directions(chain, forward))

  missing = set(result.chain_by_super_id) - expanded
  if missing:
    raise ValueError(f"solver result missing super edge ids: {sorted(missing)}")
  return lifted
