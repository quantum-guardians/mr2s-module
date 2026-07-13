"""ISSUE-52: 차수-2 체인 탐지기 검증.

멀티그래프 오탐 방지가 핵심 회귀 대상: 내부 판정은 "incident 간선 수 2" 기준이어야
하며, simple graph 투영("이웃 정점 수 2")은 평행 간선이 붙은 정점을 내부로 오판한다.
"""

import pytest

from mr2s_module.domain import Edge, Graph
from mr2s_module.reduction import (
  ChainKind,
  DegreeTwoChainDetector,
)


def _detect(edges: list[Edge], min_internal_vertices: int = 1):
  return DegreeTwoChainDetector().run(
    Graph(edges=edges), min_internal_vertices=min_internal_vertices
  )


def test_detects_basic_path_chain() -> None:
  # 0 - 10 - 11 - 12 - 1, 끝점 0/1 은 본체(차수 3 이상)에 연결
  chain_edges = [
    Edge(0, 10, 1, False),
    Edge(10, 11, 1, False),
    Edge(11, 12, 1, False),
    Edge(12, 1, 1, False),
  ]
  body = [
    Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False),
    Edge(2, 3, 1, False),
  ]
  report = _detect(chain_edges + body)

  assert len(report.chains) == 1
  chain = report.chains[0]
  assert chain.kind == ChainKind.PATH
  assert set(chain.endpoints) == {0, 1}
  assert set(chain.interior_vertices) == {10, 11, 12}
  assert chain.edge_ids == tuple(
    e.id for e in (chain_edges if chain.endpoints[0] == 0 else reversed(chain_edges))
  )
  assert report.saved_vars == 3  # 4간선 → 1변수
  assert report.contractible_edges == 4


def test_parallel_edges_block_false_interior() -> None:
  # u ══(평행 2개)══ v ──── w : v 는 이웃 2개지만 간선 3개 → 내부 아님
  edges = [
    Edge(0, 1, 1, False),
    Edge(0, 1, 1, False),
    Edge(1, 2, 1, False),
    # u, w 를 본체에 붙여 차수 3 이상으로
    Edge(0, 3, 1, False), Edge(0, 4, 1, False),
    Edge(2, 3, 1, False), Edge(2, 4, 1, False),
    Edge(3, 4, 1, False),
  ]
  report = _detect(edges)
  for chain in report.chains:
    assert 1 not in chain.interior_vertices


def test_hanging_cycle_detected() -> None:
  # a(0) - p(10) - q(11) - r(12) - a(0), a 는 본체 연결
  cycle_edges = [
    Edge(0, 10, 1, False),
    Edge(10, 11, 1, False),
    Edge(11, 12, 1, False),
    Edge(12, 0, 1, False),
  ]
  # 본체는 K4 (모든 정점 차수 3 이상 — 의도치 않은 체인 방지)
  body = [
    Edge(0, 1, 1, False), Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  report = _detect(cycle_edges + body)

  cycles = [c for c in report.chains if c.kind == ChainKind.CYCLE]
  assert len(cycles) == 1
  assert cycles[0].endpoints == (0, 0)
  assert set(cycles[0].interior_vertices) == {10, 11, 12}
  assert cycles[0].length == 4
  # 사이클은 방향을 lift 시점에 고정 → 변수 전부 절감
  assert sum(c.length for c in cycles) == 4


def test_parallel_pair_hanging_two_cycle() -> None:
  # u(0) ══ v(1) : v 는 평행 간선 2개뿐 → 길이-2 매달린 사이클
  edges = [
    Edge(0, 1, 1, False),
    Edge(0, 1, 1, False),
    # 본체 K4 {0,2,3,4}
    Edge(0, 2, 1, False), Edge(0, 3, 1, False), Edge(0, 4, 1, False),
    Edge(2, 3, 1, False), Edge(2, 4, 1, False), Edge(3, 4, 1, False),
  ]
  report = _detect(edges)

  cycles = [c for c in report.chains if c.kind == ChainKind.CYCLE]
  assert len(cycles) == 1
  assert cycles[0].endpoints == (0, 0)
  assert cycles[0].interior_vertices == (1,)
  assert cycles[0].length == 2


def test_chain_coexists_with_direct_parallel_edge() -> None:
  # a-1-2-b 체인 + a-b 직행 간선: 멀티그래프라 축약 시 평행 super edge 허용,
  # 직행 간선 존재가 체인 탐지를 막으면 안 된다 (구식 코드의 스킵 가드 제거 확인).
  chain_edges = [
    Edge(0, 10, 1, False),
    Edge(10, 11, 1, False),
    Edge(11, 1, 1, False),
  ]
  edges = chain_edges + [
    Edge(0, 1, 1, False),  # 직행
    # 본체 K4 {0,1,2,3}의 나머지
    Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  report = _detect(edges)

  assert len(report.chains) == 1
  assert report.chains[0].kind == ChainKind.PATH
  assert set(report.chains[0].interior_vertices) == {10, 11}


def test_chain_with_directed_edge_is_forced() -> None:
  edges = [
    Edge(0, 10, 1, False),
    Edge(10, 11, 1, True),  # 방향 고정 → 체인 전체 강제
    Edge(11, 1, 1, False),
    Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  report = _detect(edges)

  forced = [c for c in report.chains if c.kind == ChainKind.FORCED]
  assert len(forced) == 1
  assert forced[0].length == 3
  # directed 간선은 애초에 변수가 아님 → 무방향 2개만 절감으로 집계
  assert report.saved_vars == 2


def test_min_internal_vertices_boundary() -> None:
  # 내부 정점 1개짜리 체인 0-10-1
  edges = [
    Edge(0, 10, 1, False),
    Edge(10, 1, 1, False),
    Edge(0, 1, 1, False),
    Edge(0, 2, 1, False), Edge(1, 2, 1, False), Edge(2, 3, 1, False),
    Edge(0, 3, 1, False), Edge(1, 3, 1, False),
  ]
  assert len(_detect(edges, min_internal_vertices=1).chains) == 1
  assert len(_detect(edges, min_internal_vertices=2).chains) == 0
  with pytest.raises(ValueError):
    _detect(edges, min_internal_vertices=0)


def test_whole_component_cycle() -> None:
  # 그래프 전체가 사이클 0-1-2-3-0: 모든 정점 차수 2
  edges = [
    Edge(0, 1, 1, False),
    Edge(1, 2, 1, False),
    Edge(2, 3, 1, False),
    Edge(3, 0, 1, False),
  ]
  report = _detect(edges)

  assert len(report.chains) == 1
  chain = report.chains[0]
  assert chain.kind == ChainKind.CYCLE
  assert chain.endpoints[0] == chain.endpoints[1]
  assert chain.length == 4
  assert report.saved_vars == 4


def test_no_degree_two_vertices_yields_empty_report() -> None:
  # K4: 전 정점 차수 3
  edges = [
    Edge(0, 1, 1, False), Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  report = _detect(edges)

  assert report.chains == ()
  assert report.saved_vars == 0
  assert report.contractible_edges == 0
  assert report.total_edges == 6


def test_equal_endpoint_weight_chains_counted() -> None:
  # 체인 1: 첫·끝 가중치 같음 (5, 3, 5) / 체인 2: 다름 (5, 2)
  edges = [
    Edge(0, 10, 5, False), Edge(10, 11, 3, False), Edge(11, 1, 5, False),
    Edge(2, 12, 5, False), Edge(12, 3, 2, False),
    # 본체: 끝점들 차수 3 이상 확보
    Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False),
    Edge(0, 1, 1, False), Edge(2, 3, 1, False),
  ]
  report = _detect(edges)

  paths = [c for c in report.chains if c.kind == ChainKind.PATH]
  assert len(paths) == 2
  assert report.equal_endpoint_weight_chains == 1
