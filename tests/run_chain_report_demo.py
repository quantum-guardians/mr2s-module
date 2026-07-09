"""ISSUE-52: 차수-2 체인 축약률 리포트 데모.

실데이터 Graph 에 대해 `print_chain_report(graph)` 를 호출하면 축약 시 QUBO 변수
절감률을 출력한다. 이 수치로 축약 본체(super edge presolve) 진행 여부를 결정한다.
직접 실행하면 합성 그래프(grid 형 delaunay, 체인 부착 그래프)로 출력 형식을 시연한다.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mr2s_module.domain import Edge, Graph
from mr2s_module.reduction import (
  ChainKind,
  ChainReport,
  DegreeTwoChainDetector,
)
from tests.util.graph_fixtures import delaunay_graph


def print_chain_report(graph: Graph, min_internal_vertices: int = 1) -> ChainReport:
  """체인 탐지 후 축약률 리포트를 stdout 에 출력하고 ChainReport 를 반환한다."""
  report = DegreeTwoChainDetector().run(
    graph, min_internal_vertices=min_internal_vertices
  )

  kind_counts = Counter(chain.kind for chain in report.chains)
  length_hist = Counter(chain.length for chain in report.chains)
  saved_ratio = report.saved_vars / report.total_edges if report.total_edges else 0.0
  path_count = kind_counts.get(ChainKind.PATH, 0)

  print(f"총 간선(=QUBO 변수 상한): {report.total_edges}")
  print(
    f"체인 수: {len(report.chains)} "
    f"(path {path_count}, cycle {kind_counts.get(ChainKind.CYCLE, 0)}, "
    f"forced {kind_counts.get(ChainKind.FORCED, 0)})"
  )
  print(f"체인 소속 간선: {report.contractible_edges}")
  print(f"절감 변수: {report.saved_vars} ({saved_ratio:.1%})")
  if path_count:
    print(
      f"첫·끝 가중치 동일 path: {report.equal_endpoint_weight_chains}/{path_count} "
      "(동일 비율 높을수록 flow 항 무손실 축약 범위 넓음)"
    )
  if length_hist:
    hist = ", ".join(f"길이 {k}: {v}개" for k, v in sorted(length_hist.items()))
    print(f"체인 길이 분포: {hist}")
  print(
    "(주의: 단일 패스 기준 — 사이클 제거로 부착점이 새로 차수-2 가 되는 경우는"
    " 미반영, 실제 절감은 이 이상)"
  )
  return report


def build_body_with_chains(chain_length: int, chain_count: int) -> Graph:
  """K4 본체에 길이 chain_length 체인 chain_count 개를 부착한 합성 그래프."""
  edges = [
    Edge(0, 1, 1, False), Edge(0, 2, 1, False), Edge(0, 3, 1, False),
    Edge(1, 2, 1, False), Edge(1, 3, 1, False), Edge(2, 3, 1, False),
  ]
  next_vertex = 10
  for i in range(chain_count):
    start, end = i % 4, (i + 1) % 4
    prev = start
    for _ in range(chain_length - 1):
      edges.append(Edge(prev, next_vertex, 1, False))
      prev = next_vertex
      next_vertex += 1
    edges.append(Edge(prev, end, 1, False))
  return Graph(edges=edges)


def main() -> None:
  print("=== delaunay 그래프 (n=30, 내부 차수-2 거의 없음) ===")
  print_chain_report(delaunay_graph(30, seed=42))

  print()
  print("=== K4 본체 + 길이 7 체인 3개 ===")
  print_chain_report(build_body_with_chains(chain_length=7, chain_count=3))


if __name__ == "__main__":
  main()
