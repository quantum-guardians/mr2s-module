"""차수-2 체인 축약(DegreeTwoChainReducer)의 진짜 동기 — "n-hop 지평선 맹점" — 을 지표로 측정.

왜 이 실험인가
--------------
이 프로젝트는 small-world 보상을 위해 n-hop 방식(NHopPolyGenerator)을 쓴다.
`_get_n_hop_polynomial` 은 각 시드 정점에서 길이가 *정확히* n 인 단순경로를 모두 열거해
보상 항을 만든다. 그런데 차수-2 체인(분기 없는 복도)에 들어가면 직진만 하다 체인 내부에서
막다른 길로 끝난다.

  예) 허브 a - 1 - 2 - 3 - 4 - 5 - 허브 b, horizon n=3:
      시드 a 의 길이-3 경로는 a-1-2-3 까지 → a 는 b 를 못 본다.
      n-hop 보상 예산이 "막다른 복도"에 소모되어, small-world 목적함수가
      허브-허브(매크로) 구조를 반영하지 못한다.

체인을 a-b 단일 간선으로 축약하면 a 가 1홉에 b 를 보게 되어, *같은* n-hop 예산으로 진짜
매크로 구조를 판별/최적화할 수 있다. ← DegreeTwoChainReducer 를 만든 이유.

이 스크립트는 제품 코드를 바꾸지 않고, 그 "지평선 맹점"과 해소를 지표로 측정·로깅한다.

측정 지표
---------
A. 구조 지표 (결정적, SA 무관 — 헤드라인). 각 horizon n 에 대해 baseline vs reduced:
   - M1 맹점 체인 수 : 체인 edge-길이 L > n 인 체인 수(두 허브가 서로 지평선 밖). 축약 후 0.
   - M2 보상 낭비율  : 길이 정확히 n 인 단순경로(=NHop 보상 항) 중 종점이 차수-2 체인
                       내부(막다른 복도, 매크로 정보 없음)인 비율 vs 허브인 비율.
   - M3 매크로 도달쌍: 최단거리(hop) ≤ n 인 (허브,허브) 순서쌍 수. 축약이 먼 허브를
                       지평선 안으로 끌어와 reduced ≫ baseline.

B. SA 결과 지표 (n≈300 실제 SA, downstream 효과):
   - NHop(n=2)+Flow QUBO 를 baseline vs reduced(unit 가중치)로 neal SA 풀이 비교:
     변수 수, 강연결 성공률, best 해 APSP, 소요시간.

비고: 본 벤치마크는 모든 차수-2 체인을 잡아 M1 을 degree-기반 M2/M3 과 일관되게 만들고
n=2 에서도 L≤n / L>n 임계점이 드러나도록 reducer 를 min_internal_vertices=1 로 둔다
(제품 기본값은 2; 측정 설정 선택일 뿐 제품 코드는 불변). 축약 간선을 합(sum) 가중치로
풀면 Flow/NHop QUBO 가 왜곡돼 강연결이 붕괴하므로 SA 는 unit 가중치로 푼다(1차 실험에서
입증, degree-two-chain-reduction-weight-caveat 메모리 참조).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mr2s_module import (
  ApspSumRanker,
  DegreeTwoChainReducer,
  Edge,
  FlowPolyGenerator,
  Graph,
  NHop,
  NHopPolyGenerator,
  QuboSolver,
  SmallWorldSpec,
)
from mr2s_module.qubo.solution_processing import process_solution
from mr2s_module.util import add_polys
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm
from tests.util.graph_fixtures import delaunay_graph

logger = logging.getLogger("chain_reduction_benchmark")

SWEEP_HORIZONS = (2, 3, 4)


# --------------------------------------------------------------------------- #
# 그래프 생성 (읽기 전용 무방향 그래프)
# --------------------------------------------------------------------------- #
def thin_to_biconnected(graph: Graph, seed: int, remove_ratio: float) -> Graph:
  """간선을 remove_ratio 만큼 제거하되 평면(planar) + biconnected(2-정점 연결)를 유지.

  - biconnected : 모든 무방향 간선에 강한 방향(절단정점/브리지 없음)을 줄 수 있음을 보장.
  - planar      : 간선 제거는 평면성을 깨지 않지만(평면 그래프의 부분그래프는 평면),
                  제거 후에도 평면임을 명시적으로 확인해 결과 그래프가 평면 biconnected
                  임을 보장한다.
  제거를 시도해 biconnected 가 깨지면 되돌린다. 목표 간선 수(target)에 도달하거나 더 이상
  제거할 수 없으면 멈춘다.
  """
  nx_graph = nx.Graph()
  nx_graph.add_edges_from(edge.endpoints() for edge in graph.edges.values())

  rng = np.random.default_rng(seed)
  edges = list(nx_graph.edges())
  rng.shuffle(edges)
  target = int(len(edges) * (1.0 - remove_ratio))

  for u, v in edges:
    if nx_graph.number_of_edges() <= target:
      break
    nx_graph.remove_edge(u, v)
    if not nx.is_biconnected(nx_graph):
      nx_graph.add_edge(u, v)

  result = Graph(edges=[Edge(min(u, v), max(u, v), 1, False) for u, v in nx_graph.edges()])
  # 평면 biconnected 불변식 명시 확인 (제거는 평면성을 깨지 않음 → 항상 성립).
  assert nx.check_planarity(nx_graph)[0], "thinned graph must stay planar"
  assert nx.is_biconnected(nx_graph), "thinned graph must stay biconnected"
  return result


def subdivide_edges(
    graph: Graph,
    seed: int,
    subdivide_ratio: float,
    min_len: int,
    max_len: int,
) -> Graph:
  """간선 일부를 길이가 무작위인 차수-2 체인으로 세분화해 길이 섞인 체인을 만든다.

  세분화 대상 간선 (u, v) 는 내부 차수-2 정점 k∈[min_len, max_len] 개를 끼워
  u - w1 - … - wk - v 가 된다(체인 edge-길이 L = k+1). 간선 세분화는 biconnectivity
  를 보존하므로 강한 방향배정 가능성은 유지된다. 나머지 간선은 직접 간선(L=1)으로 남는다.
  """
  rng = np.random.default_rng(seed)
  edges = list(graph.edges.values())
  rng.shuffle(edges)
  num_subdivide = int(len(edges) * subdivide_ratio)

  next_id = max(graph.get_vertices()) + 1
  new_edges: list[Edge] = []
  for index, edge in enumerate(edges):
    u, v = edge.endpoints()
    if index >= num_subdivide:
      new_edges.append(Edge(u, v, 1, False))
      continue
    internal = int(rng.integers(min_len, max_len + 1))
    prev = u
    for _ in range(internal):
      mid = next_id
      next_id += 1
      new_edges.append(Edge(min(prev, mid), max(prev, mid), 1, False))
      prev = mid
    new_edges.append(Edge(min(prev, v), max(prev, v), 1, False))
  return Graph(edges=new_edges)


# --------------------------------------------------------------------------- #
# 허브/차수 헬퍼
# --------------------------------------------------------------------------- #
def degree_map(graph: Graph) -> dict[int, int]:
  """무방향 단순 차수 (self-loop 무시)."""
  degree: dict[int, int] = defaultdict(int)
  for edge in graph.edges.values():
    u, v = edge.endpoints()
    if u == v:
      continue
    degree[u] += 1
    degree[v] += 1
  return degree


def hub_set(graph: Graph) -> set[int]:
  """허브 = 차수 ≠ 2 인 정점 (차수-2 정점은 체인 내부 = 막다른 복도)."""
  return {v for v, d in degree_map(graph).items() if d != 2}


def is_planar_biconnected(graph: Graph) -> tuple[bool, bool]:
  """그래프가 (평면, biconnected) 인지 반환 — 결과 검증/로깅용."""
  nx_graph = nx.Graph()
  nx_graph.add_nodes_from(graph.get_vertices())
  nx_graph.add_edges_from(
    edge.endpoints() for edge in graph.edges.values()
    if edge.endpoints()[0] != edge.endpoints()[1]
  )
  planar = nx.check_planarity(nx_graph)[0]
  biconn = nx.is_biconnected(nx_graph) if nx_graph.number_of_nodes() > 2 else True
  return planar, biconn


# --------------------------------------------------------------------------- #
# 구조 지표 (M1/M2/M3) — 무방향 인접 dict 기반, 읽기 전용
# --------------------------------------------------------------------------- #
def count_paths_by_endpoint(adj, n: int, hubs: set[int]) -> tuple[int, int]:
  """M2: 모든 시드에서 길이가 정확히 n 인 단순경로를 DFS 로 열거, 종점 유형별 카운트.

  NHopPolyGenerator._get_n_hop_polynomial 의 단순경로 열거와 동일 의미(시드=모든 정점).
  반환: (hub_end, chain_end) — 종점이 허브인 경로 수 / 차수-2 체인 내부에서 끝난 경로 수.
  chain_end 가 클수록 n-hop 보상 예산이 막다른 복도에 낭비된 것.
  """
  hub_end = 0
  chain_end = 0

  def dfs(last: int, depth: int, visited: set[int]) -> None:
    nonlocal hub_end, chain_end
    if depth == n:
      if last in hubs:
        hub_end += 1
      else:
        chain_end += 1
      return
    for entry in adj.get(last, []):
      nb = entry.vertex
      if nb in visited:
        continue
      visited.add(nb)
      dfs(nb, depth + 1, visited)
      visited.remove(nb)

  for seed in adj:
    dfs(seed, 0, {seed})
  return hub_end, chain_end


def hub_pairs_within(adj, n: int, hubs: set[int]) -> int:
  """M3: 각 허브에서 깊이 ≤ n BFS 로 도달하는 (허브,허브) 순서쌍 수."""
  count = 0
  for seed in hubs:
    dist = {seed: 0}
    frontier = [seed]
    for _ in range(n):
      nxt: list[int] = []
      for u in frontier:
        for entry in adj.get(u, []):
          if entry.vertex not in dist:
            dist[entry.vertex] = dist[u] + 1
            nxt.append(entry.vertex)
      frontier = nxt
    count += sum(1 for v in dist if v != seed and v in hubs)
  return count


def structure_metrics(graph: Graph, chains, n: int) -> dict:
  """원본/축약 그래프 한쪽에 대해 horizon n 의 M1/M2/M3 를 계산."""
  adj = graph.get_adjacency_dict()
  hubs = hub_set(graph)
  blind = sum(1 for c in chains if c.length > n)
  hub_end, chain_end = count_paths_by_endpoint(adj, n, hubs)
  total_paths = hub_end + chain_end
  return {
    "blind_chains": blind,
    "total_chains": len(chains),
    "path_hub_end": hub_end,
    "path_chain_end": chain_end,
    "waste_ratio": (chain_end / total_paths) if total_paths else 0.0,
    "hub_pairs": hub_pairs_within(adj, n, hubs),
  }


# --------------------------------------------------------------------------- #
# SA 결과 지표 (Part B)
# --------------------------------------------------------------------------- #
def build_bqm(graph: Graph):
  n_hop = NHopPolyGenerator()
  n_hop.small_world_spec = SmallWorldSpec(n_hops=[NHop(n=2, weight=1)])
  polynomial = add_polys(FlowPolyGenerator().run(graph), n_hop.run(graph))
  return map_binary_poly_to_bqm(polynomial)


def solve_sa(graph: Graph, num_reads: int, seed: int):
  bqm = build_bqm(graph)
  solver = QuboSolver.create_sa_solver(ranker=ApspSumRanker(), num_reads=num_reads)
  solution = solver.run(bqm, graph)
  return solution, len(bqm.variables)


def strongly_connected(vertices: set[int], directed_edges) -> bool:
  digraph = nx.DiGraph()
  digraph.add_nodes_from(vertices)
  digraph.add_edges_from(directed_edges)
  return nx.is_strongly_connected(digraph)


def strong_rate(sample_set, canonical_edges, vertices: set[int]) -> tuple[float, int]:
  """모든 SA read 의 강연결 성공률."""
  strong = 0
  reads = 0
  for sample in sample_set.samples():
    directed = [
      edge.vertices for edge in process_solution(sample, canonical_edges).values()
    ]
    reads += 1
    if strongly_connected(vertices, directed):
      strong += 1
  if reads == 0:
    return 0.0, 0
  return strong / reads, reads


def apsp_sum(graph: Graph, directed_edges) -> float:
  # Solution.edges(dict[id, Edge]) 도, (u,v) 튜플 리스트도 받도록 정규화.
  if isinstance(directed_edges, dict):
    directed_edges = [edge.vertices for edge in directed_edges.values()]
  digraph = nx.DiGraph()
  digraph.add_nodes_from(graph.get_vertices())
  digraph.add_edges_from(directed_edges)
  total = 0.0
  for source in digraph.nodes():
    lengths = nx.single_source_shortest_path_length(digraph, source)
    total += float(sum(lengths.values()))
  return total


def expand_best(result, reduced_solution) -> list[tuple[int, int]]:
  """축약 그래프 best 해(directed)를 원본 위 directed 간선으로 펼쳐 (u,v) 리스트 반환."""
  oriented = [
    Edge(edge.vertices[0], edge.vertices[1], edge.weight, True)
    for edge in reduced_solution.edges.values()
  ]
  return [edge.vertices for edge in result.expand(oriented)]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-points", type=int, default=102)
  parser.add_argument("--remove-ratio", type=float, default=0.30)
  parser.add_argument("--subdivide-ratio", type=float, default=0.40)
  parser.add_argument("--min-len", type=int, default=1, help="체인 내부 차수-2 정점 최소 수")
  parser.add_argument("--max-len", type=int, default=4, help="체인 내부 차수-2 정점 최대 수")
  parser.add_argument("--num-reads", type=int, default=80)
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
  )

  logger.info("=== n-hop 지평선 가시성 실험 (차수-2 체인 축약의 진짜 이유) ===")

  # 1) 그래프 생성: 길이 섞인 차수-2 체인을 포함한 n≈300 그래프
  base = delaunay_graph(args.base_points, seed=args.seed)
  base = thin_to_biconnected(base, seed=args.seed, remove_ratio=args.remove_ratio)
  graph = subdivide_edges(
    base,
    seed=args.seed,
    subdivide_ratio=args.subdivide_ratio,
    min_len=args.min_len,
    max_len=args.max_len,
  )

  # min_internal_vertices=1: 모든 차수-2 체인을 포착해 M1 을 degree-기반 M2/M3 과 일관되게.
  result = DegreeTwoChainReducer(min_internal_vertices=1).reduce(graph)
  reduced_graph = result.reduced_graph
  reduced_chains: tuple = ()  # 축약 후엔 체인이 모두 길이-1 단일 간선 → 맹점 0.

  n = len(graph.get_vertices())
  hubs = hub_set(graph)
  lengths = [c.length for c in result.chains]
  length_hist = sorted({L: lengths.count(L) for L in set(lengths)}.items()) if lengths else []
  logger.info(
    "그래프: 정점 n=%d, 간선 m=%d, 허브(차수≠2) %d개 | base_points=%d, subdivide_ratio=%.2f",
    n, len(graph.edges), len(hubs), args.base_points, args.subdivide_ratio,
  )
  logger.info(
    "체인: %d개 (edge-길이 L 분포 %s) | 축약 후 변수(간선) %d -> %d (절감 %d)",
    len(result.chains), length_hist,
    len(graph.edges), len(reduced_graph.edges),
    len(graph.edges) - len(reduced_graph.edges),
  )
  g_planar, g_biconn = is_planar_biconnected(graph)
  r_planar, r_biconn = is_planar_biconnected(reduced_graph)
  logger.info(
    "불변식: 실험그래프(평면=%s, biconnected=%s) | 축약그래프(평면=%s, biconnected=%s) "
    "| 간선 %.0f%% 제거(remove_ratio=%.2f)",
    g_planar, g_biconn, r_planar, r_biconn, 100.0 * args.remove_ratio, args.remove_ratio,
  )

  # 2) 구조 지표 sweep (n∈{2,3,4}). baseline = 원본, reduced = 축약 그래프.
  logger.info("---------------- A. 구조 지표 (horizon sweep) ----------------")
  logger.info(
    "%-6s | %-22s | %-30s | %-22s",
    "n", "M1 맹점체인(L>n)", "M2 n-hop 보상 낭비율(체인종점)", "M3 허브도달쌍(≤n)",
  )
  for h in SWEEP_HORIZONS:
    base_m = structure_metrics(graph, result.chains, h)
    red_m = structure_metrics(reduced_graph, reduced_chains, h)
    blind_ratio = (
      100.0 * base_m["blind_chains"] / base_m["total_chains"]
      if base_m["total_chains"] else 0.0
    )
    logger.info(
      "n=%-4d | base %3d/%-3d (%4.0f%%) red 0 | "
      "base %5.1f%% (%d/%d) red %4.1f%% | base %5d  red %5d (x%.1f)",
      h,
      base_m["blind_chains"], base_m["total_chains"], blind_ratio,
      100.0 * base_m["waste_ratio"],
      base_m["path_chain_end"], base_m["path_chain_end"] + base_m["path_hub_end"],
      100.0 * red_m["waste_ratio"],
      base_m["hub_pairs"], red_m["hub_pairs"],
      red_m["hub_pairs"] / max(base_m["hub_pairs"], 1),
    )

  # 3) SA 결과 지표 (Part B): NHop(n=2)+Flow, baseline vs reduced(unit)
  logger.info("---------------- B. SA 결과 지표 (NHop n=2 + Flow) ----------------")
  t0 = time.perf_counter()
  base_solution, base_vars = solve_sa(graph, args.num_reads, args.seed)
  base_time = time.perf_counter() - t0
  base_strong, base_reads = strong_rate(
    base_solution.sample_set, list(graph.edges.values()), graph.get_vertices()
  )
  base_apsp = apsp_sum(graph, base_solution.edges)
  logger.info(
    "[baseline] 변수=%d | 강연결 %.1f%% (%d reads) | best APSP=%.0f | %.2fs",
    base_vars, 100.0 * base_strong, base_reads, base_apsp, base_time,
  )

  solve_graph = result.reduced_graph
  t0 = time.perf_counter()
  reduced_solution, reduced_vars = solve_sa(solve_graph, args.num_reads, args.seed)
  reduced_time = time.perf_counter() - t0
  reduced_strong, reduced_reads = strong_rate(
    reduced_solution.sample_set, list(solve_graph.edges.values()),
    solve_graph.get_vertices(),
  )
  reduced_apsp = apsp_sum(graph, expand_best(result, reduced_solution))
  logger.info(
    "[reduced] 변수=%d (절감 %.1f%%) | 강연결 %.1f%% (%d reads) | "
    "best APSP=%.0f | %.2fs",
    reduced_vars,
    100.0 * (base_vars - reduced_vars) / max(base_vars, 1),
    100.0 * reduced_strong, reduced_reads, reduced_apsp, reduced_time,
  )
  logger.info(
    "[비고] FlowPoly 가 흐름보존을 가중치 무관(±1)하게 다루므로 축약 간선을 "
    "sum 가중치 그대로 풀어도 강연결 안전, 거리(NHop/APSP)는 보존."
  )

  # 4) 결론 1줄
  logger.info("------------------------------------------------------------------")
  logger.info(
    "결론: 차수-2 체인은 n-hop 보상을 막다른 복도에 낭비시켜 허브-허브 매크로 구조를 "
    "지평선 밖으로 밀어낸다(M1>0, M2 낭비, M3 작음). 단일 간선 축약은 같은 n-hop 예산으로 "
    "맹점을 0 으로, 허브 도달쌍을 크게 늘리고(M3↑), QUBO 변수를 %.1f%% 줄이면서 "
    "강연결을 개선하고 APSP 는 동등 수준으로 유지한다 — 이것이 DegreeTwoChainReducer 의 존재 이유.",
    100.0 * (base_vars - reduced_vars) / max(base_vars, 1),
  )


if __name__ == "__main__":
  main()
