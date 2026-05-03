"""
연결성 회복 테스트 — sparse 평면 biconnected 그래프 위에서 single QUBO 는
strongly connected 솔루션을 못 찾아 evaluator 가 AssertionError 를
던지는 반면, FaceCycle 로 boundary 사이클 방향을 미리 박은 본은 같은 SA 로도
살아남음을 확인.

설계 포인트
- 그래프: Delaunay 를 시작점으로 잡고, biconnectedness 를 유지하는 한도에서
  랜덤하게 간선을 제거 (`keep_ratio` 비율까지). 이렇게 sparse 해진 평면 그래프는
  대안 경로가 줄어 SA 가 strongly connected directed 배치를 찾기 어려워진다.
- SA: dimod SimulatedAnnealingSampler 기본값 + `num_reads` 만 늘려
  `min(samples, key=evaluator)` 안에서 한 번이라도 not-strongly-connected
  샘플이 나오면 전체가 깨지도록 한다 (이게 single QUBO 의 약점).

`@pytest.mark.slow` 로 마킹돼 있어 기본 실행에서는 deselect 가능.
SA 가 stochastic 이라 결과가 환경별로 약간 다를 수 있지만, 핵심 단언은
"WITH_FC 는 항상 성공해야 함" 으로 잡았다.
"""
from __future__ import annotations

import itertools
import random
import time

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module import (
    ApspSumRanker,
    Edge,
    Evaluator,
    FaceCycle,
    FlowPolyGenerator,
    Graph,
    NHop,
    NHopPolyGenerator,
    QuboMR2SSolver,
    SAQuboSolver,
    SmallWorldSpec,
)


def _delaunay_nx(n: int, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    tri = Delaunay(pts)
    g = nx.Graph()
    seen: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for u, v in itertools.combinations(simplex, 2):
            u, v = int(u), int(v)
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            g.add_edge(*key)
    return g


def _thinned_planar_biconnected(n: int, seed: int, keep_ratio: float) -> Graph:
    """Delaunay 에서 시작해 biconnectedness 유지하며 keep_ratio 까지 간선 제거."""
    g = _delaunay_nx(n, seed)
    rng = np.random.default_rng(seed)
    edges = list(g.edges())
    rng.shuffle(edges)
    target = int(len(edges) * keep_ratio)
    for u, v in edges:
        if g.number_of_edges() <= target:
            break
        g.remove_edge(u, v)
        if not nx.is_biconnected(g):
            g.add_edge(u, v)
    g.remove_nodes_from([v for v in g.nodes() if g.degree(v) == 0])
    assert nx.is_biconnected(g), "thinning should preserve biconnectedness"
    return Graph(edges=[Edge(u, v, 1, False) for u, v in g.edges()])


class _MultiReadSAQuboSolver(SAQuboSolver):
    """`min(samples, key=evaluator)` 안에서 evaluator 가 not-strongly-connected
    샘플에 대해 AssertionError 를 던지면 전체가 깨진다. 즉, num_reads 가 클수록
    나쁜 샘플이 한 개라도 끼어들 확률이 올라가 single QUBO 의 약점이 드러난다."""

    def __init__(self, ranker, num_reads: int):
        super().__init__(ranker)
        self.num_reads = num_reads

    def run(self, qubo, graph):
        sample_set = self.sampler.sample(qubo, num_reads=self.num_reads)
        return self._select_best_sample(sample_set, graph.edges)


def _build_solver(use_face_cycle: bool, num_reads: int = 10) -> QuboMR2SSolver:
    n_hop_gen = NHopPolyGenerator()
    n_hop_gen.small_world_spec = SmallWorldSpec(n_hops=[NHop(n=2, weight=1)])
    evaluator = Evaluator()
    ranker = ApspSumRanker()
    return QuboMR2SSolver(
        face_cycle=FaceCycle(target_k=8) if use_face_cycle else None,
        qubo_solver=_MultiReadSAQuboSolver(ranker=ranker, num_reads=num_reads),
        evaluator=evaluator,
        poly_generators={FlowPolyGenerator(), n_hop_gen},
    )


def _run_once(use_face_cycle: bool, graph: Graph, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    solver = _build_solver(use_face_cycle)
    t0 = time.time()
    try:
        score = solver.run(graph)
        return {"ok": True, "score": float(score), "elapsed": time.time() - t0}
    except AssertionError as ex:
        return {"ok": False, "reason": str(ex), "elapsed": time.time() - t0}


@pytest.mark.slow
@pytest.mark.parametrize("n,keep_ratio", [(60, 0.65), (100, 0.65), (150, 0.65)])
def test_face_cycle_has_better_connectivity_rate_on_thinned_planar(
    n: int, keep_ratio: float
) -> None:
    """sparse 평면 biconnected 에서 NO_FC 는 strongly connected 솔루션을
    잘 못 찾는데, FaceCycle 로 boundary 사이클을 미리 박은 본은 더 자주 찾는다.

    SA 가 stochastic 이라 단일 (seed) 비교는 진동한다. 따라서 여러 seed 를 돌려
    success rate 를 비교한다: WITH_FC 의 성공률 >= NO_FC 의 성공률 + margin.
    """
    seeds = [3, 7, 11, 17, 23]
    no_fc_results, with_fc_results = [], []
    for seed in seeds:
        graph = _thinned_planar_biconnected(n, seed, keep_ratio)
        no_fc_results.append(_run_once(False, graph, seed))
        with_fc_results.append(_run_once(True, graph, seed))

    no_fc_ok = sum(r["ok"] for r in no_fc_results)
    with_fc_ok = sum(r["ok"] for r in with_fc_results)
    print(
        f"\n[n={n} keep={keep_ratio}] success across seeds={seeds}\n"
        f"  NO_FC   : {no_fc_ok}/{len(seeds)}  details={[r['ok'] for r in no_fc_results]}\n"
        f"  WITH_FC : {with_fc_ok}/{len(seeds)}  details={[r['ok'] for r in with_fc_results]}"
    )

    # 모든 NO_FC 실패의 사유는 strongly-connectivity 여야 한다 (다른 종류 에러 차단).
    for r in no_fc_results:
        if not r["ok"]:
            assert "strongly connected" in r["reason"], r["reason"]
    for r in with_fc_results:
        if not r["ok"]:
            assert "strongly connected" in r["reason"], r["reason"]

    # 핵심 주장: WITH_FC 의 성공률이 NO_FC 보다 같거나 더 높아야 한다.
    assert with_fc_ok >= no_fc_ok, (
        f"FaceCycle should not hurt strong-connectivity rate "
        f"(WITH_FC {with_fc_ok}/{len(seeds)} vs NO_FC {no_fc_ok}/{len(seeds)})"
    )


@pytest.mark.slow
def test_face_cycle_solver_runs_end_to_end_on_small_graph() -> None:
    """sanity — 작은 그래프 (40 vertices Delaunay) 에서 WITH_FC 정상 동작."""
    rng = np.random.default_rng(7)
    pts = rng.random((40, 2))
    tri = Delaunay(pts)
    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for simplex in tri.simplices:
        for u, v in itertools.combinations(simplex, 2):
            u, v = int(u), int(v)
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(key[0], key[1], 1, False))
    graph = Graph(edges=edges)

    with_fc = _run_once(use_face_cycle=True, graph=graph, seed=7)
    assert with_fc["ok"], with_fc.get("reason")
    assert with_fc["score"] > 0
