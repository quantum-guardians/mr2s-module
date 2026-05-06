"""
Flow 넣다뺐다 벤치마크 — 간선 제거/유지 상태에서 flow score 비교 시각화.

x축: (정점 개수, edge 제거 비율)
y축: flow score — "넣다"(전체 그래프)와 "뺐다"(간선 제거 그래프) 동시 표시

PNG 로 저장한다.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module import (
    ApspSumRanker,
    Edge,
    Evaluator,
    FlowPolyGenerator,
    Graph,
    SAQuboSolver,
)
from mr2s_module.domain import Solution
from mr2s_module.util.qubo_util import map_binary_poly_to_bqm

_OUTPUT_DIR = Path(__file__).parent / "output"

VERTEX_COUNTS = [10, 20, 30, 40, 50]
REMOVE_RATIOS = [0.0, 0.2, 0.4, 0.6]
NUM_READS = 50
SEED = 42


def _build_planar_graph(num_points: int, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    points = rng.random((num_points, 2))
    tri = Delaunay(points)

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

    return Graph(edges=edges)


def _thin_graph(graph: Graph, seed: int, remove_ratio: float) -> Graph:
    if remove_ratio <= 0.0:
        return graph

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge.id for edge in graph.edges)

    rng = np.random.default_rng(seed)
    edges = list(nx_graph.edges())
    rng.shuffle(edges)

    target_edge_count = int(len(edges) * (1.0 - remove_ratio))

    for u, v in edges:
        if nx_graph.number_of_edges() <= target_edge_count:
            break
        nx_graph.remove_edge(u, v)
        if not nx.is_biconnected(nx_graph):
            nx_graph.add_edge(u, v)

    nx_graph.remove_nodes_from(
        [v for v in nx_graph.nodes() if nx_graph.degree(v) == 0]
    )

    return Graph(edges=[
        Edge(min(u, v), max(u, v), 1, False)
        for u, v in nx_graph.edges()
    ])


def _solve_and_eval_flow(graph: Graph, num_reads: int) -> float:
    """그래프에 FlowPoly QUBO를 풀고 flow score를 반환한다."""
    polynomial = FlowPolyGenerator().run(graph)
    bqm = map_binary_poly_to_bqm(polynomial)

    solver = SAQuboSolver(ranker=ApspSumRanker())
    solution = solver.run(bqm, graph)

    return Evaluator().eval_flow(solution)


@dataclass
class FlowResult:
    vertex_count: int
    remove_ratio: float
    flow_full: float
    flow_thinned: float


@pytest.mark.slow
def test_flow_insert_remove_benchmark() -> None:
    """정점 개수 × 제거 비율별 flow score (넣다 vs 뺐다) 벤치마크."""
    results: list[FlowResult] = []

    for n in VERTEX_COUNTS:
        full_graph = _build_planar_graph(n, SEED)
        flow_full = _solve_and_eval_flow(full_graph, NUM_READS)

        for ratio in REMOVE_RATIOS:
            if ratio == 0.0:
                flow_thinned = flow_full
            else:
                thinned = _thin_graph(full_graph, SEED, ratio)
                flow_thinned = _solve_and_eval_flow(thinned, NUM_READS)

            results.append(FlowResult(
                vertex_count=n,
                remove_ratio=ratio,
                flow_full=flow_full,
                flow_thinned=flow_thinned,
            ))

    _draw_chart(results)

    for r in results:
        assert r.flow_full >= 0.0
        assert r.flow_thinned >= 0.0


def _draw_chart(results: list[FlowResult]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x_labels = [f"V={r.vertex_count}\nR={r.remove_ratio}" for r in results]
    x = np.arange(len(results))
    width = 0.35

    bars_full = [r.flow_full for r in results]
    bars_thinned = [r.flow_thinned for r in results]

    ax.bar(x - width / 2, bars_full, width, label="Insert (full graph)", color="#1f77b4")
    ax.bar(x + width / 2, bars_thinned, width, label="Remove (thinned graph)", color="#ff7f0e")

    ax.set_xlabel("Vertex count (V) / Edge removal ratio (R)")
    ax.set_ylabel("Flow Score")
    ax.set_title("Flow Insert-Remove Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "flow_insert_remove_benchmark.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
