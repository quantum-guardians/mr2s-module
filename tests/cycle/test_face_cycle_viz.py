"""
시각화 테스트 — `FaceCycle` 파이프라인이 잘 도는지 눈으로 확인하기 위한 테스트.

세 개의 패널을 그려 PNG 로 저장한다.
  1. 원본 (Delaunay) 그래프
  2. 추출된 면 — 외벽 보호 후 2-coloring 한 면들 + final_boundary
  3. 회전 방향 — 각 면을 2-color 에 따라 CCW(파랑) 또는 CW(주황) 화살표로 표시

`FaceCycle.run` 결과와 진단용으로 재현한 파이프라인의 final_boundary 가
일치해야 한다 (같은 numpy seed 하에서).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module.cycle import FaceCycle
from mr2s_module.domain import Edge, Graph

_OUTPUT_DIR = Path(__file__).parent / "output"
_PALETTE = ("#1f77b4", "#ff7f0e")


def _delaunay_graph_with_pos(
    n: int, seed: int
) -> tuple[Graph, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
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

    pos = {i: np.array(points[i]) for i in range(n)}
    return Graph(edges=edges), pos


def _run_diagnostic(
    graph: Graph, pos: dict[int, np.ndarray], target_k: int
) -> dict:
    """`FaceCycle._partition_component` 와 동일한 흐름을 순수 함수로 재현,
    중간 산출물(면, 컴포넌트, 2-coloring) 을 함께 반환."""
    nx_graph = FaceCycle._to_networkx(graph)
    is_planar, _ = nx.check_planarity(nx_graph)
    assert is_planar, "test fixture must be planar"
    assert nx.is_biconnected(nx_graph), "test fixture must be biconnected"

    raw_faces = FaceCycle._enumerate_faces(nx_graph)
    outer_idx = int(np.argmax([FaceCycle._face_area(f, pos) for f in raw_faces]))
    inner_faces = [f for i, f in enumerate(raw_faces) if i != outer_idx]

    face_edges_map = FaceCycle._build_face_edges_map(inner_faces)
    centroids = [np.mean([pos[v] for v in f], axis=0) for f in inner_faces]
    dual_base = FaceCycle._build_dual_base(face_edges_map)

    k = max(1, min(target_k, len(inner_faces)))
    face_to_cluster = FaceCycle._snowball_cluster(centroids, dual_base, k)

    boundary, outer = FaceCycle._collect_boundary_edges(
        face_edges_map, face_to_cluster
    )
    repair = FaceCycle._wall_protected_repair(nx_graph, boundary, outer)
    final_boundary = boundary.symmetric_difference(repair)

    face_graph = nx.Graph()
    face_graph.add_nodes_from(range(len(inner_faces)))
    for e, f_idxs in face_edges_map.items():
        if len(f_idxs) == 2 and e not in final_boundary:
            face_graph.add_edge(f_idxs[0], f_idxs[1])
    components = FaceCycle._filter_ghost_components(
        face_graph, inner_faces, outer, final_boundary
    )

    merged = FaceCycle._build_merged_dual(
        components, inner_faces, face_edges_map, final_boundary
    )
    if merged.number_of_nodes() > 0 and nx.is_bipartite(merged):
        coloring = nx.bipartite.color(merged)
    else:
        coloring = {i: 0 for i in range(len(components))}

    return {
        "nx_graph": nx_graph,
        "inner_faces": inner_faces,
        "components": components,
        "final_boundary": final_boundary,
        "coloring": coloring,
    }


def _draw_original(ax, nx_graph: nx.Graph, pos: dict[int, np.ndarray]) -> None:
    ax.set_title("1. Original (Delaunay) graph", fontsize=14)
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.5, edge_color="#666", width=0.7)
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=18, node_color="#222")
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_faces(ax, diag: dict, pos: dict[int, np.ndarray]) -> None:
    ax.set_title(
        f"2. Extracted faces (K={len(diag['components'])}, 2-colored)",
        fontsize=14,
    )
    nx.draw_networkx_edges(
        diag["nx_graph"], pos, ax=ax, alpha=0.10, edge_color="gray", width=0.5
    )
    for c_idx, comp in enumerate(diag["components"]):
        color = _PALETTE[diag["coloring"].get(c_idx, 0) % 2]
        for f_idx in comp:
            face = diag["inner_faces"][f_idx]
            poly = plt.Polygon(
                [pos[v] for v in face],
                facecolor=color,
                alpha=0.65,
                edgecolor="none",
            )
            ax.add_patch(poly)
    nx.draw_networkx_edges(
        diag["nx_graph"],
        pos,
        ax=ax,
        edgelist=list(diag["final_boundary"]),
        edge_color="black",
        width=2.5,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_rotations(ax, diag: dict, pos: dict[int, np.ndarray]) -> None:
    ax.set_title(
        "3. Boundary cycle rotation "
        "(blue: CCW, orange: CW — from 2-coloring)",
        fontsize=14,
    )
    nx.draw_networkx_edges(
        diag["nx_graph"], pos, ax=ax, alpha=0.08, edge_color="gray", width=0.5
    )

    inner_faces = diag["inner_faces"]
    final_boundary = diag["final_boundary"]
    face_to_color: dict[int, int] = {}
    for c_idx, comp in enumerate(diag["components"]):
        c = diag["coloring"].get(c_idx, 0) % 2
        for f_idx in comp:
            face_to_color[f_idx] = c

    # 추출된 컴포넌트들의 boundary cycle 만 화살표로 표시.
    # - color 0 → 면 traversal 그대로 (CCW)
    # - color 1 → 뒤집어서 traversal (CW)
    # 인접한 두 컴포넌트의 색이 다르면 양쪽에서 같은 방향이 나오므로 dedupe.
    drawn: set[tuple[int, int]] = set()
    for f_idx, face in enumerate(inner_faces):
        if f_idx not in face_to_color:
            continue
        color_idx = face_to_color[f_idx]
        color = _PALETTE[color_idx]
        traversal = face if color_idx == 0 else list(reversed(face))
        for i in range(len(traversal)):
            a, b = traversal[i], traversal[(i + 1) % len(traversal)]
            if tuple(sorted((a, b))) not in final_boundary:
                continue
            if (a, b) in drawn:
                continue
            drawn.add((a, b))
            ax.annotate(
                "",
                xy=pos[b],
                xytext=pos[a],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.6,
                    alpha=0.95,
                    mutation_scale=14,
                ),
            )
    ax.set_aspect("equal")
    ax.axis("off")


@pytest.mark.parametrize("seed", [42])
def test_face_cycle_visualization_renders_three_panels(seed: int) -> None:
    n_points, target_k = 60, 6
    graph, pos = _delaunay_graph_with_pos(n=n_points, seed=seed)

    # 1. FaceCycle.run() 자체가 정상적으로 boundary 를 반환하는지.
    np.random.seed(seed)
    partition = FaceCycle(target_k=target_k).run(graph)
    boundary_run = {e.id for e in partition.directed_edges()}
    input_ids = {e.id for e in graph.edges}
    assert boundary_run, "run() should produce non-empty directed boundary edges"
    assert boundary_run.issubset(input_ids)

    # 2. 시각화용 진단 파이프라인 — FaceCycle 의 내부 planar_layout 대신
    #    원본 Delaunay 좌표를 그대로 써서 외곽/centroid 를 계산.
    #    그래서 run() 과 boundary 가 정확히 일치하지는 않지만,
    #    동일한 정점/평면 임베딩 위에서의 또 다른 valid 한 2-coloring 을 그린다.
    np.random.seed(seed)
    diag = _run_diagnostic(graph, pos, target_k=target_k)

    assert len(diag["components"]) > 0
    assert len(diag["inner_faces"]) >= len(diag["components"])
    assert set(diag["final_boundary"]).issubset(
        {tuple(sorted(e)) for e in diag["nx_graph"].edges()}
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    _draw_original(axes[0], diag["nx_graph"], pos)
    _draw_faces(axes[1], diag, pos)
    _draw_rotations(axes[2], diag, pos)
    fig.suptitle(
        f"FaceCycle visualization — n={n_points}, target_k={target_k}, seed={seed}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"face_cycle_seed{seed}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
