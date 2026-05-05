"""
파티션 시각화 테스트 — `FaceCycle.run()` 가 반환한 `GraphPartitionResult` 를
면 단위로 색상화하여 PNG 로 저장한다.

각 거대 면(macro) 마다 고유한 색을 부여하고:
  - 면 외각선(directed `remaining_edges`)        → 그 면의 강한 색 (굵게)
  - 면 내부의 간선(`sub_graphs[i].edges`)        → 그 면의 약한 색 (얇게)
  - 어떤 면에도 닿지 않는 간선(undirected remain) → 회색 (얇은 배경)
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module.cycle import FaceCycle
from mr2s_module.domain import Edge, Graph, GraphPartitionResult

_OUTPUT_DIR = Path(__file__).parent / "output"
_BACKGROUND_COLOR = "#bcbcbc"


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


def _macro_palette(num_macros: int) -> list:
    """qualitative 컬러맵에서 macro 개수만큼 균등 추출."""
    if num_macros <= 0:
        return []
    cmap = plt.colormaps.get_cmap("tab20")
    return [cmap(i / max(num_macros, 1)) for i in range(num_macros)]


def _vertex_to_macro(sub_graphs: list[Graph]) -> dict[int, int]:
    """정점 → 가장 먼저 나타난 macro id (경계 정점은 한 macro 에만 귀속)."""
    mapping: dict[int, int] = {}
    for macro_id, sg in enumerate(sub_graphs):
        for v in sg.get_vertices():
            mapping.setdefault(v, macro_id)
    return mapping


def _draw_partition(
    ax,
    partition: GraphPartitionResult,
    pos: dict[int, np.ndarray],
) -> None:
    palette = _macro_palette(len(partition.sub_graphs))
    vertex_to_macro = _vertex_to_macro(partition.sub_graphs)

    # 1) macro 와 무관한 remaining edge (bridge / 외톨이) — 옅은 회색 배경
    for edge in partition.remaining_edges:
        u, v = edge.id
        if u == v or edge.directed:
            continue
        if u in vertex_to_macro or v in vertex_to_macro:
            # macro 정점에 닿지만 외각선은 아님 — 약한 색으로 처리
            macro_id = vertex_to_macro.get(u, vertex_to_macro.get(v))
            color, alpha, lw = palette[macro_id], 0.30, 0.9
        else:
            color, alpha, lw = _BACKGROUND_COLOR, 0.5, 0.8
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            alpha=alpha,
            linewidth=lw,
            zorder=1,
        )

    # 2) sub_graph edges — 면 내부의 약한 색, 얇게
    for macro_id, sg in enumerate(partition.sub_graphs):
        color = palette[macro_id]
        for edge in sg.edges:
            u, v = edge.id
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=color,
                alpha=0.55,
                linewidth=1.0,
                zorder=2,
            )

    # 3) directed remaining edges — 면의 외각선, 두껍게 강조
    for edge in partition.remaining_edges:
        u, v = edge.id
        if u == v or not edge.directed:
            continue
        macro_id = vertex_to_macro.get(u, vertex_to_macro.get(v))
        color = palette[macro_id] if macro_id is not None else _BACKGROUND_COLOR
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            alpha=0.98,
            linewidth=3.2,
            zorder=3,
        )

    # 4) 정점 — macro 에 속한 정점은 그 색으로, 외톨이는 회색
    for v, p in pos.items():
        macro_id = vertex_to_macro.get(v)
        if macro_id is not None:
            ax.scatter(p[0], p[1], color=palette[macro_id], s=24, zorder=4)
        else:
            ax.scatter(p[0], p[1], color="#666666", s=18, zorder=4)


@pytest.mark.parametrize("seed,n_points,target_k", [(42, 60, 6), (7, 80, 8)])
def test_face_cycle_partition_visualization_renders(
    seed: int, n_points: int, target_k: int
) -> None:
    graph, pos = _delaunay_graph_with_pos(n=n_points, seed=seed)

    np.random.seed(seed)
    partition = FaceCycle(target_k=target_k).run(graph)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_title(
        f"FaceCycle partition — n={n_points}, target_k={target_k}, "
        f"macros={len(partition.sub_graphs)}",
        fontsize=14,
    )
    _draw_partition(ax, partition, pos)
    ax.set_aspect("equal")
    ax.axis("off")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        _OUTPUT_DIR
        / f"face_cycle_partition_seed{seed}_n{n_points}_k{target_k}.png"
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
