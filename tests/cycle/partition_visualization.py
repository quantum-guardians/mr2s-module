from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mr2s_module.domain import Edge, Graph, GraphPartitionResult
from mr2s_module.protocols import FaceCycleProtocol
from mr2s_module.util import face_edges, inner_faces
from tests.util.graph_fixtures import delaunay_graph_with_pos


BACKGROUND_COLOR = "#bcbcbc"


def render_face_cycle_partition_png(
    graph: Graph,
    pos: dict[int, np.ndarray],
    face_cycle: FaceCycleProtocol,
    path: Path,
    title: str,
) -> GraphPartitionResult:
    partition = face_cycle.run(graph)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_title(title, fontsize=14)
    draw_partition(ax, graph, partition, pos)
    ax.set_aspect("equal")
    ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return partition


def partition_balance_report(partition: GraphPartitionResult) -> dict[str, object]:
    sizes = [len(sub_graph.get_vertices()) for sub_graph in partition.sub_graphs]
    if not sizes:
        return {
            "sizes": [],
            "target": 0.0,
            "max_deviation": 0.0,
            "mean_deviation": 0.0,
            "score": 0.0,
        }

    target = sum(sizes) / len(sizes)
    deviations = [abs(size - target) for size in sizes]
    max_deviation = max(deviations)
    mean_deviation = sum(deviations) / len(deviations)
    score = 100.0 if target == 0 else 100.0 * target / (target + mean_deviation)
    return {
        "sizes": sizes,
        "target": target,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "score": score,
    }


def draw_partition(
    ax,
    graph: Graph,
    partition: GraphPartitionResult,
    pos: dict[int, np.ndarray],
) -> None:
    palette = macro_palette(len(partition.sub_graphs))
    vertex_to_macro = _vertex_to_macro(partition.sub_graphs)

    _fill_partition_faces(ax, graph, partition, pos, palette)

    for edge in partition.remaining_edges:
        _plot_edge(ax, edge, pos, color=BACKGROUND_COLOR, alpha=0.5, linewidth=0.8, zorder=1)

    for macro_id, sub_graph in enumerate(partition.sub_graphs):
        color = palette[macro_id]
        for edge in sub_graph.edges:
            if not edge.directed:
                _plot_edge(ax, edge, pos, color=color, alpha=0.55, linewidth=1.0, zorder=2)

    for macro_id, sub_graph in enumerate(partition.sub_graphs):
        color = palette[macro_id]
        for edge in sub_graph.edges:
            if edge.directed:
                _plot_edge(ax, edge, pos, color=color, alpha=0.98, linewidth=3.2, zorder=3)

    for vertex, point in pos.items():
        macro_id = vertex_to_macro.get(vertex)
        if macro_id is not None:
            ax.scatter(point[0], point[1], color=palette[macro_id], s=24, zorder=4)
        else:
            ax.scatter(point[0], point[1], color="#666666", s=18, zorder=4)


def macro_palette(num_macros: int) -> list:
    if num_macros <= 0:
        return []
    cmap = plt.colormaps.get_cmap("tab20")
    return [cmap(i / max(num_macros, 1)) for i in range(num_macros)]


def _vertex_to_macro(sub_graphs: list[Graph]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for macro_id, sub_graph in enumerate(sub_graphs):
        for vertex in sub_graph.get_vertices():
            mapping.setdefault(vertex, macro_id)
    return mapping


def _fill_partition_faces(
    ax,
    graph: Graph,
    partition: GraphPartitionResult,
    pos: dict[int, np.ndarray],
    palette: list,
) -> None:
    if not palette:
        return

    macro_edge_ids = [
        {edge.id for edge in sub_graph.edges}
        for sub_graph in partition.sub_graphs
    ]

    for face in inner_faces(graph, pos):
        owner = _best_face_owner(face_edges(face), macro_edge_ids)
        if owner is None:
            continue
        polygon = plt.Polygon(
            [pos[vertex] for vertex in face],
            facecolor=palette[owner],
            edgecolor="none",
            alpha=0.24,
            zorder=0,
        )
        ax.add_patch(polygon)


def _best_face_owner(
    face_edges: set[tuple[int, int]],
    macro_edge_ids: list[set[tuple[int, int]]],
) -> int | None:
    scored = [
        (len(face_edges.intersection(edge_ids)), -macro_id, macro_id)
        for macro_id, edge_ids in enumerate(macro_edge_ids)
    ]
    if not scored:
        return None
    best_count, _, best_macro = max(scored)
    if best_count == 0:
        return None
    return best_macro


def _plot_edge(
    ax,
    edge: Edge,
    pos: dict[int, np.ndarray],
    *,
    color,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    u, v = edge.id
    if u == v:
        return
    ax.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
