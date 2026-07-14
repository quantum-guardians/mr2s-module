"""FaceClusterPartition → macro dual graph visualization.

For each random planar graph:
  1. Generate large Delaunay triangulation (80 vertices)
  2. Run FaceClusterPartition(target_k=2)
  3. Build macro dual: nodes = macros, edges = shared boundary edges
  4. Keep original partitioned subgraph on left, bipartite macro dual on right
  5. Save as PNG

Usage:
  python tests/visualize_macro_weak_dual.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Sequence
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mr2s_module import FaceClusterPartition, Edge, Graph
from matplotlib.colors import ListedColormap

from mr2s_module.util.planar_graph import (
    domain_graph_to_networkx,
    enumerate_faces,
    polygon_area,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTICES = 80


# ── graph generation ─────────────────────────────────────────────

def delaunay_graph(n: int, seed: int) -> tuple[Graph, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    tri = Delaunay(points)

    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                u, v = int(simplex[i]), int(simplex[j])
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
                edges.append(Edge(u, v, 1, False))
    pos = {i: points[i] for i in range(n)}
    return Graph(edges=edges), pos


# ── macro dual construction ──────────────────────────────────────

def build_macro_dual(sub_graphs: list[Graph]) -> nx.Graph:
    outline_keys: list[set[int]] = []
    for sg in sub_graphs:
        keys: set[int] = set()
        for e in sg.edges.values():
            if e.directed:
                keys.add(e.id)
        outline_keys.append(keys)

    dual = nx.Graph()
    dual.add_nodes_from(range(len(sub_graphs)))
    for i in range(len(sub_graphs)):
        for j in range(i + 1, len(sub_graphs)):
            if outline_keys[i] & outline_keys[j]:
                dual.add_edge(i, j)
    return dual


# ── face → macro assignment ─────────────────────────────────────

def assign_faces_to_macros(
    inner_faces: list[list[int]],
    sub_graphs: list[Graph],
) -> dict[int, int]:
    macro_internal: list[set[tuple[int, int]]] = []
    for sg in sub_graphs:
        keys: set[tuple[int, int]] = set()
        for e in sg.edges.values():
            if not e.directed:
                keys.add(e.endpoints())
        macro_internal.append(keys)

    assignment: dict[int, int] = {}
    for f_idx, face in enumerate(inner_faces):
        fe = {
            (min(face[i], face[(i + 1) % len(face)]), max(face[i], face[(i + 1) % len(face)]))
            for i in range(len(face))
        }
        best_m, best_n = -1, -1
        for m_idx, ikeys in enumerate(macro_internal):
            n = len(fe & ikeys)
            if n > best_n:
                best_n = n
                best_m = m_idx
        if best_m >= 0 and best_n > 0:
            assignment[f_idx] = best_m
    return assignment


def fill_unassigned(
    assignment: dict[int, int],
    inner_faces: list[list[int]],
) -> None:
    fem: dict[frozenset[int], list[int]] = {}
    for f_idx, face in enumerate(inner_faces):
        for i in range(len(face)):
            ek = frozenset({face[i], face[(i + 1) % len(face)]})
            fem.setdefault(ek, []).append(f_idx)
    dual = nx.Graph()
    for fi in fem.values():
        if len(fi) == 2:
            dual.add_edge(fi[0], fi[1])
    for f_idx in range(len(inner_faces)):
        if f_idx in assignment:
            continue
        visited = {f_idx}
        q = [f_idx]
        while q:
            cur = q.pop(0)
            if cur in assignment:
                assignment[f_idx] = assignment[cur]
                break
            for nbr in dual.neighbors(cur):
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)
        if f_idx not in assignment:
            assignment[f_idx] = 0


# ── drawing ──────────────────────────────────────────────────────

# .colors 는 스텁상 ArrayLike — 실제로는 RGB 튜플 시퀀스다.
COLORS = cast(
    "Sequence[tuple[float, float, float]]",
    cast(ListedColormap, matplotlib.colormaps["Set2"]).colors,
)

def _face_centroid(face: list[int],
                   pos: dict[int, np.ndarray]) -> np.ndarray:
    return np.mean([pos[v] for v in face], axis=0)


def draw_graph_panel(ax, nx_g, pos, inner_faces, raw_faces, outer_idx,
                     assignment, sub_graphs, title: str) -> None:
    """Draw original planar graph with face coloring + boundary highlight."""
    for f_idx, face in enumerate(inner_faces):
        m = assignment.get(f_idx)
        if m is None:
            continue
        pts = np.array([pos[v] for v in face])
        ax.fill(pts[:, 0], pts[:, 1],
                color=COLORS[m % len(COLORS)], alpha=0.25, ec="none")

    oface = raw_faces[outer_idx]
    opts = np.array([pos[v] for v in oface])
    ax.fill(opts[:, 0], opts[:, 1], color="white", alpha=0.08, ec="#ddd", lw=0.5)

    nx.draw_networkx_edges(nx_g, pos, ax=ax, alpha=0.1, width=0.5, edge_color="gray")

    for m_idx, sg in enumerate(sub_graphs):
        for e in sg.edges.values():
            if not e.directed:
                continue
            u, v = e.endpoints()
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=COLORS[m_idx % len(COLORS)],
                    linewidth=2.0, alpha=0.8, zorder=3)

    nx.draw_networkx_nodes(nx_g, pos, ax=ax,
                           node_size=12, node_color="black", linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def draw_dual_panel(ax, sub_graphs, assignment, inner_faces, pos,
                    title: str) -> None:
    """Draw macro dual with bipartite layout."""
    if len(sub_graphs) <= 1:
        ax.text(0.5, 0.5, f"macro dual: {len(sub_graphs)} macro(s)\n(no partition)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        return

    dual = build_macro_dual(sub_graphs)

    # macro centroids
    macro_pts: dict[int, list[np.ndarray]] = {m: [] for m in range(len(sub_graphs))}
    for f_idx, m in assignment.items():
        if m is not None and m in macro_pts:
            macro_pts[m].append(_face_centroid(inner_faces[f_idx], pos))

    raw_centroids = {
        m: np.mean(pts, axis=0) if pts else np.array([0.0, 0.0])
        for m, pts in macro_pts.items()
    }

    # bipartite coloring
    try:
        color_a, color_b = nx.bipartite.sets(dual)
        node_color_map = {}
        for n in color_a:
            node_color_map[n] = 0
        for n in color_b:
            node_color_map[n] = 1
    except nx.AmbiguousSolution:
        node_color_map = nx.greedy_color(dual, strategy="largest_first")
        color_a = {n for n, c in node_color_map.items() if c == 0}
        color_b = {n for n, c in node_color_map.items() if c != 0}

    # bipartite layout (horizontal, left=color0, right=color1)
    left_nodes = [n for n, c in node_color_map.items() if c == 0]
    bip_pos = nx.bipartite_layout(dual, left_nodes, align="horizontal", scale=2.0)

    ref_pts = np.array(list(raw_centroids.values()))
    center = ref_pts.mean(axis=0) if len(ref_pts) else np.array([0.5, 0.5])
    scale = max(1.0, np.ptp(ref_pts[:, 0]) * 0.6) if len(ref_pts) > 1 else 1.0

    final_pos = {}
    for n in dual.nodes():
        bp = bip_pos[n]
        final_pos[n] = center + bp * scale

    # node colors: left side darkened, right side bright
    node_c = []
    for n in dual.nodes():
        c = node_color_map.get(n, 0)
        base = np.array(COLORS[n % len(COLORS)])
        node_c.append(base * 0.7 if c == 0 else base)

    # edge labels: shared boundary count
    outline_keys = []
    for sg in sub_graphs:
        keys = set()
        for e in sg.edges.values():
            if e.directed:
                keys.add(e.id)
        outline_keys.append(keys)

    edge_labels = {}
    for i, j in dual.edges():
        shared = len(outline_keys[i] & outline_keys[j])
        edge_labels[(i, j)] = str(shared)

    nx.draw(dual, pos=final_pos, ax=ax,
            node_color=node_c, node_size=600,
            edge_color="#555", width=2.5, alpha=0.9,
            with_labels=True, font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(dual, pos=final_pos, ax=ax,
                                 edge_labels=edge_labels, font_size=8,
                                 label_pos=0.5)

    lc, rc = len(color_a), len(color_b)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\n({lc}L | {rc}R bipartite)", fontsize=9)


# ── per-run ──────────────────────────────────────────────────────

def run_visualization(seed: int) -> None:
    rng = np.random.default_rng(seed)
    graph, pos = delaunay_graph(n=N_VERTICES, seed=int(rng.integers(0, 2**31)))

    nx_g = domain_graph_to_networkx(graph)
    # 단순 투영 그래프(edge subdivision 이 아님)라 면은 원본 정점 링이다.
    raw_faces = cast("list[list[int]]", enumerate_faces(nx_g))
    if len(raw_faces) < 2:
        print(f"[{seed:02d}] < 2 faces, skip")
        return

    outer_idx = max(
        range(len(raw_faces)),
        key=lambda i: abs(polygon_area(raw_faces[i], pos)),
    )
    inner_faces = [f for i, f in enumerate(raw_faces) if i != outer_idx]
    if not inner_faces:
        print(f"[{seed:02d}] no inner faces, skip")
        return

    partition = FaceClusterPartition(target_k=2, repair_mode="toggle").run(graph)
    sg = partition.sub_graphs

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    if len(sg) <= 1:
        draw_graph_panel(axes[0], nx_g, pos, inner_faces, raw_faces, outer_idx,
                         {}, sg,
                         f"Seed {seed} — {len(inner_faces)} faces, no partition")
        draw_dual_panel(axes[1], sg, {}, inner_faces, pos, "Macro dual")
        fname = f"macro_dual_seed{seed:02d}_nopart.png"
    else:
        assignment = assign_faces_to_macros(inner_faces, sg)
        fill_unassigned(assignment, inner_faces)
        draw_graph_panel(axes[0], nx_g, pos, inner_faces, raw_faces, outer_idx,
                         assignment, sg,
                         f"Seed {seed} — {len(inner_faces)} faces, {len(sg)} macros")
        draw_dual_panel(axes[1], sg, assignment, inner_faces, pos,
                        "Macro dual (k=2)")
        fname = f"macro_dual_seed{seed:02d}.png"

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    status = f"{len(sg)} macros" if len(sg) > 1 else "NO PARTITION"
    print(f"[Seed {seed:02d}] {status} → {fname}")


def main() -> None:
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()
    print(f"Output → {OUTPUT_DIR}/  (n={N_VERTICES})")
    for seed in range(10):
        run_visualization(seed)
    print("Done.")


if __name__ == "__main__":
    main()
