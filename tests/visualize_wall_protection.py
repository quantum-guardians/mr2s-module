"""Wall-protected T-join repair: ON vs OFF visualization.

FaceClusterPartition 의 boundary 패리티 수리(_wall_protected_repair)는 홀수 차수
정점을 짝수로 만들기 위한 min-weight T-join 이다. 외벽 보호 ON 은 외곽 간선에
거대 가중치(999999)를 줘 수리 경로가 외벽을 우회하도록 강제한다.

이 스크립트는 동일한 면 군집을 고정한 채 repair 경로만 두 가지로 계산해 비교한다.
  - 왼쪽  : 외벽 보호 ON  (현재 동작, 외곽 간선 가중치 999999)
  - 오른쪽: 외벽 보호 OFF (균일 가중치 1 — 수리 경로가 외벽을 가로지를 수 있음)

각 패널:
  - 면 군집을 옅은 색으로 채움
  - 외벽(outer wall) 간선: 굵은 검정
  - 수리 후 최종 boundary: 파랑
  - 수리 간선(내륙): 초록
  - 외벽을 파괴한 수리 간선(repair ∩ outer): 굵은 빨강  ← OFF 의 피해
제목에 홀수 정점 수 / 수리 간선 수 / 외벽 파괴 간선 수 / 최종 macro 수를 표기한다.

모든 간선 집합은 도메인 edge id 로 다룬다 — 파이프라인 본체와 같은 모델이라
평행 간선이 붕괴하지 않는다. 면 열거는 edge subdivision 위에서 한다.

Usage:
  python tests/visualize_wall_protection.py
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# 한글 깨짐 방지: 한글 지원 폰트가 있으면 사용
for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
    try:
        font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mr2s_module import FaceClusterPartition, Graph
from mr2s_module.cycle.face_clusterer import SnowballFaceClusterer
from mr2s_module.util.planar_graph import (
    build_dual_base,
    build_edge_id_face_edges_map,
    domain_graph_to_edge_subdivision,
    enumerate_faces,
    face_edge_steps,
    face_vertex_ring,
    is_edge_node,
    polygon_area,
)
from tests.util.graph_fixtures import delaunay_graph_with_pos

OUTPUT_DIR = Path(__file__).resolve().parent / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTICES = 60
TARGET_K = 4


# ── unprotected repair (외벽 보호 OFF) ────────────────────────────


def unprotected_repair(
    g_euler: nx.Graph,
    boundary_edges: set[int],
    outer_edges: set[int],
) -> set[int]:
    """_wall_protected_repair 와 동일하되 외곽 간선에 가중치 패널티를 주지 않는다."""
    del outer_edges  # 균일 가중치 — 외벽을 특별 취급하지 않음
    edge_endpoints = FaceClusterPartition._edge_endpoints(g_euler)
    b_sub = nx.MultiGraph()
    for edge_id in boundary_edges:
        u, v = edge_endpoints[edge_id]
        b_sub.add_edge(u, v, key=edge_id)
    odd_nodes = [v for v, d in b_sub.degree() if d % 2 != 0]
    if not odd_nodes:
        return set()

    g_repair = g_euler.copy()
    for u, v in g_repair.edges():
        g_repair[u][v]["weight"] = 1

    dist_map = dict(nx.all_pairs_dijkstra_path_length(g_repair, weight="weight"))
    complete = nx.Graph()
    for u, v in itertools.combinations(odd_nodes, 2):
        if v in dist_map.get(u, {}):
            complete.add_edge(u, v, weight=dist_map[u][v])

    repair_edges: set[int] = set()
    for u, v in nx.min_weight_matching(complete):
        path = nx.shortest_path(g_repair, u, v, weight="weight")
        for node in path:
            if is_edge_node(node):
                repair_edges.add(node[1])
    return repair_edges


class NoWallProtection(FaceClusterPartition):
    """외벽 보호를 끈 전체 파이프라인 (최종 macro 수 비교용)."""

    @staticmethod
    def _wall_protected_repair(g_euler, boundary_edges, outer_edges):
        return unprotected_repair(g_euler, boundary_edges, outer_edges)


# ── intermediate state (clustering 고정, repair 두 종류) ──────────


def largest_biconnected(graph: Graph) -> nx.Graph | None:
    """면 분할이 도는 최대 이중연결 컴포넌트 (edge subdivision 위에서)."""
    subdivision = domain_graph_to_edge_subdivision(graph)
    if not nx.check_planarity(subdivision)[0]:
        return None
    comps = FaceClusterPartition()._extract_biconnected_components(subdivision)
    if not comps:
        return None
    return max(comps, key=lambda g: g.number_of_edges())


def compute_state(
    component: nx.Graph, pos: dict[int, np.ndarray], seed: int
) -> dict | None:
    raw_faces = enumerate_faces(component)
    if len(raw_faces) < 2:
        return None

    face_steps = [face_edge_steps(face) for face in raw_faces]
    rings = [face_vertex_ring(steps) for steps in face_steps]
    outer_idx = int(np.argmax([abs(polygon_area(ring, pos)) for ring in rings]))
    inner_face_steps = [f for i, f in enumerate(face_steps) if i != outer_idx]
    inner_rings = [r for i, r in enumerate(rings) if i != outer_idx]
    if not inner_face_steps:
        return None

    face_edges_map = build_edge_id_face_edges_map(inner_face_steps)
    centroids = [np.mean([pos[v] for v in ring], axis=0) for ring in inner_rings]
    dual_base = build_dual_base(face_edges_map)
    target_k = max(1, min(TARGET_K, len(inner_face_steps)))

    # ON/OFF 가 동일한 군집을 보도록 clustering 난수를 고정
    np.random.seed(seed)
    face_to_cluster = SnowballFaceClusterer().run(centroids, dual_base, target_k)

    boundary, outer = FaceClusterPartition._collect_boundary_edges(
        face_edges_map, face_to_cluster
    )
    repair_on = FaceClusterPartition._wall_protected_repair(component, boundary, outer)
    repair_off = unprotected_repair(component, boundary, outer)

    return {
        "pos": pos,
        "edge_endpoints": FaceClusterPartition._edge_endpoints(component),
        "inner_rings": inner_rings,
        "face_to_cluster": face_to_cluster,
        "boundary": boundary,
        "outer": outer,
        "repair_on": repair_on,
        "repair_off": repair_off,
        "final_on": boundary.symmetric_difference(repair_on),
        "final_off": boundary.symmetric_difference(repair_off),
    }


# ── drawing ──────────────────────────────────────────────────────

# .colors 는 스텁상 ArrayLike — 실제로는 RGB 튜플 시퀀스다.
COLORS = cast(
    "Sequence[tuple[float, float, float]]",
    cast(ListedColormap, matplotlib.colormaps["Set3"]).colors,
)


def _seg(ax, st, edge_id: int, **kw):
    pos = st["pos"]
    u, v = st["edge_endpoints"][edge_id]
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], **kw)


def draw_panel(
    ax, st, repair_edges: set[int], final_boundary: set[int], title: str
) -> int:
    pos = st["pos"]
    outer = st["outer"]
    f2c = st["face_to_cluster"]

    # 면 군집 채색
    for f_idx, ring in enumerate(st["inner_rings"]):
        c = f2c.get(f_idx)
        if c is None:
            continue
        pts = np.array([pos[v] for v in ring])
        ax.fill(
            pts[:, 0],
            pts[:, 1],
            color=COLORS[c % len(COLORS)],
            alpha=0.45,
            ec="none",
            zorder=0,
        )

    # 전체 그래프 연하게
    for edge_id in st["edge_endpoints"]:
        _seg(ax, st, edge_id, color="gray", alpha=0.12, linewidth=0.6, zorder=1)

    # 외벽 (검정)
    for edge_id in outer:
        _seg(ax, st, edge_id, color="black", linewidth=2.4, zorder=2)

    # 최종 boundary (파랑)
    for edge_id in final_boundary:
        _seg(ax, st, edge_id, color="#1565c0", linewidth=2.0, alpha=0.9, zorder=3)

    # 수리 간선: 외벽 파괴분은 빨강, 내륙분은 초록
    wall_damage = repair_edges & outer
    for edge_id in repair_edges - outer:
        _seg(ax, st, edge_id, color="#2e7d32", linewidth=2.6, alpha=0.95, zorder=4)
    for edge_id in wall_damage:
        _seg(ax, st, edge_id, color="#d32f2f", linewidth=4.0, alpha=0.95, zorder=5)

    xs = [pos[v][0] for v in pos]
    ys = [pos[v][1] for v in pos]
    ax.scatter(xs, ys, s=10, color="black", zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    return len(wall_damage)


def run_visualization(seed: int) -> bool:
    graph, pos = delaunay_graph_with_pos(N_VERTICES, seed)
    component = largest_biconnected(graph)
    if component is None:
        print(f"[seed {seed:02d}] non-planar / no bcc, skip")
        return False
    st = compute_state(component, pos, seed)
    if st is None:
        print(f"[seed {seed:02d}] < 2 faces, skip")
        return False

    # 전체 파이프라인 macro 수 (군집 난수 고정해 동일 조건)
    np.random.seed(seed)
    macros_on = len(FaceClusterPartition(target_k=TARGET_K).run(graph).sub_graphs)
    np.random.seed(seed)
    macros_off = len(NoWallProtection(target_k=TARGET_K).run(graph).sub_graphs)

    b_sub = nx.MultiGraph()
    for edge_id in st["boundary"]:
        u, v = st["edge_endpoints"][edge_id]
        b_sub.add_edge(u, v, key=edge_id)
    odd = sum(1 for _, d in b_sub.degree() if d % 2)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    draw_panel(
        axes[0],
        st,
        st["repair_on"],
        st["final_on"],
        f"외벽 보호 ON (현재 동작)\n"
        f"odd={odd}  repair={len(st['repair_on'])}  "
        f"wall-damage={len(st['repair_on'] & st['outer'])}  → macros={macros_on}",
    )
    dmg = draw_panel(
        axes[1],
        st,
        st["repair_off"],
        st["final_off"],
        f"외벽 보호 OFF (균일 가중치)\n"
        f"odd={odd}  repair={len(st['repair_off'])}  "
        f"wall-damage={len(st['repair_off'] & st['outer'])}  → macros={macros_off}",
    )

    legend = [
        Line2D([0], [0], color="black", lw=2.4, label="외벽(outer wall)"),
        Line2D([0], [0], color="#1565c0", lw=2.0, label="최종 boundary"),
        Line2D([0], [0], color="#2e7d32", lw=2.6, label="수리 간선(내륙)"),
        Line2D([0], [0], color="#d32f2f", lw=4.0, label="외벽 파괴 수리"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(
        f"Seed {seed} — T-join boundary repair: wall protection ON vs OFF",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fname = f"wall_protection_seed{seed:02d}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[seed {seed:02d}] odd={odd:2d}  "
        f"repair ON={len(st['repair_on']):2d} OFF={len(st['repair_off']):2d}  "
        f"wall-damage(OFF)={dmg:2d}  "
        f"macros ON={macros_on} OFF={macros_off}  → {fname}"
    )
    return dmg > 0 or macros_on != macros_off


def main() -> None:
    for f in OUTPUT_DIR.glob("wall_protection_*.png"):
        f.unlink()
    print(f"Output → {OUTPUT_DIR}/  (n={N_VERTICES}, k={TARGET_K})")
    interesting = 0
    for seed in range(12):
        if run_visualization(seed):
            interesting += 1
    print(f"Done. {interesting} seed(s) show wall damage or macro-count divergence.")


if __name__ == "__main__":
    main()
