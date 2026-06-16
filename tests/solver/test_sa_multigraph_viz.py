"""시각화 테스트 — SA 솔버가 멀티그래프(평행간선)를 붕괴 없이 방향짓는지 눈으로 확인.

n≈50 정점 Delaunay 그래프에 평행간선(같은 양끝점, 다른 가중치)을 일부 끼워 멀티그래프를
만들고, `SAMR2SSolver` 로 방향을 정한 뒤 PNG 로 렌더한다. 핵심 검증 포인트:

  - Solution.edges 가 edge_id 로 키잉되어 평행 same-direction 간선도 붕괴하지 않는다
    (해 간선 수 == 입력 간선 수, 평행 그룹별 개수 보존).
  - 시각화에서 *각 간선의 방향* 이 화살촉으로 명확하다. 평행간선은 곡률을 달리 줘
    서로 겹치지 않게 분리해 그린다.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

pytest.importorskip("scipy.spatial")

from mr2s_module.domain import Edge, Graph  # noqa: E402
from mr2s_module.reduction.degree_two_chain import DegreeTwoChainReducer  # noqa: E402
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver  # noqa: E402
from tests.util.graph_fixtures import delaunay_graph_with_pos  # noqa: E402

_OUTPUT_DIR = Path(__file__).parent / "output"


def _build_multigraph(
    n: int,
    seed: int,
) -> tuple[Graph, dict[int, np.ndarray], list[frozenset[int]]]:
    """Delaunay 그래프 + 평행간선 주입 → 멀티그래프. 평행 그룹의 endpoint_key 도 반환."""
    base, pos = delaunay_graph_with_pos(n, seed)

    undirected = sorted(
        (edge for edge in base.edges.values() if not edge.directed),
        key=lambda edge: edge.endpoints(),
    )
    # 결정적으로 일부 간선에 평행 사본을 추가(다른 가중치로 → 멀티그래프 의미 부여).
    parallel_keys: list[frozenset[int]] = []
    edges: list[Edge] = list(undirected)
    for index in range(0, len(undirected), 9):
        original = undirected[index]
        u, v = original.endpoints()
        edges.append(Edge(u, v, original.weight + 3, False))
        parallel_keys.append(original.endpoint_key())

    return Graph(edges=edges), pos, parallel_keys


def _curvatures(count: int) -> list[float]:
    """평행간선 개수에 맞춰 겹치지 않도록 곡률(rad) 분배."""
    if count == 1:
        return [0.0]
    return list(np.linspace(-0.35, 0.35, count))


def _render(
    graph: Graph,
    solution_edges: dict[int, Edge],
    pos: dict[int, np.ndarray],
    parallel_keys: set[frozenset[int]],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_title(title, fontsize=15)

    # 같은 양끝점을 잇는 간선들을 묶어 곡률을 다르게 줘 분리해 그린다.
    by_endpoint: dict[frozenset[int], list[Edge]] = defaultdict(list)
    for edge in solution_edges.values():
        by_endpoint[edge.endpoint_key()].append(edge)

    for endpoint_key, group in by_endpoint.items():
        is_parallel = endpoint_key in parallel_keys
        group = sorted(group, key=lambda edge: edge.id)
        for edge, rad in zip(group, _curvatures(len(group))):
            source, target = edge.vertices
            color = "#d62728" if is_parallel else "#2c5d8f"
            arrow = FancyArrowPatch(
                tuple(pos[source]),
                tuple(pos[target]),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=24 if is_parallel else 20,
                linewidth=2.6 if is_parallel else 1.7,
                color=color,
                alpha=0.97 if is_parallel else 0.85,
                shrinkA=9,
                shrinkB=11,
                zorder=3 if is_parallel else 2,
            )
            ax.add_patch(arrow)

    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    ax.scatter(xs, ys, color="#222222", s=70, zorder=4)
    for vertex, point in pos.items():
        ax.annotate(
            str(vertex),
            point,
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            zorder=5,
        )

    handles = [
        plt.Line2D([], [], color="#3b6ea5", lw=2, label="single edge (directed)"),
        plt.Line2D([], [], color="#d62728", lw=2.5, label="parallel edge (multigraph)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def test_sa_solver_orients_multigraph_without_collapse():
    graph, pos, parallel_key_list = _build_multigraph(n=50, seed=7)
    parallel_keys = set(parallel_key_list)

    input_group_sizes = defaultdict(int)
    for edge in graph.edges.values():
        input_group_sizes[edge.endpoint_key()] += 1
    assert any(size >= 2 for size in input_group_sizes.values()), "multigraph 입력이어야 함"

    solution = SAMR2SSolver(random_seed=7).run(graph)

    # 평행 same-direction 간선이 붕괴하지 않고 모두 보존됐는지(핵심).
    assert len(solution.edges) == len(graph.edges)
    solution_group_sizes = defaultdict(int)
    for edge in solution.edges.values():
        solution_group_sizes[edge.endpoint_key()] += 1
    assert solution_group_sizes == input_group_sizes

    # 모든 해 간선이 방향이 정해진 directed Edge 인지(시각화 화살표 보장).
    assert all(edge.directed for edge in solution.edges.values())

    output_path = _OUTPUT_DIR / "sa_multigraph_orientation.png"
    _render(
        graph,
        solution.edges,
        pos,
        parallel_keys,
        output_path,
        title=(
            f"SA multigraph orientation — {len(graph.get_vertices())} vertices, "
            f"{len(graph.edges)} directed edges "
            f"({len(parallel_keys)} parallel groups)"
        ),
    )
    assert output_path.exists() and output_path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 압축(차수-2 체인 축약) 추적 + 양방향 가중치 시각화
# --------------------------------------------------------------------------- #
def _build_hub_chain_multigraph() -> tuple[
    Graph, dict[int, np.ndarray], dict[int, np.ndarray]
]:
    """허브 링(차수-2 체인으로 연결) + 교차 코드(일부 평행) 멀티그래프.

    체인 내부 정점은 차수 2 → 축약 대상. 허브는 교차 코드 덕에 차수 3 → 체인 끝점.
    반환: graph, pos_full(허브+체인내부 좌표), pos_hub(허브만 — 축약 그래프 좌표).
    """
    hub_count = 10
    radius = 10.0
    pos: dict[int, np.ndarray] = {}
    for hub in range(hub_count):
        angle = 2.0 * math.pi * hub / hub_count
        pos[hub] = np.array([radius * math.cos(angle), radius * math.sin(angle)])

    edges: list[Edge] = []
    next_id = hub_count
    internal_counts = [3, 4, 5, 3, 4, 5, 3, 4, 5, 4]  # 합 40 → 총 50 정점
    for hub in range(hub_count):
        a, b = hub, (hub + 1) % hub_count
        pa, pb = pos[a], pos[b]
        direction = pb - pa
        normal = np.array([-direction[1], direction[0]])
        norm = float(np.linalg.norm(normal))
        normal = normal / norm if norm else normal
        prev = a
        count = internal_counts[hub]
        for j in range(1, count + 1):
            t = j / (count + 1)
            vid = next_id
            next_id += 1
            # 체인이 허브-허브 직선과 안 겹치게 살짝 활처럼 휘어 배치.
            pos[vid] = pa + direction * t + normal * 1.4 * math.sin(math.pi * t)
            edges.append(Edge(prev, vid, 1, False))
            prev = vid
        edges.append(Edge(prev, b, 1, False))

    # 교차 코드: 모든 허브 차수>=3 보장 + 일부 평행(다른 weight)로 양방향 소재.
    chords = [(0, 5, [1, 4]), (1, 6, [2]), (2, 7, [2, 5]), (3, 8, [2]), (4, 9, [1, 3])]
    for a, b, weights in chords:
        for weight in weights:
            edges.append(Edge(a, b, weight, False))

    pos_hub = {hub: pos[hub] for hub in range(hub_count)}
    return Graph(edges=edges), pos, pos_hub


def _edge_label_point(
    p_src: np.ndarray, p_tgt: np.ndarray, rad: float
) -> np.ndarray:
    """곡선 화살표 중점에서 곡률 방향으로 살짝 떨어진 라벨 위치."""
    mid = (p_src + p_tgt) / 2.0
    direction = p_tgt - p_src
    normal = np.array([-direction[1], direction[0]])
    norm = float(np.linalg.norm(normal))
    if norm:
        normal = normal / norm
    bow = 0.5 + 2.5 * abs(rad)
    return mid + normal * bow * (1.0 if rad >= 0 else -1.0)


def _draw_panel(
    ax,
    title: str,
    directed_edges,
    pos: dict[int, np.ndarray],
    *,
    chain_color: dict[frozenset[int], object],
    super_weight: dict[frozenset[int], int] | None,
) -> None:
    """한 패널에 directed 간선들을 그린다.

    chain_color: endpoint_key → 색 (체인/super-edge 추적). super_weight: super-edge
    endpoint_key → Σ가중치 라벨(축약 패널만). 평행/양방향 간선엔 각 weight 라벨.
    """
    ax.set_title(title, fontsize=13)
    groups: dict[frozenset[int], list[Edge]] = defaultdict(list)
    for edge in directed_edges:
        groups[edge.endpoint_key()].append(edge)

    for key, group in groups.items():
        group = sorted(group, key=lambda e: e.id)
        is_parallel = len(group) >= 2
        tracked = chain_color.get(key)
        for edge, rad in zip(group, _curvatures(len(group))):
            p_src, p_tgt = pos[edge.vertices[0]], pos[edge.vertices[1]]
            if tracked is not None:          # 체인 / super-edge: 추적색, 굵게
                color, linewidth, scale, zorder = tracked, 3.2, 22, 3
            elif is_parallel:                # 평행/양방향: 빨강
                color, linewidth, scale, zorder = "#d62728", 2.6, 24, 3
            else:                            # 단일: 회색
                color, linewidth, scale, zorder = "#7a7a7a", 1.5, 18, 2
            arrow = FancyArrowPatch(
                tuple(p_src),
                tuple(p_tgt),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=scale,
                linewidth=linewidth,
                color=color,
                alpha=0.95,
                shrinkA=7,
                shrinkB=9,
                zorder=zorder,
            )
            ax.add_patch(arrow)

            label = None
            if super_weight is not None and key in super_weight and not is_parallel:
                label = f"Σw={super_weight[key]}"   # 축약 super-edge 합가중치
            elif is_parallel:
                label = f"w={edge.weight}"               # 양방향/평행 각 가중치
            if label is not None:
                point = _edge_label_point(p_src, p_tgt, rad)
                ax.annotate(
                    label, point, color=color, fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75),
                )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, color="#222222", s=60, zorder=4)
    for vertex, point in pos.items():
        ax.annotate(str(vertex), point, color="white", fontsize=6,
                    ha="center", va="center", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")


def _render_reduction(
    result,
    reduced_solution,
    expanded: list[Edge],
    pos_full: dict[int, np.ndarray],
    pos_hub: dict[int, np.ndarray],
    path: Path,
) -> None:
    palette = plt.colormaps["tab10"].colors
    chain_edge_color: dict[frozenset[int], object] = {}   # 원본 체인 간선 key → 색 (좌)
    super_color: dict[frozenset[int], object] = {}        # super-edge key → 색 (우)
    super_weight: dict[frozenset[int], int] = {}
    for i, chain in enumerate(result.chains):
        color = palette[i % len(palette)]
        super_key = frozenset(chain.endpoints)
        super_color[super_key] = color
        super_weight[super_key] = chain.collapsed_weight
        for original_edge in chain.original_edges:
            chain_edge_color[original_edge.endpoint_key()] = color

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(26, 13))
    _draw_panel(
        ax_left,
        f"ORIGINAL — {len(pos_full)} vertices, {len(expanded)} directed edges",
        expanded, pos_full, chain_color=chain_edge_color, super_weight=None,
    )
    _draw_panel(
        ax_right,
        f"REDUCED — {len(pos_hub)} vertices, {len(reduced_solution.edges)} edges "
        f"({len(result.chains)} chains collapsed)",
        reduced_solution.edges.values(), pos_hub,
        chain_color=super_color, super_weight=super_weight,
    )

    handles = [
        plt.Line2D([], [], color="#1f77b4", lw=3,
                   label="collapsed chain (color matched L↔R, Σw on right)"),
        plt.Line2D([], [], color="#d62728", lw=2.5,
                   label="parallel / bidirectional (per-edge weight)"),
        plt.Line2D([], [], color="#7a7a7a", lw=1.5, label="single edge"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12)
    fig.suptitle(
        "Degree-2 chain reduction tracking + bidirectional edge weights",
        fontsize=16,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _key_tuples(edges) -> list[tuple[int, ...]]:
    return sorted(tuple(sorted(edge.endpoint_key())) for edge in edges)


def test_sa_reduction_tracks_chains_and_bidirectional_weights():
    graph, pos_full, pos_hub = _build_hub_chain_multigraph()

    result = DegreeTwoChainReducer().reduce(graph)
    assert len(result.chains) >= 1, "축약 체인이 실제로 생겨야 함"
    assert len(result.reduced_graph.edges) < len(graph.edges), "변수(간선) 절감"

    # super-edge 가중치 == 체인 원본 간선 weight 합 (압축 가중치 추적).
    reduced_weight_by_key = {
        e.endpoint_key(): e.weight for e in result.reduced_graph.edges.values()
    }
    for chain in result.chains:
        key = frozenset(chain.endpoints)
        assert chain.collapsed_weight == sum(e.weight for e in chain.original_edges)
        assert reduced_weight_by_key[key] == chain.collapsed_weight

    reduced_solution = SAMR2SSolver(random_seed=11).run(result.reduced_graph)
    oriented = [
        Edge(e.vertices[0], e.vertices[1], e.weight, True)
        for e in reduced_solution.edges.values()
    ]
    expanded = result.expand(oriented)

    # expand 가 원본 무방향 간선 전체를 정확히 1회씩 방향배정(평행 보존 포함).
    assert len(expanded) == len(graph.edges)
    assert _key_tuples(expanded) == _key_tuples(graph.edges.values())
    assert all(edge.directed for edge in expanded)

    output_path = _OUTPUT_DIR / "sa_reduction_bidirectional.png"
    _render_reduction(result, reduced_solution, expanded, pos_full, pos_hub, output_path)
    assert output_path.exists() and output_path.stat().st_size > 0
