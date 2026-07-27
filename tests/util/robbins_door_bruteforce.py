"""Boesch–Tindell (mixed-graph Robbins) 전수 열거 실험.

door-only pre-orientation 의 이론적 근거를 실험으로 보인다: 평면 그래프의
공유 간선(=인접 두 bounded face 사이의 간선 = door)에만 방향을 부여하고,
그 2^n 방향 조합을 전수 열거해 "모든 셀(bounded face)을 강연결로 완성할 수
있는" 유효 조합 수를 센다.

배경: Robbins 정리(무방향 bridgeless 그래프는 강연결 배향 가능)의 mixed-graph
확장(Boesch–Tindell). 각 셀은 하나의 면(사이클)이며, 공유 간선은 이미 방향이
고정된 arc, 비공유 간선은 양방향으로 자유롭게 완성할 수 있는 mixed graph 다.
underlying 그래프가 bridgeless 이므로, 아래 강연결 판정이 곧 "이 방향 고정
하에서 셀을 강연결로 완성 가능한가" 와 동치다.

실험 전용 스크립트 — 프로덕션(mr2s_module/)에 두지 않는다.
직접 실행: `python tests/util/robbins_door_bruteforce.py`
"""

from __future__ import annotations

import itertools
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")  # headless 저장 전용
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

SEED = 42
TARGET_EDGES = 200
TARGET_REMOVAL_RATIO = 0.4
N_THRESHOLD = 24  # 2^n 전수 열거 임계치
OUTPUT_PNG = "tests/util/robbins_door_bruteforce.png"


# ---------------------------------------------------------------------------
# [그래프 생성]
# ---------------------------------------------------------------------------
def build_delaunay(n_points: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """무작위 2D 점의 Delaunay 삼각분할 → 무방향 평면 그래프.

    평면 삼각분할 간선 수 ≈ 3V-6. TARGET_EDGES≈200 이면 V≈68.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((n_points, 2))
    tri = Delaunay(pts)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_points))
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            graph.add_edge(a, b)
    return graph, pts


def prune_keep_biconnected(
    graph: nx.Graph, removal_ratio: float, seed: int
) -> nx.Graph:
    """biconnected(=bridgeless) + planar 을 유지하며 간선을 무작위 제거.

    Delaunay 부분그래프이므로 planar 은 자동 유지. is_biconnected 가 유지되면
    bridge 가 없음이 보장된다. 목표 비율 도달 또는 더 제거 시 biconnected 가
    깨지면 중단.
    """
    graph = graph.copy()
    rng = np.random.default_rng(seed + 1)
    edges = list(graph.edges())
    rng.shuffle(edges)
    start_m = graph.number_of_edges()
    target_removed = int(start_m * removal_ratio)
    removed = 0
    for u, v in edges:
        if removed >= target_removed:
            break
        graph.remove_edge(u, v)
        if nx.is_biconnected(graph):
            removed += 1  # 제거 확정
        else:
            graph.add_edge(u, v)  # bridge/절단점 생김 → 되돌림
    return graph


# ---------------------------------------------------------------------------
# [셀과 공유 간선]
# ---------------------------------------------------------------------------
def enumerate_faces(
    embedding: nx.PlanarEmbedding,
) -> list[list[int]]:
    """PlanarEmbedding 의 모든 면을 정점 사이클 리스트로 열거.

    각 방향 half-edge (u,v) 를 한 번씩 방문하며 traverse_face 로 면을 수집.
    """
    faces: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for u, v in embedding.edges():
        if (u, v) not in visited:
            face = embedding.traverse_face(u, v, mark_half_edges=visited)
            faces.append(face)
    return faces


def polygon_area(cycle: list[int], pts: np.ndarray) -> float:
    """shoelace 로 다각형 면적(절댓값). 외곽 면 식별에 사용."""
    coords = pts[cycle]
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def face_edge_keys(cycle: list[int]) -> list[frozenset[int]]:
    """면 사이클 → 무방향 간선 키(frozenset) 리스트."""
    return [
        frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
        for i in range(len(cycle))
    ]


def extract_cells_and_shared(
    graph: nx.Graph, pts: np.ndarray
) -> tuple[list[list[int]], list[frozenset[int]]]:
    """bounded face(셀)들과 공유 간선(두 bounded face 사이 간선)을 추출."""
    ok, embedding = nx.check_planarity(graph)
    assert ok, "그래프가 planar 이 아님 (Delaunay 부분그래프여야 함)"

    faces = enumerate_faces(embedding)
    # 외곽 면 = 면적 최대. 나머지가 bounded face = 셀.
    areas = [polygon_area(f, pts) for f in faces]
    outer_idx = int(np.argmax(areas))
    cells = [f for i, f in enumerate(faces) if i != outer_idx]

    # 각 간선이 접하는 bounded face 수 → 2 이면 공유 간선.
    edge_face_count: dict[frozenset[int], int] = {}
    for cell in cells:
        for key in face_edge_keys(cell):
            edge_face_count[key] = edge_face_count.get(key, 0) + 1
    shared = sorted(
        (k for k, c in edge_face_count.items() if c == 2),
        key=lambda k: tuple(sorted(k)),
    )
    return cells, shared


# ---------------------------------------------------------------------------
# [강연결 판정]
# ---------------------------------------------------------------------------
def _reachable(adj: dict[int, list[int]], start: int) -> set[int]:
    seen = {start}
    dq = deque([start])
    while dq:
        x = dq.popleft()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                dq.append(y)
    return seen


def is_strongly_orientable(
    cell: list[int], fixed_dirs: dict[frozenset[int], tuple[int, int]]
) -> bool:
    """셀(면 사이클)을 강연결로 완성 가능한지 판정.

    공유 간선은 fixed_dirs 방향의 arc 1개, 비공유 간선은 양방향 arc 2개로 본
    mixed→digraph 변환 후, two-pass BFS 로 단일 SCC 여부 판정.

    underlying bridgeless 가 이미 보장되므로, 이 판정이 곧 "이 방향 고정
    하에서 셀을 강연결 배향으로 완성 가능한가" 와 동치다(Boesch–Tindell).
    """
    fwd: dict[int, list[int]] = {}
    bwd: dict[int, list[int]] = {}

    def add_arc(a: int, b: int) -> None:
        fwd.setdefault(a, []).append(b)
        bwd.setdefault(b, []).append(a)

    verts = set(cell)
    for key in face_edge_keys(cell):
        u, v = tuple(key)  # 무방향 키
        if key in fixed_dirs:
            a, b = fixed_dirs[key]  # 고정 방향 arc 1개
            add_arc(a, b)
        else:
            add_arc(u, v)  # 자유 간선 → 양방향
            add_arc(v, u)

    start = next(iter(verts))
    if _reachable(fwd, start) != verts:
        return False
    return _reachable(bwd, start) == verts


# ---------------------------------------------------------------------------
# [전수 열거]
# ---------------------------------------------------------------------------
def brute_force_enumerate(
    cells: list[list[int]], shared: list[frozenset[int]]
) -> tuple[int, list[int] | None, float]:
    """공유 간선 2^n 방향 조합 전수 순회 → 유효 조합 수/예시/소요시간."""
    n = len(shared)
    # 각 공유 간선의 canonical 두 방향: bit 0 → (a,b), bit 1 → (b,a).
    endpoints = [tuple(sorted(k)) for k in shared]

    valid_count = 0
    example: list[int] | None = None
    t0 = time.perf_counter()
    for combo in itertools.product((0, 1), repeat=n):
        fixed_dirs = {
            shared[i]: (
                (endpoints[i][0], endpoints[i][1])
                if combo[i] == 0
                else (endpoints[i][1], endpoints[i][0])
            )
            for i in range(n)
        }
        if all(is_strongly_orientable(cell, fixed_dirs) for cell in cells):
            valid_count += 1
            if example is None:
                example = list(combo)
    elapsed = time.perf_counter() - t0
    return valid_count, example, elapsed


# ---------------------------------------------------------------------------
# [시각화]
# ---------------------------------------------------------------------------
def visualize(
    graph: nx.Graph,
    pts: np.ndarray,
    cells: list[list[int]],
    shared: list[frozenset[int]],
    example: list[int] | None,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    shared_set = set(shared)

    # 셀(면)마다 다른 색으로 채워 면 분할을 눈으로 구분.
    cmap = plt.get_cmap("tab20")
    for ci, cell in enumerate(cells):
        centroid = pts[cell].mean(axis=0)
        poly = plt.Polygon(
            pts[cell], closed=True, facecolor=cmap(ci % 20),
            edgecolor="none", alpha=0.45, zorder=0,
        )
        ax.add_patch(poly)
        ax.text(
            centroid[0], centroid[1], str(ci),
            ha="center", va="center", fontsize=9, color="#222",
            zorder=5,
        )

    # 간선: 공유=빨강 굵게, 비공유=옅은 회색.
    for u, v in graph.edges():
        is_shared = frozenset((u, v)) in shared_set
        ax.plot(
            [pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]],
            color="#d62728" if is_shared else "#cccccc",
            linewidth=2.2 if is_shared else 0.8,
            zorder=2 if is_shared else 1,
        )

    # 유효 조합이 있으면 공유 간선 화살표 배향 겹쳐 그리기.
    if example is not None:
        endpoints = [tuple(sorted(k)) for k in shared]
        for i, bit in enumerate(example):
            a, b = endpoints[i] if bit == 0 else endpoints[i][::-1]
            pa, pb = pts[a], pts[b]
            ax.annotate(
                "", xy=pb, xytext=pa,
                arrowprops=dict(arrowstyle="-|>", color="#1a5e1a", lw=1.8),
                zorder=3,
            )

    ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#333333", zorder=4)
    ax.set_aspect("equal")
    ax.set_title("Boesch–Tindell brute-force: shared edges (red) oriented")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# [드라이버]
# ---------------------------------------------------------------------------
def build_instance(
    n_points: int, seed: int
) -> tuple[nx.Graph, np.ndarray, list[list[int]], list[frozenset[int]]]:
    raw, pts = build_delaunay(n_points, seed)
    graph = prune_keep_biconnected(raw, TARGET_REMOVAL_RATIO, seed)
    cells, shared = extract_cells_and_shared(graph, pts)
    return graph, pts, cells, shared


def main() -> None:
    # V≈68 → 3V-6≈198 간선. prune 후 공유 간선 n 을 임계치 이하로 낮춘다.
    n_points = 68
    graph, pts, cells, shared = build_instance(n_points, SEED)

    # n 이 임계치 초과면 정점 수를 줄여 재생성 (n ≤ N_THRESHOLD 까지).
    while len(shared) > N_THRESHOLD and n_points > 8:
        print(
            f"[warn] n={len(shared)} > {N_THRESHOLD}; "
            f"정점 {n_points}→{n_points - 6} 로 재생성"
        )
        n_points -= 6
        graph, pts, cells, shared = build_instance(n_points, SEED)

    n = len(shared)
    combos = 1 << n
    print("=" * 60)
    print(f"정점 수      : {graph.number_of_nodes()}")
    print(f"간선 수      : {graph.number_of_edges()}")
    print(f"셀(bounded face) 수 : {len(cells)}")
    print(f"공유 간선 n  : {n}")
    print(f"2^n 조합     : {combos:,}")
    est = combos * len(cells) * 3e-6
    print(f"예상 소요    : ~{est:.1f}s (러프 추정)")
    print("=" * 60)

    valid_count, example, elapsed = brute_force_enumerate(cells, shared)

    print(f"유효 조합 수 : {valid_count:,} / {combos:,}")
    print(f"전수 열거 소요: {elapsed:.3f}s")
    if example is not None:
        print(f"유효 예시 bits: {example}")
    else:
        print("유효 조합 없음")

    visualize(graph, pts, cells, shared, example, OUTPUT_PNG)
    print(f"시각화 저장  : {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
