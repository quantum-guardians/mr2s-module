"""실험용 그래프 인스턴스 생성·저장·복원.

그래프 계열(family)은 delaunay(기본)·grid·hexagonal·apollonian·voronoi 다섯 가지다. 모두 이중연결 평면
그래프를 만들고 같은 간선 제거를 거친다. 계열은 graph_id 에 들어가지 않고 그래프 디렉터리
(`data/graphs_{family}`)로 구분한다. grid/hexagonal 은 seed 와 무관한 격자라 제거 0% 그래프는
seed 마다 동일하다.

Delaunay 생성은 tests/util/graph_fixtures.delaunay_graph_with_pos 와 같은 알고리즘이다
(테스트가 간선 집합 동등성을 고정한다). 간선 제거는 이중연결(biconnected)을 유지하는
탐욕 1패스: 셔플 순서대로 하나씩 제거해 보고 이중연결이 깨지면 되돌린다. 목표 비율에
못 미칠 수 있으므로 실제 비율을 기록한다. 같은 seed 의 제거 순서가 비율과 무관하게
같으므로 p30 의 제거 집합은 p10 의 제거 집합을 포함한다(중첩 인스턴스).

간선 id 는 프로세스 전역 카운터라 저장하지 않는다. Delaunay 그래프는 단순 그래프이므로
(u, v) 쌍이 간선의 유일 키이고, to_domain_graph 는 저장된 간선 순서대로 Edge 를 만든다.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, Voronoi

from experiments import config
from mr2s_module.domain import Edge, Graph

SCHEMA_VERSION = 1
DEFAULT_GRAPH_DIR = Path(__file__).resolve().parent / "data" / "graphs"
DEFAULT_FAMILY = "delaunay"


def default_graph_dir(family: str) -> Path:
    if family == DEFAULT_FAMILY:
        return DEFAULT_GRAPH_DIR
    return DEFAULT_GRAPH_DIR.with_name(f"graphs_{family}")


@dataclass(frozen=True)
class GraphRecord:
    graph_id: str
    vertices_requested: int
    graph_seed: int
    remove_ratio_target: float
    remove_ratio_actual: float
    n_vertices: int
    n_edges: int
    n_edges_original: int
    weight: int
    pos: list[tuple[float, float]]  # index = vertex id
    edges: list[tuple[int, int]]  # (u < v), 정렬 고정
    family: str = DEFAULT_FAMILY

    @property
    def path_name(self) -> str:
        return f"{self.graph_id}.json"


def generate_delaunay(n: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """[0,1)^2 균등 난수 점 n 개의 Delaunay 삼각분할 그래프와 좌표."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    triangulation = Delaunay(points)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for simplex in triangulation.simplices:
        for u, v in itertools.combinations(simplex, 2):
            u, v = int(u), int(v)
            if u != v:
                graph.add_edge(min(u, v), max(u, v))
    return graph, points


def _relabel_sorted(graph: nx.Graph) -> tuple[nx.Graph, list]:
    """정점을 정렬 순서대로 0..n-1 로 다시 붙인다. (새 그래프, 원래 정점 목록)"""
    nodes = sorted(graph.nodes())
    return nx.relabel_nodes(graph, {node: i for i, node in enumerate(nodes)}), nodes


def generate_grid(n: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """정점 수가 n 에 가까운 rows×cols 사각 격자. 이분 그래프라 홀수 사이클이 없다."""
    rows = max(2, round(math.sqrt(n)))
    cols = max(2, round(n / rows))
    graph, nodes = _relabel_sorted(nx.grid_2d_graph(rows, cols))
    points = np.array([(i / (rows - 1), j / (cols - 1)) for i, j in nodes])
    return graph, points


def generate_hexagonal(n: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """정점 수가 n 에 가장 가까운 정사각형에 가까운(k-m ≤ 2) m×k 벌집 격자. 면이 전부 6각형.

    정점 수는 2(m+1)(k+1)-2 라 요청한 n 과 정확히 같지 않을 수 있다.
    """
    best = min(
        ((m, k) for m in range(1, 64) for k in range(m, m + 3)),
        key=lambda mk: (abs(2 * (mk[0] + 1) * (mk[1] + 1) - 2 - n), mk[1] - mk[0]),
    )
    lattice = nx.hexagonal_lattice_graph(*best)
    graph, nodes = _relabel_sorted(lattice)
    points = np.array([lattice.nodes[node]["pos"] for node in nodes], dtype=float)
    points /= points.max(axis=0)
    return graph, points


def generate_apollonian(n: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """무작위 Apollonian 네트워크: 삼각형에서 시작해 균등 무작위 면에 정점을 넣고 세 꼭짓점과 잇는다.

    극대 평면·3-연결이고 차수 분포가 멱법칙을 따른다. 새 정점 좌표를 면 삼각형의 무게중심으로
    두면 그대로 직선 평면 임베딩이 된다.
    """
    if n < 3:
        raise ValueError(f"apollonian needs at least 3 vertices, got {n}")
    rng = np.random.default_rng(seed)
    points = np.zeros((n, 2))
    points[:3] = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2)]
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    faces: list[tuple[int, int, int]] = [(0, 1, 2)]
    for v in range(3, n):
        index = int(rng.integers(len(faces)))
        a, b, c = faces[index]
        points[v] = (points[a] + points[b] + points[c]) / 3
        graph.add_edges_from([(a, v), (b, v), (c, v)])
        faces[index] = (a, b, v)
        faces.append((b, c, v))
        faces.append((a, c, v))
    return graph, points


def generate_voronoi(n: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """[0,1)^2 균등 난수 점 k=(n-2)//2 개의 Voronoi 다이어그램을 단위 정사각형으로 자른 그래프.

    점을 네 변에 대해 거울상으로 복제해 Voronoi 를 구하면 원래 점의 셀이 모두 유한해지고 경계가
    정확히 정사각형으로 잘린다. 원래 점의 셀 경계를 이루는 ridge 만 간선으로 쓴다. 정점 수는
    2k+2 (모서리 4개만 차수 2, 나머지는 차수 3). Delaunay 의 쌍대라 면이 5~7각형으로 섞인다.
    """
    k = max(3, (n - 2) // 2)
    rng = np.random.default_rng(seed)
    points = rng.random((k, 2))
    mirrored = np.vstack(
        [
            points,
            points * [-1, 1],
            points * [1, -1],
            points * [-1, 1] + [2, 0],
            points * [1, -1] + [0, 2],
        ]
    )
    voronoi = Voronoi(mirrored)
    raw = nx.Graph()
    for (i, j), (a, b) in zip(
        voronoi.ridge_points, voronoi.ridge_vertices, strict=True
    ):
        if (i < k or j < k) and a != -1 and b != -1:
            raw.add_edge(int(a), int(b))
    graph, nodes = _relabel_sorted(raw)
    coords = np.clip(voronoi.vertices[nodes], 0.0, 1.0)
    return graph, coords


GENERATORS = {
    "delaunay": generate_delaunay,
    "grid": generate_grid,
    "hexagonal": generate_hexagonal,
    "apollonian": generate_apollonian,
    "voronoi": generate_voronoi,
}


def thin_biconnected(
    graph: nx.Graph, seed: int, remove_ratio: float
) -> tuple[nx.Graph, int]:
    """이중연결을 유지하며 간선을 최대 remove_ratio 비율만큼 제거한다. (결과, 제거 수)"""
    if not 0.0 <= remove_ratio < 1.0:
        raise ValueError(f"remove_ratio must be in [0.0, 1.0), got {remove_ratio}")
    thinned = nx.Graph(graph)
    target_remove = round(thinned.number_of_edges() * remove_ratio)
    if target_remove == 0:
        return thinned, 0

    rng = np.random.default_rng(seed)
    edges = [(min(u, v), max(u, v)) for u, v in thinned.edges()]
    edges.sort()
    order = rng.permutation(len(edges))

    removed = 0
    for index in order:
        if removed >= target_remove:
            break
        u, v = edges[index]
        thinned.remove_edge(u, v)
        if nx.is_biconnected(thinned):
            removed += 1
        else:
            thinned.add_edge(u, v)
    return thinned, removed


def build_record(
    vertices: int, seed: int, remove_ratio: float, family: str = DEFAULT_FAMILY
) -> GraphRecord:
    base, points = GENERATORS[family](vertices, seed)
    if not nx.is_biconnected(base):
        raise RuntimeError(
            f"{family} graph v={vertices} seed={seed} is not biconnected"
        )
    thinned, removed = thin_biconnected(base, seed, remove_ratio)
    if not nx.is_biconnected(thinned) or not nx.check_planarity(thinned)[0]:
        raise RuntimeError("thinned graph lost biconnectivity or planarity")

    edges = sorted((min(u, v), max(u, v)) for u, v in thinned.edges())
    return GraphRecord(
        graph_id=config.graph_id(vertices, seed, remove_ratio),
        vertices_requested=vertices,
        graph_seed=seed,
        remove_ratio_target=remove_ratio,
        remove_ratio_actual=removed / base.number_of_edges(),
        n_vertices=thinned.number_of_nodes(),
        n_edges=len(edges),
        n_edges_original=base.number_of_edges(),
        weight=1,
        pos=[(float(x), float(y)) for x, y in points],
        edges=edges,
        family=family,
    )


def save_graph(path: Path, record: GraphRecord) -> None:
    payload = {"schema_version": SCHEMA_VERSION, **asdict(record)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def load_graph(path: Path) -> GraphRecord:
    payload = json.loads(path.read_text())
    version = payload.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported graph schema version {version!r} in {path}")
    payload["pos"] = [(float(x), float(y)) for x, y in payload["pos"]]
    payload["edges"] = [(int(u), int(v)) for u, v in payload["edges"]]
    return GraphRecord(**payload)


def graph_path(graph_dir: Path, graph_id: str) -> Path:
    return graph_dir / f"{graph_id}.json"


def to_domain_graph(record: GraphRecord) -> Graph:
    """저장된 간선 순서대로 무방향 Edge 를 만들어 project Graph 를 구성한다."""
    return Graph(edges=[Edge(u, v, record.weight, False) for u, v in record.edges])


def to_networkx(record: GraphRecord) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(record.n_vertices))
    graph.add_edges_from(record.edges)
    return graph


def write_manifest(graph_dir: Path) -> Path:
    rows = []
    for path in sorted(graph_dir.glob("*.json")):
        record = load_graph(path)
        nx_graph = to_networkx(record)
        rows.append(
            {
                "graph_id": record.graph_id,
                "family": record.family,
                "vertices": record.vertices_requested,
                "graph_seed": record.graph_seed,
                "remove_ratio_target": record.remove_ratio_target,
                "remove_ratio_actual": f"{record.remove_ratio_actual:.4f}",
                "n_vertices": record.n_vertices,
                "n_edges": record.n_edges,
                "n_edges_original": record.n_edges_original,
                "biconnected": nx.is_biconnected(nx_graph),
                "planar": nx.check_planarity(nx_graph)[0],
            }
        )
    manifest = graph_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else ["graph_id"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="실험용 그래프 인스턴스 생성")
    parser.add_argument("--family", choices=list(GENERATORS), default=DEFAULT_FAMILY)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본값 data/graphs[_{family}]"
    )
    parser.add_argument(
        "--vertices", type=_parse_ints, default=list(config.VERTEX_COUNTS)
    )
    parser.add_argument("--seeds", type=_parse_ints, default=list(config.GRAPH_SEEDS))
    parser.add_argument(
        "--remove-ratios", type=_parse_floats, default=list(config.REMOVE_RATIOS)
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else default_graph_dir(args.family)

    for vertices in args.vertices:
        for seed in args.seeds:
            for ratio in args.remove_ratios:
                path = graph_path(out_dir, config.graph_id(vertices, seed, ratio))
                if path.exists() and not args.overwrite:
                    continue
                record = build_record(vertices, seed, ratio, args.family)
                save_graph(path, record)
                print(
                    f"{record.graph_id}: edges {record.n_edges}/{record.n_edges_original} "
                    f"(removed {record.remove_ratio_actual:.3f}, target {ratio})"
                )
    manifest = write_manifest(out_dir)
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
