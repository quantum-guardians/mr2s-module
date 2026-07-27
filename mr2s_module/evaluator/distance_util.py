from collections.abc import Iterable

import networkx as nx

from mr2s_module.domain import Solution
from mr2s_module.protocols import Graph


def _inverse_distance(weight: float, *, edge_id: int | None = None) -> float:
    if weight <= 0:
        ref = f"Edge {edge_id} " if edge_id is not None else ""
        raise ValueError(f"{ref}weight must be positive to invert, got {weight}")
    return 1.0 / weight


def _distance_digraph(
    distance_edges: Iterable[tuple[int, int, float]],
    vertices: Iterable[int] = (),
) -> nx.DiGraph:
    """(source, target, distance) 방향 간선을 거리 속성 DiGraph 로.

    방향화 그래프 build 의 단일 진실 — solution 경로와 SA 경로가 공유한다.
    같은 방향 평행 간선은 가장 빠른(작은 distance) 것만 최단거리에 유효.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(vertices)
    for u, v, distance in distance_edges:
        if graph.has_edge(u, v):
            distance = min(distance, graph[u][v]["distance"])
        graph.add_edge(u, v, distance=distance)
    return graph


def build_directed_distance_graph(solution: Solution) -> nx.DiGraph:
    """방향화 결과를 거리 = 1/weight 인 DiGraph 로 변환."""
    return _distance_digraph(
        (
            (
                u,
                v,
                _inverse_distance(
                    solution.graph.edges[edge_id].weight, edge_id=edge_id
                ),
            )
            for edge_id, (u, v) in solution.edges.items()
        ),
        solution.graph.get_vertices(),
    )


def build_directed_distance_graph_from_weighted_edges(
    weighted_edges: Iterable[tuple[int, int, float]],
    vertices: Iterable[int] = (),
) -> nx.DiGraph:
    """(source, target, weight) 방향 간선을 거리 = 1/weight DiGraph 로.

    SA objective 처럼 Solution 없이 raw 방향 간선만 있을 때 쓴다.
    """
    return _distance_digraph(
        ((u, v, _inverse_distance(weight)) for u, v, weight in weighted_edges),
        vertices,
    )


def stretch_totals(
    directed_lengths: dict[int, dict[int, float]],
    undirected_lengths: dict[int, dict[int, float]],
    vertices: Iterable[int],
) -> tuple[float, int, int]:
    """쌍별 stretch(방향화 거리 / 무방향 거리) 집계의 단일 진실.

    도달 가능한 순서쌍만 stretch 를 더하고, 도달 불가 쌍은 따로 센다.
    반환: ``(total_stretch, reachable_pairs, unreachable_pairs)``.
    ranker 는 평균(total/reachable)을, SA 는 total + unreachable 을 쓴다.
    """
    vertices = list(vertices)
    total_stretch = 0.0
    reachable_pairs = 0
    unreachable_pairs = 0
    for source in vertices:
        source_lengths = directed_lengths.get(source, {})
        for target in vertices:
            if source == target:
                continue
            directed_distance = source_lengths.get(target)
            if directed_distance is None:
                unreachable_pairs += 1
            else:
                total_stretch += directed_distance / undirected_lengths[source][target]
                reachable_pairs += 1
    return total_stretch, reachable_pairs, unreachable_pairs


def build_undirected_distance_graph(graph: Graph) -> nx.Graph:
    """방향화 전 원본 그래프를 거리 = 1/weight 인 무방향 nx.Graph 로 변환."""
    nx_graph = nx.Graph()
    for edge in graph.edges.values():
        u, v = edge.endpoints()
        distance = _inverse_distance(edge.weight, edge_id=edge.id)
        if nx_graph.has_edge(u, v):
            distance = min(distance, nx_graph[u][v]["distance"])
        nx_graph.add_edge(u, v, distance=distance)
    nx_graph.add_nodes_from(graph.get_vertices())
    return nx_graph


class UndirectedApspCache:
    """무방향 APSP(거리 = 1/weight) 결과를 Graph 인스턴스 단위로 캐시.

    같은 그래프의 solution 을 반복 평가할 때 분모(무방향 기준) 재계산을 없앤다.
    캐시 값에 graph 참조를 함께 들고 있어 id 재사용 충돌이 없다.
    """

    def __init__(self):
        self._cache: dict[int, tuple[Graph, dict[int, dict[int, float]]]] = {}

    def get_lengths(self, graph: Graph) -> dict[int, dict[int, float]]:
        key = id(graph)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is graph:
            return cached[1]

        nx_graph = build_undirected_distance_graph(graph)
        lengths = dict(nx.all_pairs_dijkstra_path_length(nx_graph, weight="distance"))
        self._cache[key] = (graph, lengths)
        return lengths
