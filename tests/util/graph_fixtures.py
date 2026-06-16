from __future__ import annotations

import itertools
import random

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from mr2s_module.domain import Edge, Graph
from mr2s_module.util import domain_graph_to_networkx, networkx_to_domain_graph


def graph_from_pairs(
    pairs: list[tuple[int, int]] | set[tuple[int, int]],
    *,
    weight: int = 1,
    directed: bool = False,
) -> Graph:
    """Build a project Graph from vertex pairs with one shared edge policy."""
    return Graph(edges=[
        Edge(u, v, weight, directed)
        for u, v in pairs
    ])


def delaunay_graph(n: int, seed: int, *, weight: int = 1) -> Graph:
    """Build a deterministic project Graph from a 2D Delaunay triangulation."""
    graph, _ = delaunay_graph_with_pos(n, seed, weight=weight)
    return graph


def delaunay_graph_with_pos(
    n: int,
    seed: int,
    *,
    weight: int = 1,
) -> tuple[Graph, dict[int, np.ndarray]]:
    """Build a Delaunay graph and return the original point positions."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    triangulation = Delaunay(points)

    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for simplex in triangulation.simplices:
        for u, v in itertools.combinations(simplex, 2):
            u, v = int(u), int(v)
            if u == v:
                continue
            edge_id = (min(u, v), max(u, v))
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edges.append(Edge(edge_id[0], edge_id[1], weight, False))

    pos = {index: np.array(points[index]) for index in range(n)}
    return Graph(edges=edges), pos


def random_planar_nx_graph(node_count: int, seed: int) -> nx.Graph:
    """Grow a random planar NetworkX graph from a labeled tree."""
    rng = random.Random(seed)
    graph = nx.random_labeled_tree(node_count, seed=seed)
    candidates = [
        (u, v)
        for u in graph.nodes
        for v in graph.nodes
        if u < v and not graph.has_edge(u, v)
    ]
    rng.shuffle(candidates)

    for u, v in candidates:
        graph.add_edge(u, v)
        if not nx.check_planarity(graph)[0]:
            graph.remove_edge(u, v)
    return graph


def nx_graph_to_domain_graph(graph: nx.Graph, *, weight: int = 1) -> Graph:
    """Convert a NetworkX graph fixture to the project Graph model."""
    return networkx_to_domain_graph(graph, weight=weight)


def domain_graph_to_nx_graph(graph: Graph) -> nx.Graph:
    """Convert a project Graph fixture to an unweighted NetworkX graph."""
    return domain_graph_to_networkx(graph)


def directed_tuples(edges: dict[int, Edge]) -> set[tuple[int, int]]:
    """Solution.edges (dict[edge_id, directed Edge]) → directed (source, target) 튜플 집합.

    평행간선을 구분하지 않는 단순그래프 단언용 헬퍼. 멀티그래프에서는 평행
    same-direction 간선이 한 튜플로 합쳐지므로 개수 검증엔 len(edges) 를 쓸 것.
    """
    return {edge.vertices for edge in edges.values()}


def directed_edges_dict(
    tuples: list[tuple[int, int]] | set[tuple[int, int]],
    *,
    weight: int = 1,
) -> dict[int, Edge]:
    """directed (source, target) 튜플들 → Solution.edges 형태 dict[id, directed Edge].

    테스트 스텁이 Solution 을 직접 만들 때 사용. 키는 합성 인덱스(merge 는 양끝점
    기준으로 매칭하므로 그래프 id 와 일치할 필요 없음)."""
    return {
        index: Edge(source, target, weight, True)
        for index, (source, target) in enumerate(tuples)
    }
