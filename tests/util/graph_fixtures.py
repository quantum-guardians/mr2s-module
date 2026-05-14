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
