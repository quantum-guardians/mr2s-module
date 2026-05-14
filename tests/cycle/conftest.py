import random

import networkx as nx

from mr2s_module.domain import Edge, Graph


def remove_edges_by_percent(
    graph: Graph,
    remove_percent: int,
    seed: int = 7,
    weight: int = 1,
) -> tuple[Graph, int]:
    """Biconnectivity 를 유지하면서 가능한 만큼 간선을 제거한다."""
    if not 0 <= remove_percent < 100:
        raise ValueError("remove_percent must be in [0, 100).")

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge.id for edge in graph.edges)

    edge_count = nx_graph.number_of_edges()
    target_remove_count = round(edge_count * remove_percent / 100)
    shuffled_edges = list(nx_graph.edges())
    random.Random(seed).shuffle(shuffled_edges)

    removed_count = 0
    for u, v in shuffled_edges:
        if removed_count >= target_remove_count:
            break

        nx_graph.remove_edge(u, v)
        if nx.is_biconnected(nx_graph):
            removed_count += 1
        else:
            nx_graph.add_edge(u, v)

    thinned_graph = Graph(edges=[
        Edge(min(u, v), max(u, v), weight, False)
        for u, v in nx_graph.edges()
    ])
    return thinned_graph, removed_count
