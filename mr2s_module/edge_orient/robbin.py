from mr2s_module.domain.graph import Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.util import domain_graph_to_networkx, robbins_orient

import networkx as nx


class Robbin:
    """단순 dfs 순회하면서 부모관계 정립"""

    def run(self, graph: Graph) -> OrientedEdges:
        if graph.is_empty():
            return OrientedEdges()

        nx_graph = domain_graph_to_networkx(graph)

        # 브릿지 존재 시 강한 방향성 불가능 → 방향 결정 포기.
        if nx.has_bridges(nx_graph):
            return OrientedEdges()

        start_node = next(iter(graph.get_vertices()))
        directed_edges = robbins_orient(nx_graph, start_node)

        return OrientedEdges(edges=list(directed_edges.values()))
