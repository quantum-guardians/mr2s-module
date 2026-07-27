import networkx as nx

from mr2s_module.domain.graph import Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.util import domain_graph_to_networkx_multi, robbins_orient


class Robbin:
    """단순 dfs 순회하면서 부모관계 정립"""

    def run(self, graph: Graph) -> OrientedEdges:
        if graph.is_empty():
            return OrientedEdges()

        nx_graph = domain_graph_to_networkx_multi(graph)

        # 브릿지 존재 시 강한 방향성 불가능 → 방향 결정 포기.
        # 평행 copy 가 있는 쌍은 브릿지가 아니므로 simple 뷰 브릿지 중 다중도 1 만 진짜다.
        simple_view = nx.Graph(nx_graph)
        if any(nx_graph.number_of_edges(u, v) == 1 for u, v in nx.bridges(simple_view)):
            return OrientedEdges()

        start_node = next(iter(graph.get_vertices()))
        directed_edges = robbins_orient(nx_graph, start_node)

        return OrientedEdges(edges=list(directed_edges.values()))
