import itertools
import networkx as nx


from mr2s_module.domain import Edge, Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.util import domain_graph_to_networkx


class Tjoin:
    def run(self, graph: Graph) -> OrientedEdges:
        if graph.is_empty():
            return OrientedEdges()

        nx_graph = domain_graph_to_networkx(graph)

        # 1. Identify odd-degree nodes
        odd_nodes = [v for v, d in nx_graph.degree() if d % 2 != 0]

        # 2. Minimum Weight T-join
        j_edges_keys: set[frozenset[int]] = set()
        if odd_nodes:
            # All-pairs shortest paths
            dist_map = dict(nx.all_pairs_dijkstra_path_length(nx_graph, weight="weight"))

            # Complete graph of odd nodes
            complete = nx.Graph()
            for u, v in itertools.combinations(odd_nodes, 2):
                if v in dist_map.get(u, {}):
                    complete.add_edge(u, v, weight=dist_map[u][v])

            # Min weight matching
            matching = nx.min_weight_matching(complete, weight="weight")

            # Edges in the paths
            path_edges_count: dict[frozenset[int], int] = {}
            for u, v in matching:
                path = nx.shortest_path(nx_graph, u, v, weight="weight")
                for a, b in zip(path[:-1], path[1:]):
                    e = frozenset({a, b})
                    path_edges_count[e] = path_edges_count.get(e, 0) + 1

            # Symmetric difference J: edges that appear an odd number of times in the matching paths
            j_edges_keys = {e for e, count in path_edges_count.items() if count % 2 != 0}

        # 3. Eulerian subgraph G_E = G \Delta J
        eulerian_edge_keys = set(graph.edges.keys()) ^ j_edges_keys

        g_eulerian = nx.Graph()
        for e_key in eulerian_edge_keys:
            u, v = sorted(e_key)
            g_eulerian.add_edge(u, v)

        # 4. Orient edges
        oriented_edges: list[Edge] = []
        for component in nx.connected_components(g_eulerian):
            sub = g_eulerian.subgraph(component)
            if sub.number_of_edges() == 0:
                continue

            circuit = list(nx.eulerian_circuit(sub))
            for u, v in circuit:
                orig_edge = graph.edges[frozenset({u, v})]
                oriented_edges.append(Edge(u, v, orig_edge.weight, True))

        return OrientedEdges(edges=oriented_edges)
