import itertools
import networkx as nx


from mr2s_module.domain import Graph, GraphPartitionResult
from mr2s_module.util import domain_graph_to_networkx, clone_edge


class TjoinCycle:
    def __init__(self):
        self.target_k = 1  # DnCMr2sSolver target_k binary search 호환용 (TjoinCycle은 사용 안 함)

    def run(self, graph: Graph) -> GraphPartitionResult:
        if graph.is_empty():
            return GraphPartitionResult()

        nx_graph = domain_graph_to_networkx(graph)

        # 1. Identify odd-degree nodes
        odd_nodes = [v for v, d in nx_graph.degree() if d % 2 != 0]

        # 2. Minimum Weight T-join
        j_edges_keys = set()
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
            path_edges_count = {}
            for u, v in matching:
                path = nx.shortest_path(nx_graph, u, v, weight="weight")
                for a, b in zip(path[:-1], path[1:]):
                    e = tuple(sorted((a, b)))
                    path_edges_count[e] = path_edges_count.get(e, 0) + 1

            # Symmetric difference J: edges that appear an odd number of times in the matching paths
            j_edges_keys = {e for e, count in path_edges_count.items() if count % 2 != 0}

        # 3. Eulerian subgraph G_E = G \Delta J
        edge_map = {edge.id: edge for edge in graph.edges}
        eulerian_edge_keys = set(edge_map.keys()) ^ j_edges_keys

        g_eulerian = nx.Graph()
        for e_key in eulerian_edge_keys:
            u, v = e_key
            g_eulerian.add_edge(u, v)

        # 4. Orient edges
        oriented_edges = []
        for component in nx.connected_components(g_eulerian):
            sub = g_eulerian.subgraph(component)
            if sub.number_of_edges() == 0:
                continue

            # Eulerian circuit (may contain multiple cycles if component is Eulerian)
            circuit = list(nx.eulerian_circuit(sub))
            for u, v in circuit:
                e_key = tuple(sorted((u, v)))
                orig_edge = edge_map[e_key]
                # Create directed edge
                new_edge = clone_edge(orig_edge)
                new_edge.directed = True
                new_edge.vertices = (u, v)
                oriented_edges.append(new_edge)

        # 5. Remaining edges (the T-join edges)
        remaining_edges = [edge_map[e_key] for e_key in j_edges_keys]

        return GraphPartitionResult(
            sub_graphs=[Graph(edges=oriented_edges)],
            remaining_edges=remaining_edges
        )
