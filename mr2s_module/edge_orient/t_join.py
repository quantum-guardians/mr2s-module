import itertools
import warnings

import networkx as nx


from mr2s_module.domain import Edge, Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.util import domain_graph_to_networkx


class Tjoin:
    """DEPRECATED. T-join 기반 부분 방향 배정.

    문제점:
    - Eulerian 부분그래프를 대칭차(G \\ J)로 만들어 J 간선을 '제거'한다.
      간선 집합 제거는 2-connected planar 그래프에서도 내부 정점을 고립시킬 수
      있어(경로 내부 차수-2 정점의 두 간선이 모두 J에 포함되는 경우) 전역
      강연결을 보장하지 못한다.
    - 고립된 정점을 남은 QUBO 자유변수로 복구할 수 있다는 보장도 없다
      (고정된 Euler-circuit 방향이 강방향으로 확장 가능하란 보장 없음 +
      QUBO 는 soft-penalty 휴리스틱).
    - `pair_to_edge` 가 쌍당 1 간선(단순그래프)을 전제하므로 멀티그래프의
      평행 간선을 잃는다.

    강연결이 필요하면 IteratedLocalSearch/Robbin(완전 방향) 을 쓰거나,
    T-join 을 되살릴 경우 제거 대신 복제(supergraph)로 재구현해야 한다.
    """

    def run(self, graph: Graph) -> OrientedEdges:
        warnings.warn(
            "Tjoin is deprecated: symmetric-difference removal can isolate "
            "vertices and does not guarantee strong connectivity, and it "
            "loses parallel edges on multigraphs. Use IteratedLocalSearch or "
            "Robbin for a full orientation.",
            DeprecationWarning,
            stacklevel=2,
        )
        if graph.is_empty():
            return OrientedEdges()

        nx_graph = domain_graph_to_networkx(graph)

        # 정체성(int id) 과 분리된 끝점 뷰. 단순그래프 전제(쌍당 1 간선).
        pair_to_edge = {e.pair_key(): e for e in graph.edges.values()}

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
        eulerian_edge_keys = set(pair_to_edge.keys()) ^ j_edges_keys

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
                orig_edge = pair_to_edge[frozenset({u, v})]
                oriented_edges.append(Edge(u, v, orig_edge.weight, True))

        return OrientedEdges(edges=oriented_edges)
