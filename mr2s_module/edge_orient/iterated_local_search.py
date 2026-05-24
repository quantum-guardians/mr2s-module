from typing import Callable

import networkx as nx
import numpy as np

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.util import domain_graph_to_networkx, robbins_orient


EdgeMap = dict[frozenset[int], Edge]


class IteratedLocalSearch:
    def __init__(
        self,
        max_iter: int = 30,
        patience: int = 5,
        is_relaxed: bool = False,
        perturb_strength: int = 2,
    ):
        self.max_iter = max_iter
        self.patience = patience
        self.is_relaxed = is_relaxed
        self.perturb_strength = perturb_strength

        self.rng = np.random.default_rng()
        self._neighborhoods: list[Callable] = [n1_search, n2_search, n3_search]

    def run(self, graph: Graph) -> OrientedEdges:
        if graph.is_empty():
            return OrientedEdges()

        nx_graph = domain_graph_to_networkx(graph)
        nodes = list(nx_graph.nodes())

        best_edges: EdgeMap | None = None
        best_score = float('inf')
        stale_count = 0

        for iteration, (cand_edges, cand_score) in enumerate(
            self._search_generator(nx_graph, nodes)
        ):
            if best_edges is None or cand_score < best_score:
                best_edges = cand_edges
                best_score = cand_score
                stale_count = 0
            else:
                stale_count += 1

            if stale_count >= self.patience or iteration == self.max_iter - 1:
                break

        return OrientedEdges(edges=list(best_edges.values()) if best_edges else [])

    def _search_generator(self, base_graph: nx.Graph, nodes: list):
        """generator yield를 사용하여 외부(run)에서 Early Stopping 하도록 설정."""
        cur_edges = self._generate_initial_edges(base_graph)
        cur_edges, cur_score = self._vnd_local_search(cur_edges, base_graph, nodes)
        yield cur_edges, cur_score

        while True:
            cand_edges = self._perturb(cur_edges, nodes, level=self.perturb_strength)
            cand_edges, cand_score = self._vnd_local_search(cand_edges, base_graph, nodes)

            if self.is_relaxed or cand_score < cur_score:  # RILS 일 시 무조건 수정.
                cur_edges = cand_edges
                cur_score = cand_score

            yield cand_edges, cand_score

    def _vnd_local_search(
        self,
        start_edges: EdgeMap,
        base_graph: nx.Graph,
        nodes: list,
    ) -> tuple[EdgeMap, float]:
        cur_edges = start_edges
        cur_score = evaluate_score(cur_edges, nodes)

        k = 0
        while k < len(self._neighborhoods):
            cand_edges, cand_score = self._neighborhoods[k](
                cur_edges, cur_score, base_graph, nodes, self.rng
            )
            if cand_score < cur_score:
                cur_edges = cand_edges
                cur_score = cand_score
                k = 0
            else:
                k += 1

        return cur_edges, cur_score

    def _perturb(
        self,
        edges: EdgeMap,
        nodes: list,
        level: int = 1,
    ) -> EdgeMap:
        """지역 최적해에 빠져있을 경우를 위해 흔들어서 탈출하도록 도와주는 함수."""
        new_edges = dict(edges)
        edge_keys = list(new_edges.keys())
        n_edges = len(edge_keys)
        if n_edges == 0:
            return new_edges

        attempts = 0
        flips_done = 0
        max_attempts = n_edges * 2

        while flips_done < level and attempts < max_attempts:
            ek = edge_keys[int(self.rng.integers(0, n_edges))]
            original = new_edges[ek]
            new_edges[ek] = original.flip()
            if evaluate_score(new_edges, nodes) == float('inf'):
                new_edges[ek] = original
            else:
                flips_done += 1
            attempts += 1

        return new_edges

    def _generate_initial_edges(self, base_graph: nx.Graph) -> EdgeMap:
        """초기 강연결 그래프를 생성합니다."""
        nodes = list(base_graph.nodes())
        start_node = nodes[int(self.rng.integers(0, len(nodes)))]
        return robbins_orient(base_graph, start_node)


def evaluate_score(
    edges: EdgeMap,
    nodes: list,
) -> float:
    if not nodes:
        return float('inf')

    D = nx.DiGraph()
    D.add_nodes_from(nodes)
    for edge in edges.values():
        u, v = edge.vertices
        D.add_edge(u, v, weight=edge.weight)

    if not nx.is_strongly_connected(D):
        return float('inf')

    lengths = dict(nx.all_pairs_dijkstra_path_length(D, weight="weight"))

    total_distance = 0
    for targets in lengths.values():
        total_distance += sum(targets.values())
    return float(total_distance)


def n1_search(
    current_edges: EdgeMap,
    current_score: float,
    base_graph: nx.Graph,
    nodes: list,
    rng: np.random.Generator,
) -> tuple[EdgeMap, float]:
    working = dict(current_edges)
    edge_keys = list(working.keys())
    rng.shuffle(edge_keys)

    for ek in edge_keys:
        original = working[ek]
        working[ek] = original.flip()
        cand_score = evaluate_score(working, nodes)
        if cand_score < current_score:
            return working, cand_score
        working[ek] = original

    return current_edges, current_score


def n2_search(
    current_edges: EdgeMap,
    current_score: float,
    base_graph: nx.Graph,
    nodes: list,
    rng: np.random.Generator,
) -> tuple[EdgeMap, float]:
    working = dict(current_edges)
    shuffled_nodes = list(nodes)
    rng.shuffle(shuffled_nodes)

    for node in shuffled_nodes:
        incident = [
            frozenset({node, nb}) for nb in base_graph.neighbors(node)
        ]
        if not incident:
            continue

        originals = {ek: working[ek] for ek in incident}
        for ek in incident:
            working[ek] = working[ek].flip()
        cand_score = evaluate_score(working, nodes)
        if cand_score < current_score:
            return working, cand_score
        for ek, orig in originals.items():
            working[ek] = orig

    return current_edges, current_score


def n3_search(
    current_edges: EdgeMap,
    current_score: float,
    base_graph: nx.Graph,
    nodes: list,
    rng: np.random.Generator,
) -> tuple[EdgeMap, float]:
    D = nx.DiGraph()
    D.add_nodes_from(nodes)
    for edge in current_edges.values():
        D.add_edge(*edge.vertices)

    working = dict(current_edges)
    sources = list(nodes)
    rng.shuffle(sources)

    for node in sources:
        try:
            cycle_edges = nx.find_cycle(D, source=node)
        except nx.NetworkXNoCycle:
            continue

        cycle_keys = [frozenset({u, v}) for u, v in cycle_edges]
        originals = {ek: working[ek] for ek in cycle_keys}
        for ek in cycle_keys:
            working[ek] = working[ek].flip()
        cand_score = evaluate_score(working, nodes)
        if cand_score < current_score:
            return working, cand_score
        for ek, orig in originals.items():
            working[ek] = orig

    return current_edges, current_score
