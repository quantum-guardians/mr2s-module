from typing import Callable

import networkx as nx
import numpy as np

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.orientation_result import OrientedEdges
from mr2s_module.edge_orient.orientation import EdgeKey, Orientation, robbins_orient
from mr2s_module.util import domain_graph_to_networkx


class IteratedLocalSearch:
    def __init__(
        self,
        max_iter: int = 30,
        patience: int = 5,
        is_relaxed: bool = False,
        perturb_strength: int = 2,
    ):
        self._weight_map: dict[EdgeKey, int] = {}
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
        self._weight_map = {frozenset(edge.id): edge.weight for edge in graph.edges}
        nodes = list(nx_graph.nodes())

        best_orientation: Orientation | None = None
        best_score = float('inf')
        stale_count = 0

        for iteration, (cand_orientation, cand_score) in enumerate(
            self._search_generator(nx_graph, nodes)
        ):
            if best_orientation is None or cand_score < best_score:
                best_orientation = cand_orientation
                best_score = cand_score
                stale_count = 0
            else:
                stale_count += 1

            if stale_count >= self.patience or iteration == self.max_iter - 1:
                break

        return self._build_oriented_edges(best_orientation)

    def _search_generator(self, base_graph: nx.Graph, nodes: list):
        """generator yield를 사용하여 외부(run)에서 Early Stopping 하도록 설정."""
        cur_orientation = self._generate_initial_orientation(base_graph)
        cur_orientation, cur_score = self._vnd_local_search(
            cur_orientation, base_graph, nodes
        )
        yield cur_orientation, cur_score

        while True:
            cand_orientation = self._perturb(
                cur_orientation, nodes, level=self.perturb_strength
            )
            cand_orientation, cand_score = self._vnd_local_search(
                cand_orientation, base_graph, nodes
            )

            if self.is_relaxed or cand_score < cur_score:  # RILS 일 시 무조건 수정.
                cur_orientation = cand_orientation
                cur_score = cand_score

            yield cand_orientation, cand_score

    def _vnd_local_search(
        self,
        start_orientation: Orientation,
        base_graph: nx.Graph,
        nodes: list,
    ) -> tuple[Orientation, float]:
        cur_orientation = start_orientation
        cur_score = evaluate_score(cur_orientation, nodes, self._weight_map)

        k = 0
        while k < len(self._neighborhoods):
            cand_orientation, cand_score = self._neighborhoods[k](
                cur_orientation, cur_score, base_graph, nodes, self._weight_map, self.rng
            )
            if cand_score < cur_score:
                cur_orientation = cand_orientation
                cur_score = cand_score
                k = 0
            else:
                k += 1

        return cur_orientation, cur_score

    def _perturb(
        self,
        orientation: Orientation,
        nodes: list,
        level: int = 1,
    ) -> Orientation:
        """지역 최적해에 빠져있을 경우를 위해 흔들어서 탈출하도록 도와주는 함수."""
        new_orientation = orientation.copy()
        edge_keys = new_orientation.edge_keys()
        n_edges = len(edge_keys)
        if n_edges == 0:
            return new_orientation

        attempts = 0
        flips_done = 0
        max_attempts = n_edges * 2

        while flips_done < level and attempts < max_attempts:
            ek = edge_keys[int(self.rng.integers(0, n_edges))]
            new_orientation.flip(ek)
            if evaluate_score(new_orientation, nodes, self._weight_map) == float('inf'):
                new_orientation.flip(ek)
            else:
                flips_done += 1
            attempts += 1

        return new_orientation

    def _generate_initial_orientation(self, base_graph: nx.Graph) -> Orientation:
        """초기 강연결 그래프를 생성합니다."""
        nodes = list(base_graph.nodes())
        start_node = nodes[int(self.rng.integers(0, len(nodes)))]
        return robbins_orient(base_graph, start_node)

    def _build_oriented_edges(self, orientation: Orientation) -> OrientedEdges:
        edges = []
        for ek, (u, v) in orientation.items():
            weight = self._weight_map.get(ek, 1)
            edges.append(Edge(u, v, weight=weight, directed=True))
        return OrientedEdges(edges=edges)


def evaluate_score(
    orientation: Orientation,
    nodes: list,
    weight_map: dict[EdgeKey, int] | None = None,
) -> float:
    if not nodes:
        return float('inf')

    D = nx.DiGraph()
    D.add_nodes_from(nodes)
    if weight_map is None:
        D.add_edges_from(orientation.directed_edges())
    else:
        for ek, (u, v) in orientation.items():
            D.add_edge(u, v, weight=weight_map.get(ek, 1))

    if not nx.is_strongly_connected(D):
        return float('inf')

    if weight_map is None:
        lengths = dict(nx.all_pairs_shortest_path_length(D))
    else:
        lengths = dict(nx.all_pairs_dijkstra_path_length(D, weight="weight"))

    total_distance = 0
    for targets in lengths.values():
        total_distance += sum(targets.values())
    return float(total_distance)


def n1_search(
    current_orientation: Orientation,
    current_score: float,
    base_graph: nx.Graph,
    nodes: list,
    weight_map: dict[EdgeKey, int] | None,
    rng: np.random.Generator,
) -> tuple[Orientation, float]:
    working = current_orientation.copy()
    edge_keys = working.edge_keys()
    rng.shuffle(edge_keys)

    for ek in edge_keys:
        working.flip(ek)
        cand_score = evaluate_score(working, nodes, weight_map)
        if cand_score < current_score:
            return working, cand_score
        working.flip(ek)

    return current_orientation, current_score


def n2_search(
    current_orientation: Orientation,
    current_score: float,
    base_graph: nx.Graph,
    nodes: list,
    weight_map: dict[EdgeKey, int] | None,
    rng: np.random.Generator,
) -> tuple[Orientation, float]:
    working = current_orientation.copy()
    shuffled_nodes = list(nodes)
    rng.shuffle(shuffled_nodes)

    for node in shuffled_nodes:
        incident = [
            Orientation.edge_key(node, nb) for nb in base_graph.neighbors(node)
        ]
        if not incident:
            continue

        for ek in incident:
            working.flip(ek)
        cand_score = evaluate_score(working, nodes, weight_map)
        if cand_score < current_score:
            return working, cand_score
        for ek in incident:
            working.flip(ek)

    return current_orientation, current_score


def n3_search(
    current_orientation: Orientation,
    current_score: float,
    base_graph: nx.Graph,
    nodes: list,
    weight_map: dict[EdgeKey, int] | None,
    rng: np.random.Generator,
) -> tuple[Orientation, float]:
    D = nx.DiGraph()
    D.add_nodes_from(nodes)
    D.add_edges_from(current_orientation.directed_edges())

    working = current_orientation.copy()
    sources = list(nodes)
    rng.shuffle(sources)

    for node in sources:
        try:
            cycle_edges = nx.find_cycle(D, source=node)
        except nx.NetworkXNoCycle:
            continue

        cycle_keys = [Orientation.edge_key(u, v) for u, v in cycle_edges]
        for ek in cycle_keys:
            working.flip(ek)
        cand_score = evaluate_score(working, nodes, weight_map)
        if cand_score < current_score:
            return working, cand_score
        for ek in cycle_keys:
            working.flip(ek)

    return current_orientation, current_score
