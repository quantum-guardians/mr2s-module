import itertools

import networkx as nx
import numpy as np

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph


_OUTER_WALL_WEIGHT = 999_999
_INNER_WEIGHT = 1


class FaceCycle:
    def __init__(self, target_k: int = 10):
        self.target_k = target_k

    def run(self, graph: Graph) -> set[Edge]:
        nx_graph = self._to_networkx(graph)
        is_planar, _ = nx.check_planarity(nx_graph)
        if not is_planar:
            return set()

        weight_map = {
            edge.id: edge.weight
            for edge in graph.edges
            if edge.id[0] != edge.id[1]
        }

        result_edges: set[Edge] = set()
        for component in self._extract_biconnected_components(nx_graph):
            for u, v in self._process_component(component):
                key = (min(u, v), max(u, v))
                if key not in weight_map:
                    continue
                result_edges.add(Edge(u, v, weight_map[key], True))

        return result_edges

    def _extract_biconnected_components(self, graph: nx.Graph) -> list[nx.Graph]:
        if nx.is_biconnected(graph):
            return [graph]

        components: list[nx.Graph] = []
        for bcc_edges in nx.biconnected_component_edges(graph):
            if len(bcc_edges) < 3:
                continue
            sub = nx.Graph()
            for u, v in bcc_edges:
                sub.add_edge(u, v, **graph[u][v])
            if nx.check_planarity(sub)[0]:
                components.append(sub)
        return components

    def _process_component(self, component: nx.Graph) -> set[tuple[int, int]]:
        """컴포넌트 단위 boundary 사이클을 방향이 부여된 (u, v) 쌍 집합으로 반환."""
        if component.number_of_edges() < 3:
            return set()

        # 1. 면 추출 — 외곽 면은 가장 큰 면적으로 식별
        pos = nx.planar_layout(component)
        all_raw_faces = self._enumerate_faces(component)
        if len(all_raw_faces) < 2:
            return set()

        outer_idx = int(np.argmax([self._face_area(f, pos) for f in all_raw_faces]))
        inner_raw_faces = [
            f for i, f in enumerate(all_raw_faces) if i != outer_idx
        ]
        if not inner_raw_faces:
            return set()

        face_edges_map = self._build_face_edges_map(inner_raw_faces)
        face_centroids = [
            np.mean([pos[v] for v in f], axis=0) for f in inner_raw_faces
        ]
        dual_base = self._build_dual_base(face_edges_map)

        target_k = max(1, min(self.target_k, len(inner_raw_faces)))
        face_to_cluster = self._snowball_cluster(
            face_centroids, dual_base, target_k
        )

        # 2. 외벽 보호 2차 T-join 수리
        boundary_edges, outer_edges = self._collect_boundary_edges(
            face_edges_map, face_to_cluster
        )
        repair_edges = self._wall_protected_repair(
            component, boundary_edges, outer_edges
        )
        final_boundary = boundary_edges.symmetric_difference(repair_edges)

        # 4. Flood fill — 같은 군집의 면들 병합
        face_graph = nx.Graph()
        face_graph.add_nodes_from(range(len(inner_raw_faces)))
        for e, f_indices in face_edges_map.items():
            if len(f_indices) == 2 and e not in final_boundary:
                face_graph.add_edge(f_indices[0], f_indices[1])

        # 5. 유령 영토 필터링 — 외곽으로 누출된 컴포넌트 제거
        true_components = self._filter_ghost_components(
            face_graph, inner_raw_faces, outer_edges, final_boundary
        )

        # 6. 병합 쌍대 그래프 — 이분(bipartite) 성질 검증
        merged_dual = self._build_merged_dual(
            true_components, inner_raw_faces, face_edges_map, final_boundary
        )
        if merged_dual.number_of_nodes() > 0 and not nx.is_bipartite(merged_dual):
            return set()

        # 7. 방향 부여 — 2-coloring 으로 face traversal × CW/CCW 결정
        coloring = (
            nx.bipartite.color(merged_dual)
            if merged_dual.number_of_nodes() > 0
            else {}
        )
        face_to_color: dict[int, int] = {}
        for c_idx, comp in enumerate(true_components):
            c = coloring.get(c_idx, 0) % 2
            for f_idx in comp:
                face_to_color[f_idx] = c

        return self._orient_boundary(
            final_boundary, face_edges_map, inner_raw_faces, face_to_color
        )

    @staticmethod
    def _orient_boundary(
        final_boundary: set[tuple[int, int]],
        face_edges_map: dict[tuple[int, int], list[int]],
        inner_raw_faces: list[list[int]],
        face_to_color: dict[int, int],
    ) -> set[tuple[int, int]]:
        # 각 boundary 간선에 대해, 인접 면 중 true-component 에 속한 면을 우선 선택해
        # 그 면의 traversal 순서를 기준으로 방향을 결정 (color 1 이면 reverse).
        # 두 면이 서로 다른 컴포넌트에 속할 경우, 둘 모두 같은 방향을 만들어 낸다.
        directed_pairs: set[tuple[int, int]] = set()
        for e in final_boundary:
            f_indices = face_edges_map.get(e, [])
            if not f_indices:
                continue

            chosen_face = next(
                (f for f in f_indices if f in face_to_color),
                f_indices[0],
            )
            color = face_to_color.get(chosen_face, 0)
            face = inner_raw_faces[chosen_face]
            traversal = face if color == 0 else list(reversed(face))
            for i in range(len(traversal)):
                a, b = traversal[i], traversal[(i + 1) % len(traversal)]
                if tuple(sorted((a, b))) == e:
                    directed_pairs.add((a, b))
                    break
        return directed_pairs

    @staticmethod
    def _to_networkx(graph: Graph) -> nx.Graph:
        # 동일 정점 쌍이 여러 번 나오면 최소 가중치만 남기고, self-loop은 무시.
        g = nx.Graph()
        for edge in graph.edges:
            u, v = edge.id
            if u == v:
                continue
            if g.has_edge(u, v):
                if edge.weight < g[u][v]["weight"]:
                    g[u][v]["weight"] = edge.weight
            else:
                g.add_edge(u, v, weight=edge.weight)
        return g

    @staticmethod
    def _enumerate_faces(graph: nx.Graph) -> list[list[int]]:
        is_planar, embedding = nx.check_planarity(graph)
        if not is_planar:
            return []

        faces: list[list[int]] = []
        visited: set[tuple[int, int]] = set()
        for u, v in graph.edges():
            for he in ((u, v), (v, u)):
                if he in visited:
                    continue
                face: list[int] = []
                cu, cv = he
                while (cu, cv) not in visited:
                    visited.add((cu, cv))
                    face.append(cu)
                    cu, cv = cv, embedding.next_face_half_edge(cu, cv)[1]
                faces.append(face)
        return faces

    @staticmethod
    def _face_area(face: list[int], pos: dict) -> float:
        coords = [pos[v] for v in face]
        a = 0.0
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            a += x1 * y2 - x2 * y1
        return abs(a) / 2.0

    @staticmethod
    def _build_face_edges_map(
        faces: list[list[int]],
    ) -> dict[tuple[int, int], list[int]]:
        face_edges_map: dict[tuple[int, int], list[int]] = {}
        for f_idx, face in enumerate(faces):
            for i in range(len(face)):
                e = tuple(sorted((face[i], face[(i + 1) % len(face)])))
                face_edges_map.setdefault(e, []).append(f_idx)
        return face_edges_map

    @staticmethod
    def _build_dual_base(
        face_edges_map: dict[tuple[int, int], list[int]],
    ) -> nx.Graph:
        dual = nx.Graph()
        for f_indices in face_edges_map.values():
            if len(f_indices) == 2:
                dual.add_edge(f_indices[0], f_indices[1])
        return dual

    @staticmethod
    def _snowball_cluster(
        centroids: list[np.ndarray],
        dual_base: nx.Graph,
        target_k: int,
    ) -> dict[int, int]:
        n = len(centroids)
        seeds = [int(np.random.randint(n))]
        while len(seeds) < target_k:
            best_idx, max_d = -1, -1.0
            for f_idx in range(n):
                if f_idx in seeds:
                    continue
                min_d = min(
                    float(np.linalg.norm(centroids[f_idx] - centroids[s]))
                    for s in seeds
                )
                if min_d > max_d:
                    max_d, best_idx = min_d, f_idx
            if best_idx == -1:
                break
            seeds.append(best_idx)

        face_to_cluster: dict[int, int] = {s: i for i, s in enumerate(seeds)}
        frontiers: dict[int, list[int]] = {i: [s] for i, s in enumerate(seeds)}
        while len(face_to_cluster) < n:
            progressed = False
            for c_id in range(len(seeds)):
                next_frontier: list[int] = []
                for node in frontiers[c_id]:
                    if node not in dual_base:
                        continue
                    for nbr in dual_base.neighbors(node):
                        if nbr in face_to_cluster:
                            continue
                        face_to_cluster[nbr] = c_id
                        next_frontier.append(nbr)
                        progressed = True
                frontiers[c_id] = next_frontier
            if not progressed:
                break
        return face_to_cluster

    @staticmethod
    def _collect_boundary_edges(
        face_edges_map: dict[tuple[int, int], list[int]],
        face_to_cluster: dict[int, int],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        boundary_edges: set[tuple[int, int]] = set()
        outer_edges: set[tuple[int, int]] = set()
        for e, f_indices in face_edges_map.items():
            if len(f_indices) == 2:
                if face_to_cluster.get(f_indices[0]) != face_to_cluster.get(f_indices[1]):
                    boundary_edges.add(e)
            else:
                # 외곽: 한쪽에만 면이 붙어있는 간선
                boundary_edges.add(e)
                outer_edges.add(e)
        return boundary_edges, outer_edges

    @staticmethod
    def _wall_protected_repair(
        g_euler: nx.Graph,
        boundary_edges: set[tuple[int, int]],
        outer_edges: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        b_sub = nx.Graph()
        b_sub.add_edges_from(boundary_edges)
        odd_nodes = [v for v, d in b_sub.degree() if d % 2 != 0]
        if not odd_nodes:
            return set()

        # 외벽 파괴 절대 금지: 외곽 간선은 매우 큰 가중치를 부여해 우회시키고
        # 내륙 간선만 우선 이용하여 수리 경로를 잡는다.
        g_repair = g_euler.copy()
        for u, v in g_repair.edges():
            e = tuple(sorted((u, v)))
            g_repair[u][v]["weight"] = (
                _OUTER_WALL_WEIGHT if e in outer_edges else _INNER_WEIGHT
            )

        dist_map = dict(nx.all_pairs_dijkstra_path_length(g_repair, weight="weight"))
        complete = nx.Graph()
        for u, v in itertools.combinations(odd_nodes, 2):
            if v in dist_map.get(u, {}):
                complete.add_edge(u, v, weight=dist_map[u][v])

        repair_edges: set[tuple[int, int]] = set()
        for u, v in nx.min_weight_matching(complete):
            path = nx.shortest_path(g_repair, u, v, weight="weight")
            for a, b in zip(path[:-1], path[1:]):
                repair_edges.add(tuple(sorted((a, b))))
        return repair_edges

    @staticmethod
    def _filter_ghost_components(
        face_graph: nx.Graph,
        inner_raw_faces: list[list[int]],
        outer_edges: set[tuple[int, int]],
        final_boundary: set[tuple[int, int]],
    ) -> list[list[int]]:
        true_components: list[list[int]] = []
        for comp in nx.connected_components(face_graph):
            leaked = False
            for f_idx in comp:
                face = inner_raw_faces[f_idx]
                for i in range(len(face)):
                    fe = tuple(sorted((face[i], face[(i + 1) % len(face)])))
                    if fe in outer_edges and fe not in final_boundary:
                        leaked = True
                        break
                if leaked:
                    break
            if not leaked:
                true_components.append(list(comp))
        return true_components

    @staticmethod
    def _build_merged_dual(
        components: list[list[int]],
        inner_raw_faces: list[list[int]],
        face_edges_map: dict[tuple[int, int], list[int]],
        final_boundary: set[tuple[int, int]],
    ) -> nx.Graph:
        merged = nx.Graph()
        merged.add_nodes_from(range(len(components)))
        for i, j in itertools.combinations(range(len(components)), 2):
            comp_j = set(components[j])
            adjacent = False
            for f_idx in components[i]:
                face = inner_raw_faces[f_idx]
                for k in range(len(face)):
                    fe = tuple(sorted((face[k], face[(k + 1) % len(face)])))
                    if fe not in final_boundary:
                        continue
                    adj_faces = face_edges_map.get(fe, [])
                    if len(adj_faces) != 2:
                        continue
                    other = adj_faces[0] if adj_faces[1] == f_idx else adj_faces[1]
                    if other in comp_j:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                merged.add_edge(i, j)
        return merged
