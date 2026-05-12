from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph
from mr2s_module.domain.graph_partition_result import GraphPartitionResult


EdgeKey = tuple[int, int]


@dataclass(frozen=True)
class Face:
    vertices: tuple[int, ...]
    edges: frozenset[EdgeKey]
    weight: int


class RegionPartition(list[set[int]]):
    def __init__(
        self,
        regions: Iterable[set[int]],
        boundary_edges_by_region: list[set[EdgeKey]],
        face_clusters: list[set[int]],
    ) -> None:
        super().__init__(regions)
        self.boundary_edges_by_region = boundary_edges_by_region
        self.face_clusters = face_clusters


@dataclass
class PlanarRegionFaceCycle:
    target_k: int = 2

    def run(self, graph: Graph) -> GraphPartitionResult:
        if any(edge.directed for edge in graph.edges):
            raise ValueError("PlanarRegionFaceCycle requires an undirected input graph")

        nx_graph = _domain_graph_to_networkx(graph)
        is_planar, _ = nx.check_planarity(nx_graph)
        if not is_planar:
            return GraphPartitionResult(
                sub_graphs=[],
                remaining_edges=list(graph.edges),
            )

        faces = extract_faces(nx_graph)
        if not faces:
            return GraphPartitionResult(
                sub_graphs=[],
                remaining_edges=list(graph.edges),
            )

        face_graph = build_face_graph(faces)
        if not nx.is_connected(face_graph):
            return GraphPartitionResult(
                sub_graphs=[],
                remaining_edges=list(graph.edges),
            )

        target_k = max(1, min(self.target_k, len(faces)))
        if target_k == 1:
            face_clusters = [set(range(len(faces)))]
        else:
            face_clusters = select_balanced_cuts(face_graph, target_k)

        return _face_clusters_to_graph_partition(graph, faces, face_clusters)


def extract_faces(graph: nx.Graph | nx.PlanarEmbedding) -> list[Face]:
    nx_graph, embedding = _normalize_planar_input(graph)
    raw_faces = _enumerate_faces(embedding)
    if len(raw_faces) <= 1:
        return []

    outer_index = _select_outer_face(raw_faces, nx_graph)
    faces: list[Face] = []
    for index, vertices in enumerate(raw_faces):
        if index == outer_index:
            continue
        vertex_tuple = tuple(vertices)
        edges = frozenset(_face_edges(vertex_tuple))
        faces.append(Face(
            vertices=vertex_tuple,
            edges=edges,
            weight=len(set(vertex_tuple)),
        ))
    return faces


def build_face_graph(faces: list[Face]) -> nx.Graph:
    face_graph = nx.Graph()
    for index, face in enumerate(faces):
        face_graph.add_node(index, weight=face.weight)

    owners_by_edge: dict[EdgeKey, list[int]] = defaultdict(list)
    for index, face in enumerate(faces):
        for edge in face.edges:
            owners_by_edge[edge].append(index)

    for owners in owners_by_edge.values():
        if len(owners) == 2:
            face_graph.add_edge(owners[0], owners[1])
    return face_graph


def select_balanced_cuts(face_graph: nx.Graph, k: int) -> list[set[int]]:
    if k < 2:
        raise ValueError("k must be at least 2")
    if face_graph.number_of_nodes() < k:
        raise ValueError("k cannot exceed the number of internal faces")
    if not nx.is_connected(face_graph):
        raise ValueError("face adjacency graph must be connected")

    tree = nx.minimum_spanning_tree(face_graph)
    total_weight = _component_weight(face_graph, set(tree.nodes))
    target_weight = total_weight / k

    components: list[set[int]] = [set(tree.nodes)]
    removed_edges: set[tuple[int, int]] = set()

    while len(components) < k:
        best: tuple[float, int, int, tuple[int, int], set[int], set[int]] | None = None
        for component_index, component in enumerate(components):
            if len(component) <= 1:
                continue
            sub_tree = tree.subgraph(component).copy()
            for edge in sub_tree.edges:
                candidate = sub_tree.copy()
                candidate.remove_edge(*edge)
                split = [set(nodes) for nodes in nx.connected_components(candidate)]
                if len(split) != 2:
                    continue
                left, right = split
                left_weight = _component_weight(face_graph, left)
                right_weight = _component_weight(face_graph, right)
                score = min(
                    abs(left_weight - target_weight),
                    abs(right_weight - target_weight),
                )
                tie_breaker = max(left_weight, right_weight)
                candidate_key = (
                    score,
                    tie_breaker,
                    component_index,
                    edge,
                    left,
                    right,
                )
                if best is None or candidate_key[:4] < best[:4]:
                    best = candidate_key

        if best is None:
            break

        _, _, component_index, edge, left, right = best
        removed_edges.add(tuple(sorted(edge)))
        components.pop(component_index)
        components.extend([left, right])

    partitioned_tree = tree.copy()
    partitioned_tree.remove_edges_from(removed_edges)
    clusters = [set(nodes) for nodes in nx.connected_components(partitioned_tree)]
    if len(clusters) != k:
        raise AssertionError(f"failed to produce {k} face clusters: got {len(clusters)}")
    return clusters


def faces_to_regions(
    graph: nx.Graph | nx.PlanarEmbedding,
    faces: list[Face],
    face_clusters: list[set[int]],
) -> RegionPartition:
    nx_graph, _ = _normalize_planar_input(graph)
    face_counts_by_vertex: dict[int, Counter[int]] = defaultdict(Counter)
    candidate_vertices_by_region: list[set[int]] = []

    for region_index, cluster in enumerate(face_clusters):
        vertices: set[int] = set()
        for face_index in cluster:
            face_vertices = set(faces[face_index].vertices)
            vertices.update(face_vertices)
            for vertex in face_vertices:
                face_counts_by_vertex[vertex][region_index] += 1
        candidate_vertices_by_region.append(vertices)

    assigned_regions = {
        vertex: max(counts, key=lambda region: (counts[region], -region))
        for vertex, counts in face_counts_by_vertex.items()
    }
    regions = [
        {
            vertex
            for vertex in candidates
            if assigned_regions.get(vertex) == region_index
        }
        for region_index, candidates in enumerate(candidate_vertices_by_region)
    ]

    _attach_missing_graph_vertices(nx_graph, regions)
    boundary_edges_by_region = _boundary_edges_by_region(faces, face_clusters)
    return RegionPartition(regions, boundary_edges_by_region, face_clusters)


def verify(
    graph: nx.Graph | nx.PlanarEmbedding,
    regions: list[set[int]],
    *,
    raise_on_failure: bool = True,
) -> bool:
    nx_graph, _ = _normalize_planar_input(graph)
    failures: list[str] = []

    if len(regions) == 0:
        failures.append("regions is empty")

    assigned: dict[int, int] = {}
    for region_index, region in enumerate(regions):
        if not region:
            failures.append(f"region {region_index} is empty")
            continue
        subgraph = nx_graph.subgraph(region)
        if not nx.is_connected(subgraph):
            failures.append(f"region {region_index} is not connected")
        for vertex in region:
            if vertex in assigned:
                failures.append(
                    f"vertex {vertex} appears in both region "
                    f"{assigned[vertex]} and {region_index}"
                )
            assigned[vertex] = region_index

    missing_vertices = set(nx_graph.nodes).difference(assigned)
    if missing_vertices:
        failures.append(f"unassigned vertices: {sorted(missing_vertices)}")

    boundary_edges_by_region = getattr(regions, "boundary_edges_by_region", None)
    if boundary_edges_by_region is None:
        failures.append("missing face boundary metadata for Eulerian verification")
    else:
        for region_index, boundary_edges in enumerate(boundary_edges_by_region):
            odd_vertices = _odd_boundary_vertices(boundary_edges)
            if odd_vertices:
                failures.append(
                    f"region {region_index} boundary is not Eulerian; "
                    f"odd vertices: {sorted(odd_vertices)}"
                )

    if failures:
        message = "; ".join(failures)
        if raise_on_failure:
            raise AssertionError(message)
        print(f"verification failed: {message}")
        return False
    return True


def partition(graph: nx.Graph | nx.PlanarEmbedding, k: int) -> RegionPartition:
    faces = extract_faces(graph)
    face_graph = build_face_graph(faces)
    face_clusters = select_balanced_cuts(face_graph, k)
    regions = faces_to_regions(graph, faces, face_clusters)
    verify(graph, regions)
    return regions


def _normalize_planar_input(
    graph: nx.Graph | nx.PlanarEmbedding,
) -> tuple[nx.Graph, nx.PlanarEmbedding]:
    if isinstance(graph, nx.PlanarEmbedding):
        embedding = graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(embedding.nodes)
        for u in embedding.nodes:
            for v in embedding.neighbors_cw_order(u):
                if u != v:
                    nx_graph.add_edge(u, v)
        embedding.check_structure()
        return nx_graph, embedding

    nx_graph = nx.Graph(graph)
    is_planar, embedding = nx.check_planarity(nx_graph)
    if not is_planar:
        raise ValueError("graph must be planar")
    return nx_graph, embedding


def _domain_graph_to_networkx(graph: Graph) -> nx.Graph:
    nx_graph = nx.Graph()
    for edge in graph.edges:
        u, v = edge.id
        if u != v:
            nx_graph.add_edge(u, v, weight=edge.weight)
    return nx_graph


def _face_clusters_to_graph_partition(
    graph: Graph,
    faces: list[Face],
    face_clusters: list[set[int]],
) -> GraphPartitionResult:
    weight_by_edge = {edge.id: edge.weight for edge in graph.edges}
    input_edge_by_id = {edge.id: edge for edge in graph.edges}
    boundary_edges_by_region = _boundary_edges_by_region(faces, face_clusters)
    boundary_edge_ids = set().union(*boundary_edges_by_region) if boundary_edges_by_region else set()
    directed_boundary_edges = _direct_boundary_edges(boundary_edge_ids, weight_by_edge)

    sub_graphs: list[Graph] = []
    covered_edge_ids: set[EdgeKey] = set()
    for cluster, boundary_edges in zip(face_clusters, boundary_edges_by_region):
        region_edges: list[Edge] = []
        cluster_edge_counts: Counter[EdgeKey] = Counter()
        for face_index in cluster:
            cluster_edge_counts.update(faces[face_index].edges)

        for edge_id, count in sorted(cluster_edge_counts.items()):
            if edge_id in boundary_edges:
                region_edges.append(directed_boundary_edges[edge_id])
            elif count > 1:
                u, v = edge_id
                region_edges.append(Edge(u, v, weight_by_edge[edge_id], False))
            covered_edge_ids.add(edge_id)

        sub_graphs.append(Graph(edges=region_edges))

    remaining_edges = [
        edge
        for edge_id, edge in input_edge_by_id.items()
        if edge_id not in covered_edge_ids
    ]
    return GraphPartitionResult(sub_graphs=sub_graphs, remaining_edges=remaining_edges)


def _direct_boundary_edges(
    boundary_edge_ids: set[EdgeKey],
    weight_by_edge: dict[EdgeKey, int],
) -> dict[EdgeKey, Edge]:
    boundary_graph = nx.Graph()
    boundary_graph.add_edges_from(boundary_edge_ids)
    directed: dict[EdgeKey, Edge] = {}

    for component in nx.connected_components(boundary_graph):
        component_graph = boundary_graph.subgraph(component).copy()
        if component_graph.number_of_edges() == 0:
            continue
        if nx.is_eulerian(component_graph):
            circuit_edges = nx.eulerian_circuit(component_graph)
        else:
            circuit_edges = component_graph.edges()
        for u, v in circuit_edges:
            edge_id = tuple(sorted((u, v)))
            if edge_id not in directed:
                directed[edge_id] = Edge(u, v, weight_by_edge[edge_id], True)

    return directed


def _enumerate_faces(embedding: nx.PlanarEmbedding) -> list[list[int]]:
    faces: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for u in embedding.nodes:
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in visited:
                continue
            face = list(embedding.traverse_face(u, v, mark_half_edges=visited))
            faces.append(face)
    return faces


def _select_outer_face(faces: list[list[int]], graph: nx.Graph) -> int:
    try:
        pos = nx.planar_layout(graph)
    except nx.NetworkXException:
        return max(range(len(faces)), key=lambda index: len(set(faces[index])))
    areas = [_polygon_area(face, pos) for face in faces]
    return max(range(len(faces)), key=lambda index: abs(areas[index]))


def _polygon_area(face: list[int], pos: dict[int, tuple[float, float]]) -> float:
    area = 0.0
    for index, vertex in enumerate(face):
        next_vertex = face[(index + 1) % len(face)]
        x1, y1 = pos[vertex]
        x2, y2 = pos[next_vertex]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _face_edges(vertices: tuple[int, ...]) -> set[EdgeKey]:
    return {
        tuple(sorted((vertices[index], vertices[(index + 1) % len(vertices)])))
        for index in range(len(vertices))
    }


def _component_weight(face_graph: nx.Graph, component: set[int]) -> int:
    return sum(face_graph.nodes[node].get("weight", 1) for node in component)


def _boundary_edges_by_region(
    faces: list[Face],
    face_clusters: list[set[int]],
) -> list[set[EdgeKey]]:
    boundaries: list[set[EdgeKey]] = []
    for cluster in face_clusters:
        edge_counts: Counter[EdgeKey] = Counter()
        for face_index in cluster:
            for edge in faces[face_index].edges:
                edge_counts[edge] += 1
        boundaries.append({
            edge
            for edge, count in edge_counts.items()
            if count % 2 == 1
        })
    return boundaries


def _odd_boundary_vertices(boundary_edges: set[EdgeKey]) -> set[int]:
    degree: Counter[int] = Counter()
    for u, v in boundary_edges:
        degree[u] += 1
        degree[v] += 1
    return {vertex for vertex, value in degree.items() if value % 2 == 1}


def _attach_missing_graph_vertices(nx_graph: nx.Graph, regions: list[set[int]]) -> None:
    assigned = set().union(*regions) if regions else set()
    for vertex in sorted(set(nx_graph.nodes).difference(assigned)):
        neighbor_regions = Counter(
            region_index
            for neighbor in nx_graph.neighbors(vertex)
            for region_index, region in enumerate(regions)
            if neighbor in region
        )
        if neighbor_regions:
            target = max(neighbor_regions, key=lambda index: (neighbor_regions[index], -index))
        else:
            target = min(range(len(regions)), key=lambda index: len(regions[index]))
        regions[target].add(vertex)
