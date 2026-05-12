import random

import networkx as nx
import pytest

from mr2s_module.cycle.planar_region_partition import (
    PlanarRegionFaceCycle,
    build_face_graph,
    extract_faces,
    partition,
    select_balanced_cuts,
    verify,
)
from mr2s_module.domain import Edge, Graph


def _print_result(name: str, graph: nx.Graph, regions: list[set[int]]) -> None:
    eulerian = verify(regions=regions, graph=graph, raise_on_failure=False)
    sizes = [len(region) for region in regions]
    print(f"{name}: region_sizes={sizes}, eulerian={eulerian}")


def _random_planar_graph(node_count: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.random_labeled_tree(node_count, seed=seed)
    candidates = [
        (u, v)
        for u in graph.nodes
        for v in graph.nodes
        if u < v and not graph.has_edge(u, v)
    ]
    rng.shuffle(candidates)

    for u, v in candidates:
        graph.add_edge(u, v)
        if not nx.check_planarity(graph)[0]:
            graph.remove_edge(u, v)
    return graph


def test_extract_faces_excludes_outer_face() -> None:
    graph = nx.grid_2d_graph(3, 3)
    faces = extract_faces(graph)

    assert len(faces) == 4
    assert all(face.weight == 4 for face in faces)


def test_select_balanced_cuts_returns_connected_face_clusters() -> None:
    graph = nx.grid_2d_graph(3, 3)
    faces = extract_faces(graph)
    face_graph = build_face_graph(faces)

    clusters = select_balanced_cuts(face_graph, 3)

    assert len(clusters) == 3
    assert all(nx.is_connected(face_graph.subgraph(cluster)) for cluster in clusters)


def test_partition_3x3_grid_k3_prints_balance_and_eulerian() -> None:
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))

    regions = partition(graph, 3)

    assert len(regions) == 3
    assert verify(graph, regions)
    _print_result("3x3 grid k=3", graph, regions)


def test_partition_icosahedral_graph_k4_prints_balance_and_eulerian() -> None:
    graph = nx.icosahedral_graph()

    regions = partition(graph, 4)

    assert len(regions) == 4
    assert verify(graph, regions)
    _print_result("icosahedral k=4", graph, regions)


def test_partition_random_planar_graph_k3_prints_balance_and_eulerian() -> None:
    graph = _random_planar_graph(20, seed=7)

    regions = partition(graph, 3)

    assert len(regions) == 3
    assert verify(graph, regions)
    _print_result("random planar n=20 k=3", graph, regions)


def test_planar_region_face_cycle_returns_graph_partition_result() -> None:
    graph = Graph(edges=[
        Edge(0, 1, 1, False),
        Edge(1, 2, 1, False),
        Edge(2, 3, 1, False),
        Edge(0, 3, 1, False),
        Edge(0, 2, 1, False),
    ])

    result = PlanarRegionFaceCycle(target_k=2).run(graph)

    assert len(result.sub_graphs) == 2
    assert result.remaining_edges == []
    assert {edge.id for edge in result.directed_edges()}.issubset(
        {edge.id for edge in graph.edges}
    )
    assert all(
        edge.directed
        for sub_graph in result.sub_graphs
        for edge in sub_graph.edges
    )


def test_planar_region_face_cycle_keeps_bridges_remaining() -> None:
    graph = Graph(edges=[
        Edge(0, 1, 1, False),
        Edge(1, 2, 1, False),
        Edge(0, 2, 1, False),
        Edge(2, 3, 1, False),
    ])

    result = PlanarRegionFaceCycle(target_k=2).run(graph)

    assert len(result.sub_graphs) == 1
    assert {edge.id for edge in result.remaining_edges} == {(2, 3)}
    assert {edge.id for edge in result.directed_edges()} == {(0, 1), (0, 2), (1, 2)}


def test_planar_region_face_cycle_non_planar_graph_opts_out() -> None:
    graph = Graph(edges=[
        Edge(u, v, 1, False)
        for u in range(5)
        for v in range(u + 1, 5)
    ])

    result = PlanarRegionFaceCycle(target_k=2).run(graph)

    assert result.sub_graphs == []
    assert {edge.id for edge in result.remaining_edges} == {edge.id for edge in graph.edges}


def test_planar_region_face_cycle_directed_input_raises() -> None:
    graph = Graph(edges=[
        Edge(0, 1, 1, True),
        Edge(1, 2, 1, False),
    ])

    with pytest.raises(ValueError, match="undirected input graph"):
        PlanarRegionFaceCycle(target_k=2).run(graph)
