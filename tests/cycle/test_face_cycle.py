import itertools

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module.cycle import FaceCycle
from mr2s_module.domain import Edge, Graph, GraphPartitionResult


def _make_graph_from_edges(pairs: list[tuple[int, int]]) -> Graph:
    return Graph(edges=[Edge(u, v, 1, False) for u, v in pairs])


def _delaunay_graph(n: int, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    tri = Delaunay(points)
    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for simplex in tri.simplices:
        for u, v in itertools.combinations(simplex, 2):
            u, v = int(u), int(v)
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(key[0], key[1], 1, False))
    return Graph(edges=edges)


def _directed_remaining(result: GraphPartitionResult) -> list[Edge]:
    return [e for e in result.remaining_edges if e.directed]


def test_empty_graph_returns_empty_partition() -> None:
    result = FaceCycle().run(Graph(edges=[]))
    assert result.sub_graphs == []
    assert result.remaining_edges == []


def test_non_planar_graph_opts_out() -> None:
    # K5 is the canonical non-planar graph; methodology must opt out and
    # return the input untouched in remaining_edges.
    pairs = list(itertools.combinations(range(5), 2))
    graph = _make_graph_from_edges(pairs)
    result = FaceCycle().run(graph)

    assert result.sub_graphs == []
    assert {e.id for e in result.remaining_edges} == {e.id for e in graph.edges}
    assert not any(e.directed for e in result.remaining_edges)


def test_triangle_directs_its_three_boundary_edges() -> None:
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])
    result = FaceCycle().run(graph)

    directed = _directed_remaining(result)
    assert {e.id for e in directed} == {(0, 1), (0, 2), (1, 2)}


def test_k4_directed_boundary_is_subset_of_input() -> None:
    pairs = list(itertools.combinations(range(4), 2))
    graph = _make_graph_from_edges(pairs)
    result = FaceCycle().run(graph)

    input_ids = {e.id for e in graph.edges}
    directed = _directed_remaining(result)
    assert {e.id for e in directed}.issubset(input_ids)
    assert len(directed) >= 3  # at least one face boundary survives


def test_cut_vertex_components_are_processed_independently() -> None:
    # Two triangles glued at vertex 2 — graph is not biconnected.
    # Each biconnected sub-component should contribute its full triangle
    # to remaining_edges with directions assigned.
    graph = _make_graph_from_edges([
        (0, 1), (1, 2), (0, 2),
        (2, 3), (3, 4), (2, 4),
    ])
    result = FaceCycle().run(graph)
    ids = {e.id for e in _directed_remaining(result)}

    assert {(0, 1), (0, 2), (1, 2)}.issubset(ids)
    assert {(2, 3), (2, 4), (3, 4)}.issubset(ids)


def test_directed_remaining_edges_carry_input_weight() -> None:
    # FaceCycle emits oriented Edge objects in remaining_edges so that
    # define_edge_direction can replace the caller's undirected edges by id.
    graph = Graph(edges=[
        Edge(0, 1, 7, False),
        Edge(1, 2, 11, False),
        Edge(0, 2, 13, False),
    ])
    weight_by_id = {e.id: e.weight for e in graph.edges}

    directed = _directed_remaining(FaceCycle().run(graph))
    assert len(directed) == 3

    for produced in directed:
        assert produced.directed is True
        assert produced.id in weight_by_id
        assert produced.weight == weight_by_id[produced.id]
        u, v = produced.vertices
        assert u != v
        assert (min(u, v), max(u, v)) == produced.id


def test_directions_form_consistent_cycle_on_triangle() -> None:
    # Triangle has a single inner face; all three edges must traverse the
    # face in one direction (either all CW or all CCW), forming a cycle.
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])

    pairs = {edge.vertices for edge in _directed_remaining(FaceCycle().run(graph))}
    out_degree: dict[int, int] = {0: 0, 1: 0, 2: 0}
    in_degree: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for u, v in pairs:
        out_degree[u] += 1
        in_degree[v] += 1

    assert all(d == 1 for d in out_degree.values())
    assert all(d == 1 for d in in_degree.values())


def test_bridge_edges_pass_through_remaining_undirected() -> None:
    # (3,4) and (2,3) sit in 1-edge biconnected components (bridges); the
    # pipeline filters them out (len(bcc) < 3), so they must land in
    # remaining_edges untouched (still undirected).
    graph = _make_graph_from_edges([
        (0, 1), (1, 2), (0, 2),
        (2, 3), (3, 4),
    ])
    result = FaceCycle().run(graph)

    by_id = {e.id: e for e in result.remaining_edges}
    assert (3, 4) in by_id and not by_id[(3, 4)].directed
    assert (2, 3) in by_id and not by_id[(2, 3)].directed
    triangle_ids = {(0, 1), (0, 2), (1, 2)}
    assert triangle_ids.issubset({e.id for e in _directed_remaining(result)})


def test_self_loops_are_never_directed() -> None:
    graph = Graph(edges=[
        Edge(0, 0, 1, False),
        Edge(0, 1, 1, False),
        Edge(1, 2, 1, False),
        Edge(0, 2, 1, False),
    ])
    result = FaceCycle().run(graph)

    for edge in result.remaining_edges:
        if edge.id[0] == edge.id[1]:
            assert not edge.directed
    assert all(e.id[0] != e.id[1] for e in _directed_remaining(result))


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_delaunay_partition_is_complete_cover(seed: int) -> None:
    np.random.seed(seed)  # snowball seeding uses np.random
    graph = _delaunay_graph(n=60, seed=seed)
    result = FaceCycle(target_k=6).run(graph)

    input_ids = {e.id for e in graph.edges}
    directed_ids = {e.id for e in _directed_remaining(result)}
    assert directed_ids.issubset(input_ids)
    assert len(directed_ids) > 0
    # Single O(E) classification must conserve every input edge.
    sub_count = sum(len(sg.edges) for sg in result.sub_graphs)
    assert sub_count + len(result.remaining_edges) == len(graph.edges)


def test_target_k_is_capped_to_face_count() -> None:
    # Triangle has only one inner face; target_k=50 must not crash or stall.
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])
    result = FaceCycle(target_k=50).run(graph)
    assert {e.id for e in _directed_remaining(result)} == {(0, 1), (0, 2), (1, 2)}


def test_directed_boundary_edges_belong_to_input_graph() -> None:
    np.random.seed(2026)
    graph = _delaunay_graph(n=80, seed=2026)
    fc = FaceCycle(target_k=8)
    result = fc.run(graph)

    nx_graph = nx.Graph()
    for edge in graph.edges:
        u, v = edge.id
        if u != v:
            nx_graph.add_edge(u, v)

    is_planar, _ = nx.check_planarity(nx_graph)
    assert is_planar
    for edge in _directed_remaining(result):
        u, v = edge.id
        assert nx_graph.has_edge(u, v)


def test_macro_internal_edges_disjoint_from_remaining() -> None:
    # Sub-graphs hold only edges whose two adjacent faces share a macro;
    # those edges must not also appear in remaining_edges.
    np.random.seed(31)
    graph = _delaunay_graph(n=50, seed=31)
    result = FaceCycle(target_k=4).run(graph)

    sub_ids: set[tuple[int, int]] = set()
    for sg in result.sub_graphs:
        for edge in sg.edges:
            sub_ids.add(edge.id)
    remaining_ids = {e.id for e in result.remaining_edges}
    assert sub_ids.isdisjoint(remaining_ids)
