import itertools

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module.cycle import FaceCycle
from mr2s_module.domain import Edge, Graph


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


def test_empty_graph_returns_empty_set() -> None:
    assert FaceCycle().run(Graph(edges=[])) == set()


def test_non_planar_graph_returns_empty_set() -> None:
    # K5 is the canonical non-planar graph; methodology must opt out.
    pairs = list(itertools.combinations(range(5), 2))
    assert FaceCycle().run(_make_graph_from_edges(pairs)) == set()


def test_triangle_returns_its_three_edges() -> None:
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])
    result = FaceCycle().run(graph)

    assert {e.id for e in result} == {(0, 1), (0, 2), (1, 2)}


def test_k4_returns_boundary_edges_subset_of_input() -> None:
    pairs = list(itertools.combinations(range(4), 2))
    graph = _make_graph_from_edges(pairs)
    result = FaceCycle().run(graph)

    input_ids = {e.id for e in graph.edges}
    assert {e.id for e in result}.issubset(input_ids)
    assert len(result) >= 3  # at least one face boundary survives


def test_cut_vertex_components_are_processed_independently() -> None:
    # Two triangles glued at vertex 2 — graph is not biconnected.
    # Each biconnected sub-component should contribute its full triangle.
    graph = _make_graph_from_edges([
        (0, 1), (1, 2), (0, 2),
        (2, 3), (3, 4), (2, 4),
    ])
    result = FaceCycle().run(graph)
    ids = {e.id for e in result}

    assert {(0, 1), (0, 2), (1, 2)}.issubset(ids)
    assert {(2, 3), (2, 4), (3, 4)}.issubset(ids)


def test_returned_edges_are_directed_with_input_weight() -> None:
    # FaceCycle now emits oriented Edge objects so define_edge_direction can
    # replace the caller's undirected edges by id with these directed ones.
    graph = Graph(edges=[
        Edge(0, 1, 7, False),
        Edge(1, 2, 11, False),
        Edge(0, 2, 13, False),
    ])
    weight_by_id = {e.id: e.weight for e in graph.edges}

    result = FaceCycle().run(graph)
    assert len(result) == 3

    for produced in result:
        assert produced.directed is True
        assert produced.id in weight_by_id
        assert produced.weight == weight_by_id[produced.id]
        # vertices must encode a real direction (u != v) consistent with id.
        u, v = produced.vertices
        assert u != v
        assert (min(u, v), max(u, v)) == produced.id


def test_directions_form_consistent_cycle_on_triangle() -> None:
    # Triangle has a single inner face; all three edges must traverse the
    # face in one direction (either all CW or all CCW), forming a cycle.
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])

    pairs = {edge.vertices for edge in FaceCycle().run(graph)}
    out_degree: dict[int, int] = {0: 0, 1: 0, 2: 0}
    in_degree: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for u, v in pairs:
        out_degree[u] += 1
        in_degree[v] += 1

    assert all(d == 1 for d in out_degree.values())
    assert all(d == 1 for d in in_degree.values())


def test_dangling_edge_below_biconnected_threshold_is_dropped() -> None:
    # Edge (3,4) sits in a 1-edge biconnected component (a bridge),
    # which the pipeline filters out (len(bcc) < 3).
    graph = _make_graph_from_edges([
        (0, 1), (1, 2), (0, 2),
        (2, 3), (3, 4),
    ])
    result_ids = {e.id for e in FaceCycle().run(graph)}

    assert (3, 4) not in result_ids
    assert (2, 3) not in result_ids
    assert {(0, 1), (0, 2), (1, 2)}.issubset(result_ids)


def test_self_loops_in_input_are_ignored() -> None:
    graph = Graph(edges=[
        Edge(0, 0, 1, False),
        Edge(0, 1, 1, False),
        Edge(1, 2, 1, False),
        Edge(0, 2, 1, False),
    ])
    result = FaceCycle().run(graph)

    assert all(e.id[0] != e.id[1] for e in result)


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_delaunay_graph_returns_subset_of_input_edges(seed: int) -> None:
    np.random.seed(seed)  # snowball seeding uses np.random
    graph = _delaunay_graph(n=60, seed=seed)
    result = FaceCycle(target_k=6).run(graph)

    input_ids = {e.id for e in graph.edges}
    result_ids = {e.id for e in result}
    assert result_ids.issubset(input_ids)
    assert len(result_ids) > 0


def test_target_k_is_capped_to_face_count() -> None:
    # Triangle has only one inner face; target_k=50 must not crash or stall.
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])
    result = FaceCycle(target_k=50).run(graph)
    assert {e.id for e in result} == {(0, 1), (0, 2), (1, 2)}


def test_final_boundary_separates_bipartite_face_groups() -> None:
    # Sanity check the methodology's invariant: removing boundary edges from
    # the face-dual partitions inner faces into bipartite-compatible groups.
    np.random.seed(2026)
    graph = _delaunay_graph(n=80, seed=2026)
    fc = FaceCycle(target_k=8)
    result = fc.run(graph)

    nx_graph = nx.Graph()
    for edge in graph.edges:
        u, v = edge.id
        if u != v:
            nx_graph.add_edge(u, v)
    boundary_pairs = {e.id for e in result}

    is_planar, _ = nx.check_planarity(nx_graph)
    assert is_planar
    # Every returned edge must be present in the original graph.
    for u, v in boundary_pairs:
        assert nx_graph.has_edge(u, v)
