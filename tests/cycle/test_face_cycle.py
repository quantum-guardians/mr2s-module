import itertools

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import Delaunay

from mr2s_module.cycle import FaceCycle
from mr2s_module.cycle.face_cycle import _ComponentPartition
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

    directed = result.directed_edges()
    assert {e.id for e in directed} == {(0, 1), (0, 2), (1, 2)}


def test_k4_directed_boundary_is_subset_of_input() -> None:
    pairs = list(itertools.combinations(range(4), 2))
    graph = _make_graph_from_edges(pairs)
    result = FaceCycle().run(graph)

    input_ids = {e.id for e in graph.edges}
    directed = result.directed_edges()
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
    ids = {e.id for e in result.directed_edges()}

    assert {(0, 1), (0, 2), (1, 2)}.issubset(ids)
    assert {(2, 3), (2, 4), (3, 4)}.issubset(ids)


def test_directed_edges_carry_input_weight() -> None:
    # FaceCycle emits oriented Edge objects in remaining_edges so that
    # define_edge_direction can replace the caller's undirected edges by id.
    graph = Graph(edges=[
        Edge(0, 1, 7, False),
        Edge(1, 2, 11, False),
        Edge(0, 2, 13, False),
    ])
    weight_by_id = {e.id: e.weight for e in graph.edges}

    directed = FaceCycle().run(graph).directed_edges()
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

    pairs = {edge.vertices for edge in FaceCycle().run(graph).directed_edges()}
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
    assert triangle_ids.issubset({e.id for e in result.directed_edges()})


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
    assert all(e.id[0] != e.id[1] for e in result.directed_edges())


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_delaunay_partition_is_complete_cover(seed: int) -> None:
    np.random.seed(seed)  # snowball seeding uses np.random
    graph = _delaunay_graph(n=60, seed=seed)
    result = FaceCycle(target_k=6).run(graph)

    input_ids = {e.id for e in graph.edges}
    directed_ids = {e.id for e in result.directed_edges()}
    assert directed_ids.issubset(input_ids)
    assert len(directed_ids) > 0
    covered_ids = {e.id for sg in result.sub_graphs for e in sg.edges} | {
        e.id for e in result.remaining_edges
    }
    assert covered_ids == input_ids


def test_target_k_is_capped_to_face_count() -> None:
    # Triangle has only one inner face; target_k=50 must not crash or stall.
    graph = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])
    result = FaceCycle(target_k=50).run(graph)
    assert {e.id for e in result.directed_edges()} == {(0, 1), (0, 2), (1, 2)}


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
    for edge in result.directed_edges():
        u, v = edge.id
        assert nx_graph.has_edge(u, v)


def test_outline_of_returns_directed_boundary_touching_macro() -> None:
    # Triangle: 단일 macro 의 외각 3변.
    triangle = _make_graph_from_edges([(0, 1), (1, 2), (0, 2)])
    triangle_partition = FaceCycle().run(triangle)
    assert len(triangle_partition.sub_graphs) == 1
    triangle_outline = triangle_partition.outline_of(0)
    assert {e.id for e in triangle_outline} == {(0, 1), (0, 2), (1, 2)}
    assert all(e.directed for e in triangle_outline)

    # Multi-face: outline_of(i) 는 sub_graphs[i] 의 directed 간선과 동일 (id 기준).
    np.random.seed(31)
    graph = _delaunay_graph(n=50, seed=31)
    partition = FaceCycle(target_k=4).run(graph)

    for macro_id in range(len(partition.sub_graphs)):
        outline = partition.outline_of(macro_id)
        sg_directed = [e for e in partition.sub_graphs[macro_id].edges if e.directed]
        assert all(e.directed for e in outline)
        assert {id(e) for e in outline} == {id(e) for e in sg_directed}

    # 공유 boundary 는 양쪽 macro 에 동일 Edge 인스턴스로 등장해야 한다.
    n = len(partition.sub_graphs)
    by_id = [{e.id: id(e) for e in partition.outline_of(i)} for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for shared_id in by_id[i].keys() & by_id[j].keys():
                assert by_id[i][shared_id] == by_id[j][shared_id]


def test_get_inner_subgraph_returns_internal_edges_only() -> None:
    np.random.seed(31)
    graph = _delaunay_graph(n=50, seed=31)
    partition = FaceCycle(target_k=4).run(graph)

    for i, sg in enumerate(partition.sub_graphs):
        inner = partition.get_inner_subgraph(i)
        assert all(not e.directed for e in inner.edges)
        expected_ids = {e.id for e in sg.edges if not e.directed}
        assert {e.id for e in inner.edges} == expected_ids


def test_get_subgraph_combines_inner_and_outline() -> None:
    np.random.seed(31)
    graph = _delaunay_graph(n=50, seed=31)
    partition = FaceCycle(target_k=4).run(graph)

    for i in range(len(partition.sub_graphs)):
        inner = partition.get_inner_subgraph(i)
        outline = partition.outline_of(i)
        combined = partition.get_subgraph(i)

        assert combined is partition.sub_graphs[i]
        directed = [e for e in combined.edges if e.directed]
        undirected = [e for e in combined.edges if not e.directed]
        assert {e.id for e in directed} == {e.id for e in outline}
        assert {e.id for e in undirected} == {e.id for e in inner.edges}


def test_sub_graphs_disjoint_from_remaining() -> None:
    np.random.seed(31)
    graph = _delaunay_graph(n=50, seed=31)
    result = FaceCycle(target_k=4).run(graph)

    sub_ids = {e.id for sg in result.sub_graphs for e in sg.edges}
    remaining_ids = {e.id for e in result.remaining_edges}
    assert sub_ids.isdisjoint(remaining_ids)


def test_raises_when_undirected_edge_overlaps_between_subgraphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a partition bug: one undirected boundary edge assigned to two macros.
    graph = _make_graph_from_edges([(0, 1)])
    cycle = FaceCycle()

    monkeypatch.setattr(
        cycle,
        "_extract_biconnected_components",
        lambda _graph: [object()],
    )
    monkeypatch.setattr(
        cycle,
        "_partition_component",
        lambda _component: _ComponentPartition(
            macro_internal_edges=[set(), set()],
            macro_outline_keys=[{(0, 1)}, {(0, 1)}],
            directed_pairs=set(),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"Undirected edge overlap detected across subgraphs",
    ):
        cycle.run(graph)
