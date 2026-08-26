from pathlib import Path

import networkx as nx
import pytest

from experiments import config
from experiments.graphs import (
    build_record,
    generate_delaunay,
    load_graph,
    save_graph,
    thin_biconnected,
    to_domain_graph,
    write_manifest,
)
from tests.util.graph_fixtures import delaunay_graph_with_pos


def test_generate_delaunay_matches_test_fixture() -> None:
    graph, points = generate_delaunay(30, 0)
    fixture_graph, fixture_pos = delaunay_graph_with_pos(30, 0)
    fixture_edges = {edge.endpoints() for edge in fixture_graph.edges.values()}
    assert {(min(u, v), max(u, v)) for u, v in graph.edges()} == fixture_edges
    assert all(tuple(points[i]) == tuple(fixture_pos[i]) for i in range(30))
    assert nx.check_planarity(graph)[0]
    assert nx.is_biconnected(graph)


def test_thin_biconnected_keeps_biconnectivity_and_is_deterministic() -> None:
    base, _ = generate_delaunay(60, 1)
    thinned, removed = thin_biconnected(base, seed=1, remove_ratio=0.3)
    again, removed_again = thin_biconnected(base, seed=1, remove_ratio=0.3)
    assert removed == removed_again
    assert set(thinned.edges()) == set(again.edges())
    assert removed <= round(base.number_of_edges() * 0.3)
    assert nx.is_biconnected(thinned)
    assert nx.check_planarity(thinned)[0]
    assert thinned.number_of_nodes() == base.number_of_nodes()


def test_thin_biconnected_nested_across_ratios() -> None:
    base, _ = generate_delaunay(60, 2)
    p10, _ = thin_biconnected(base, seed=2, remove_ratio=0.1)
    p30, _ = thin_biconnected(base, seed=2, remove_ratio=0.3)
    assert set(p30.edges()) <= set(p10.edges())


def test_thin_biconnected_records_shortfall_on_small_graph() -> None:
    cycle = nx.cycle_graph(6)
    thinned, removed = thin_biconnected(cycle, seed=0, remove_ratio=0.5)
    assert removed == 0
    assert set(thinned.edges()) == set(cycle.edges())


def test_build_record_and_json_roundtrip(tmp_path: Path) -> None:
    record = build_record(40, 3, 0.3)
    assert record.graph_id == config.graph_id(40, 3, 0.3)
    assert record.n_edges == len(record.edges)
    assert record.edges == sorted(record.edges)
    assert all(u < v for u, v in record.edges)
    assert 0.0 < record.remove_ratio_actual <= 0.3 + 1e-9

    path = tmp_path / record.path_name
    save_graph(path, record)
    assert load_graph(path) == record

    manifest = write_manifest(tmp_path)
    text = manifest.read_text()
    assert record.graph_id in text
    assert "True,True" in text


def test_to_domain_graph_preserves_edge_order() -> None:
    record = build_record(25, 0, 0.0)
    graph = to_domain_graph(record)
    assert [edge.endpoints() for edge in graph.edges.values()] == record.edges
    assert all(not edge.directed and edge.weight == 1 for edge in graph.edges.values())
    assert graph.get_vertices() == set(range(record.n_vertices))


def test_thin_rejects_invalid_ratio() -> None:
    base, _ = generate_delaunay(10, 0)
    with pytest.raises(ValueError):
        thin_biconnected(base, seed=0, remove_ratio=1.0)
