from mr2s_module.domain import Edge, Graph
from mr2s_module.util import domain_graph_to_networkx, domain_graph_to_networkx_multi


def test_multi_conversion_preserves_parallel_edges_keyed_by_edge_id() -> None:
  edge_a = Edge(0, 1, 3, False)
  edge_b = Edge(0, 1, 5, False)
  edge_c = Edge(1, 2, 2, False)
  graph = Graph(edges=[edge_a, edge_b, edge_c])

  nx_graph = domain_graph_to_networkx_multi(graph)

  assert nx_graph.number_of_edges() == 3
  assert set(nx_graph[0][1].keys()) == {edge_a.id, edge_b.id}
  assert nx_graph[0][1][edge_a.id]["weight"] == 3
  assert nx_graph[0][1][edge_b.id]["weight"] == 5
  assert nx_graph[1][2][edge_c.id]["weight"] == 2


def test_multi_conversion_skips_self_loops_but_keeps_their_vertices() -> None:
  graph = Graph(edges=[
    Edge(0, 1, 1, False),
    Edge(2, 2, 4, False),
  ])

  nx_graph = domain_graph_to_networkx_multi(graph)

  assert nx_graph.number_of_edges() == 1
  assert 2 in nx_graph.nodes
  assert nx_graph.degree(2) == 0


def test_simple_conversion_still_collapses_parallel_edges_to_min_weight() -> None:
  graph = Graph(edges=[
    Edge(0, 1, 3, False),
    Edge(0, 1, 5, False),
  ])

  nx_graph = domain_graph_to_networkx(graph)

  assert nx_graph.number_of_edges() == 1
  assert nx_graph[0][1]["weight"] == 3
