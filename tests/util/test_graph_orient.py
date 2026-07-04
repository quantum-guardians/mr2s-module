import networkx as nx

from mr2s_module.domain import Edge, Graph
from mr2s_module.util import domain_graph_to_networkx_multi, robbins_orient


def test_robbins_orient_keys_result_by_domain_edge_id() -> None:
  edge_a = Edge(0, 1, 1, False)
  edge_b = Edge(1, 2, 1, False)
  edge_c = Edge(0, 2, 1, False)
  graph = Graph(edges=[edge_a, edge_b, edge_c])

  oriented = robbins_orient(domain_graph_to_networkx_multi(graph), 0)

  assert set(oriented.keys()) == {edge_a.id, edge_b.id, edge_c.id}
  for edge_id, edge in oriented.items():
    assert edge.id == edge_id
    assert edge.directed

  D = nx.DiGraph()
  D.add_edges_from(edge.vertices for edge in oriented.values())
  assert nx.is_strongly_connected(D)


def test_robbins_orient_alternates_parallel_copies() -> None:
  edge_a = Edge(0, 1, 3, False)
  edge_b = Edge(0, 1, 5, False)
  graph = Graph(edges=[edge_a, edge_b])

  oriented = robbins_orient(domain_graph_to_networkx_multi(graph), 0)

  assert set(oriented.keys()) == {edge_a.id, edge_b.id}
  assert oriented[edge_a.id].vertices == tuple(reversed(oriented[edge_b.id].vertices))
  assert oriented[edge_a.id].weight == 3
  assert oriented[edge_b.id].weight == 5

  D = nx.DiGraph()
  D.add_edges_from(edge.vertices for edge in oriented.values())
  assert nx.is_strongly_connected(D)
