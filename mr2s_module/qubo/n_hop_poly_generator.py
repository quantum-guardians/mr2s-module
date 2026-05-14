from dataclasses import dataclass

from dimod import BinaryPolynomial, Vartype

from mr2s_module.domain import AdjEntry
from mr2s_module.protocols import Graph
from mr2s_module.util import get_indicator_function, add_polys, multiply_polys


@dataclass
class NHop:
  n: int
  weight: int

@dataclass
class SmallWorldSpec:
  n_hops: list[NHop]

@dataclass
class NHopPolyGenerator:

  small_world_spec: SmallWorldSpec = None

  def _get_n_hop_polynomial(
      self,
      n: int,
      last_vertex: int,
      adj: dict[int, list[AdjEntry]],
      used_vertices: set[int],
      current_polynomial: BinaryPolynomial
  ) -> BinaryPolynomial:
    if n == 0: return current_polynomial

    term_n = BinaryPolynomial({}, Vartype.BINARY)

    for entry in adj.get(last_vertex, []):
      if entry.vertex in used_vertices: continue

      used_vertices.add(entry.vertex)
      step_poly = BinaryPolynomial({(): entry.weight}, Vartype.BINARY) \
        if entry.directed else get_indicator_function(last_vertex, entry.vertex, entry.weight)
      temp = self._get_n_hop_polynomial(n-1, entry.vertex, adj, used_vertices, step_poly)
      term_n = add_polys(term_n, temp)
      used_vertices.remove(entry.vertex)

    return multiply_polys(term_n, current_polynomial)

  def _get_total_n_hop_polynomial(
      self,
      n_hop: NHop,
      vertices: set[int],
      adj: dict[int, list[AdjEntry]]
  ) -> BinaryPolynomial:
    term_n = BinaryPolynomial({}, Vartype.BINARY)
    used_vertices = set()
    for vertex in vertices:
      initial_poly = BinaryPolynomial({(): 1.0}, Vartype.BINARY)
      used_vertices.add(vertex)
      temp = self._get_n_hop_polynomial(n_hop.n, vertex, adj, used_vertices, initial_poly)
      used_vertices.remove(vertex)
      term_n = add_polys(term_n, temp)
    term_n.scale(n_hop.weight)
    return term_n

  def _build_polynomial(
      self, vertices: set[int], adj: dict[int, list[AdjEntry]]
  ) -> BinaryPolynomial:
    terms = BinaryPolynomial({}, Vartype.BINARY)
    for n_hop in self.small_world_spec.n_hops:
      temp = self._get_total_n_hop_polynomial(n_hop, vertices, adj)
      terms = add_polys(terms, temp)
    terms.scale(-1)
    return terms

  def run(self, graph: Graph) -> BinaryPolynomial:
    if graph.is_empty():
      return BinaryPolynomial({}, Vartype.BINARY)

    vertices = graph.get_vertices()
    adj = graph.get_adjacency_dict()
    return self._build_polynomial(vertices, adj)