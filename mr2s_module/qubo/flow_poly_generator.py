from dimod import BinaryPolynomial, Vartype

from mr2s_module.domain import Edge
from mr2s_module.protocols import Graph
from mr2s_module.util import get_indicator_function, add_polys, multiply_polys


class FlowPolyGenerator:

  @staticmethod
  def _get_a_term(vertex: int, incident_edges: set[Edge]) -> BinaryPolynomial:
    term = BinaryPolynomial({}, Vartype.BINARY)
    for edge in incident_edges:

      if edge.directed:
        temp = 1 if edge.vertices[0] == vertex else -1
        term = add_polys(term, BinaryPolynomial({(): temp}, Vartype.BINARY))
        continue

      # 나가는 방향이 +weight, 들어오는 방향이 -weight. 흐름보존을 가중치 기반으로 다뤄
      # SA 솔버 _build_flow_score (가중 in/out) 와 목적을 일치시킨다. 변수명은 edge.to_key()
      # (id 기반) 라 같은 양끝점의 평행간선도 서로 다른 변수를 갖는다.
      indicator_function = get_indicator_function(
        vertex, edge.other_vertex(vertex), edge.weight, edge.to_key()
      )
      indicator_function.scale(2)
      temp = add_polys(indicator_function, BinaryPolynomial({(): -1}, Vartype.BINARY))
      term = add_polys(term, temp)
    return multiply_polys(term, term)

  def _get_total_term(self, vertices: set[int], edges: list[Edge]) -> BinaryPolynomial:
    term = BinaryPolynomial({}, Vartype.BINARY)

    for vertex in vertices:
      temp = self._get_a_term(vertex, {edge for edge in edges if vertex in edge.vertices})
      term = add_polys(term, temp)

    return term

  def run(self, graph: Graph) -> BinaryPolynomial:
    if graph.is_empty():
      return BinaryPolynomial({}, Vartype.BINARY)

    vertices = graph.get_vertices()

    return self._get_total_term(vertices, list(graph.edges.values()))