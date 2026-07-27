import itertools


class Edge:
  id: int
  vertices: tuple[int, int]  # directed=True 면 (tail, head), 아니면 sorted endpoints.
  weight: float  # 정수 입력 일반적, 축약 super edge 는 harmonic 합성으로 실수.
  directed: bool

  # 자동 유니크 id 카운터. 생성 순서 결정적(재현성). 평행 간선마다 독립 id.
  _id_counter = itertools.count()

  def __init__(self, vertex1: int, vertex2: int, weight: float, directed: bool):
    self.id = next(Edge._id_counter)
    self._endpoints = (vertex1, vertex2) if vertex1 <= vertex2 else (vertex2, vertex1)
    if directed:
      self.vertices = (vertex1, vertex2)
    else:
      self.vertices = self._endpoints
    self.weight = weight
    self.directed = directed

  def endpoints(self) -> tuple[int, int]:
    """무방향 정렬된 (u, v). self-loop 면 (v, v)."""
    return self._endpoints

  def other_vertex(self, vertex: int) -> int:
    if vertex not in self.vertices:
      raise ValueError(f"Vertex {vertex} is not in this edge.")

    v1, v2 = self.vertices
    return v2 if vertex == v1 else v1

  def to_key(self) -> str:
    return f"e_{self.id}"

  def oriented(self, tail: int, head: int) -> "Edge":
    """같은 논리 간선의 방향 버전(directed). 원본 id 를 잇는다(비파괴)."""
    if tuple(sorted((tail, head))) != self._endpoints:
      raise ValueError(
        f"Direction ({tail}, {head}) does not match edge endpoints {self._endpoints}."
      )
    edge = Edge(tail, head, self.weight, True)
    edge.id = self.id
    return edge

  def set_direction(self, tail: int, head: int) -> None:
    """방향을 in-place 로 박는다. id/끝점 유지."""
    if tuple(sorted((tail, head))) != self._endpoints:
      raise ValueError(
        f"Direction ({tail}, {head}) does not match edge endpoints {self._endpoints}."
      )
    self.vertices = (tail, head)
    self.directed = True

  def flip(self) -> "Edge":
    """방향이 뒤집힌 새 Edge 반환. ILS hypothesis 변형에서 사용. id 유지."""
    edge = Edge(self.vertices[1], self.vertices[0], self.weight, self.directed)
    edge.id = self.id
    return edge
