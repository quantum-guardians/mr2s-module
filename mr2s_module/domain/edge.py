import itertools


class Edge:
  id: int
  vertices: tuple[int, int]  # directed=True 면 (tail, head), 아니면 sorted endpoints.
  weight: int
  directed: bool

  # 인스턴스마다 고유한 int id. 같은 양 끝점을 잇는 평행간선도 서로 다른 id 를 갖는다
  # (frozenset(endpoints) id 시절의 dict 키/QUBO 변수 충돌을 제거). 양끝점 동일성은
  # endpoint_key() 로 따로 표현한다.
  _id_counter = itertools.count()

  def __init__(
    self,
    vertex1: int,
    vertex2: int,
    weight: int,
    directed: bool,
    *,
    id: int | None = None,
  ):
    self.id = next(Edge._id_counter) if id is None else id
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

  def endpoint_key(self) -> frozenset[int]:
    """양끝점 기반 동일성 키. 방향/평행과 무관하게 같은 두 정점을 잇는 간선을 묶는다."""
    return frozenset(self._endpoints)

  def other_vertex(self, vertex: int) -> int:
    if vertex not in self.vertices:
      raise ValueError(f"Vertex {vertex} is not in this edge.")

    v1, v2 = self.vertices
    return v2 if vertex == v1 else v1

  def to_key(self) -> str:
    return f"e_{self.id}"

  def flip(self) -> "Edge":
    """방향이 뒤집힌 새 Edge 반환. ILS hypothesis 변형에서 사용.

    방향만 뒤집힌 *동일한* 간선이므로 id 를 보존한다.
    """
    return Edge(self.vertices[1], self.vertices[0], self.weight, self.directed, id=self.id)
