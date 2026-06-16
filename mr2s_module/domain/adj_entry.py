from typing import NamedTuple


class AdjEntry(NamedTuple):
  vertex: int # Could be destination or origin vertex
  weight: int
  directed: bool # if directed is true, then vertex is destination
  edge_key: str = ""  # QUBO 변수명(= Edge.to_key()). 평행간선을 서로 다른 변수로 분리.