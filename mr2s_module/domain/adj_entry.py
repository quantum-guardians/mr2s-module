from typing import NamedTuple


class AdjEntry(NamedTuple):
  vertex: int # Could be destination or origin vertex
  weight: int
  directed: bool # if directed is true, then vertex is destination