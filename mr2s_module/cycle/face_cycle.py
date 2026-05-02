from mr2s_module.domain.edge import Edge
from mr2s_module.domain.graph import Graph


class FaceCycle:
    def run(self, graph: Graph) -> set[Edge]:
        raise NotImplementedError
