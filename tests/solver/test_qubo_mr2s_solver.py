from dimod import BinaryPolynomial, Vartype

from mr2s_module.domain import Edge, Graph
from mr2s_module.solver import QuboMR2SSolver


class StubFaceCycle:
    def __init__(self, predefined_edges: set[Edge]) -> None:
        self.predefined_edges = predefined_edges
        self.calls = 0

    def run(self, graph: Graph) -> set[Edge]:
        self.calls += 1
        return self.predefined_edges


class StubPolyGenerator:
    def __init__(self) -> None:
        self.seen_graphs: list[Graph] = []

    def run(self, graph: Graph) -> BinaryPolynomial:
        self.seen_graphs.append(graph)
        return BinaryPolynomial({(): 0.0}, Vartype.BINARY)


class StubQuboSolver:
    def __init__(self) -> None:
        self.received_graph: Graph | None = None

    def run(self, qubo, graph: Graph):
        self.received_graph = graph
        return {(edge.vertices[0], edge.vertices[1]) for edge in graph.edges}


class StubEvaluator:
    def __init__(self) -> None:
        self.received_solution = None

    def run(self, solution):
        self.received_solution = solution
        return float(len(solution))


def test_run_skips_preprocessing_when_face_cycle_is_none() -> None:
    graph = Graph(edges=[Edge(1, 2, 1, False)])
    poly_generator = StubPolyGenerator()
    qubo_solver = StubQuboSolver()
    evaluator = StubEvaluator()
    solver = QuboMR2SSolver(
        face_cycle=None,
        qubo_solver=qubo_solver,
        evaluator=evaluator,
        poly_generators={poly_generator},
    )

    score = solver.run(graph)

    assert score == 1.0
    assert poly_generator.seen_graphs == [graph]
    assert qubo_solver.received_graph is graph
    assert graph.edges[0].directed is False


def test_run_applies_preprocessing_when_face_cycle_is_provided() -> None:
    graph = Graph(edges=[Edge(1, 2, 1, False)])
    predefined_edge = Edge(1, 2, 1, True)
    face_cycle = StubFaceCycle(predefined_edges={predefined_edge})
    poly_generator = StubPolyGenerator()
    qubo_solver = StubQuboSolver()
    evaluator = StubEvaluator()
    solver = QuboMR2SSolver(
        face_cycle=face_cycle,
        qubo_solver=qubo_solver,
        evaluator=evaluator,
        poly_generators={poly_generator},
    )

    score = solver.run(graph)

    assert score == 1.0
    assert face_cycle.calls == 1
    assert graph.edges == [predefined_edge]
    assert qubo_solver.received_graph is graph
