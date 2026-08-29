import dimod

from experiments.solvers import (
    RecordingSolver,
    SeededSampler,
    build_poly_generators,
    build_solver,
    canonical_bqm,
)
from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Graph
from mr2s_module.qubo import NHop, NHopPolyGenerator, QuboSolver
from mr2s_module.reduction import ReductionMr2sSolver, SuperEdgeWeight
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.degeneracy_pruning import (
    DegeneracyPruningFaceCyclePartitionStrategy,
)
from mr2s_module.solver.predefined import create_dnc_qubo_sa_solver
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver


def test_build_solver_matches_predefined_factory_structure() -> None:
    reference = create_dnc_qubo_sa_solver()
    assert isinstance(reference, ReductionMr2sSolver)
    reference_dnc = reference.mr2s_solver
    assert isinstance(reference_dnc, DnCMr2sSolver)

    top, recorder = build_solver((2, 3), True, seed=7, num_reads=10)
    assert isinstance(top, ReductionMr2sSolver)
    assert (
        top.super_edge_weight == reference.super_edge_weight == SuperEdgeWeight.HARMONIC
    )
    assert top.mr2s_solver is recorder
    dnc = recorder.inner
    assert isinstance(dnc, DnCMr2sSolver)
    assert type(dnc.graph_partition_strategy) is type(
        reference_dnc.graph_partition_strategy
    )
    assert isinstance(
        dnc.graph_partition_strategy, DegeneracyPruningFaceCyclePartitionStrategy
    )
    assert isinstance(dnc.face_cycle, FaceClusterPartition)
    assert dnc.face_cycle.target_k == reference_dnc.face_cycle.target_k == 2
    assert isinstance(dnc.face_cycle.clusterer, KMeansFaceClusterer)
    assert isinstance(dnc.mr2s_solver, QuboMR2SSolver)
    assert dnc.graph_partition_strategy.mr2s_solver is dnc.mr2s_solver


def test_build_solver_injects_hops_reads_and_seed() -> None:
    top, recorder = build_solver((2, 3, 4), False, seed=99, num_reads=25)
    assert top is recorder
    assert isinstance(recorder.inner, DnCMr2sSolver)
    qubo = recorder.inner.mr2s_solver
    assert isinstance(qubo, QuboMR2SSolver)
    n_hop = [g for g in qubo.poly_generators if isinstance(g, NHopPolyGenerator)]
    assert len(n_hop) == 1
    assert n_hop[0].small_world_spec is not None
    assert n_hop[0].small_world_spec.n_hops == [NHop(2, 1), NHop(3, 1), NHop(4, 1)]
    assert isinstance(qubo.qubo_solver, QuboSolver)
    sampler = qubo.qubo_solver.sampler
    assert isinstance(sampler, SeededSampler)
    assert sampler.seed == 99
    assert qubo.qubo_solver.num_reads == 25


def test_seeded_sampler_forwards_seed_and_canonical_order() -> None:
    class Stub:
        def __init__(self) -> None:
            self.calls: list[tuple[list, dict]] = []

        def sample(self, bqm, **kwargs):
            self.calls.append((list(bqm.variables), kwargs))
            return bqm

    bqm = dimod.BinaryQuadraticModel(
        {"e_3": 1.0, "e_1": 2.0}, {("e_3", "e_1"): -1.0}, 0.5, "BINARY"
    )
    stub = Stub()
    returned = SeededSampler(stub, 5).sample(bqm, num_reads=3)
    assert stub.calls == [(["e_1", "e_3"], {"seed": 5, "num_reads": 3})]
    assert returned.get_linear("e_3") == 1.0
    assert returned.get_quadratic("e_1", "e_3") == -1.0
    assert returned.offset == 0.5


def test_canonical_bqm_is_order_independent() -> None:
    a = dimod.BinaryQuadraticModel(
        {"x": 1.0, "y": 2.0}, {("x", "y"): 3.0}, 0.0, "BINARY"
    )
    b = dimod.BinaryQuadraticModel(
        {"y": 2.0, "x": 1.0}, {("y", "x"): 3.0}, 0.0, "BINARY"
    )
    assert (
        list(canonical_bqm(a).variables)
        == list(canonical_bqm(b).variables)
        == ["x", "y"]
    )


def test_recording_solver_keeps_last_solution() -> None:
    class Inner:
        evaluator = "eval"

        def run(self, graph):
            return ("solved", graph)

    graph = Graph()
    recorder = RecordingSolver(Inner())  # type: ignore[arg-type]
    assert recorder.evaluator == "eval"
    assert recorder.run(graph) == ("solved", graph)
    assert recorder.last_solution == ("solved", graph)


def test_build_poly_generators_rejects_empty() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_poly_generators(())


def test_build_solver_without_dnc_wraps_qubo_solver_directly() -> None:
    top, recorder = build_solver((2,), True, seed=1, num_reads=5, use_dnc=False)
    assert isinstance(top, ReductionMr2sSolver) and top.mr2s_solver is recorder
    assert isinstance(recorder.inner, QuboMR2SSolver)
    bare, bare_recorder = build_solver((2,), False, seed=1, num_reads=5, use_dnc=False)
    assert bare is bare_recorder and isinstance(bare_recorder.inner, QuboMR2SSolver)
