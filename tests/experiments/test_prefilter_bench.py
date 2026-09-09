import networkx as nx
import pytest

from experiments.graphs import build_record
from experiments.prefilter_bench import (
    Arm,
    CandidateJob,
    CandidateOptions,
    CouplingsPruningFaceCyclePartitionStrategy,
    TreewidthPruningFaceCyclePartitionStrategy,
    _select_pending,
    auc_reject_score,
    best_threshold,
    degeneracy,
    e2e_run_id,
    generate_candidates,
    interaction_metrics,
    leave_one_group_out_accuracy,
    parse_arm,
    summarize_e2e,
    threshold_stats,
    treewidth_upper_bound,
)
from mr2s_module.solver.partition.degeneracy_pruning import (
    DegeneracyPruningFaceCyclePartitionStrategy,
)


def test_treewidth_strategy_overrides_metric_only() -> None:
    k5 = nx.complete_graph(5)
    path = nx.path_graph(6)
    assert DegeneracyPruningFaceCyclePartitionStrategy._estimate_degeneracy(k5) == 4
    assert TreewidthPruningFaceCyclePartitionStrategy._estimate_degeneracy(k5) == 4
    assert TreewidthPruningFaceCyclePartitionStrategy._estimate_degeneracy(path) == 1
    # subdivided K5: degeneracy 2, treewidth bound stays 4
    subdivided = nx.Graph()
    for u, v in k5.edges():
        mid = (u, v)
        subdivided.add_edge(u, mid)
        subdivided.add_edge(mid, v)
    assert degeneracy(subdivided) == 2
    assert treewidth_upper_bound(subdivided) == 4


def test_interaction_metrics_on_cycle() -> None:
    cycle = nx.cycle_graph(6)
    metrics = interaction_metrics(cycle, fill_in_max_vars=100)
    assert metrics["n_vars"] == 6
    assert metrics["n_couplings"] == 6
    assert metrics["max_degree"] == 2
    assert metrics["degeneracy"] == 2
    assert metrics["tw_min_degree"] == 2
    assert metrics["tw_min_fill_in"] == 2
    skipped = interaction_metrics(cycle, fill_in_max_vars=3)
    assert skipped["tw_min_fill_in"] is None
    empty = interaction_metrics(nx.Graph(), fill_in_max_vars=100)
    assert empty["degeneracy"] == 0 and empty["tw_min_degree"] == 0


def test_threshold_helpers() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    embeddable = [True, True, True, False, False]
    assert auc_reject_score(values, embeddable) == 1.0
    assert threshold_stats(values, embeddable, 3.0) == {
        "threshold": 3.0,
        "accuracy": 1.0,
        "fp": 0,
        "fn": 0,
        "accepted": 3,
    }
    assert best_threshold(values, embeddable)["threshold"] == 3.0
    loose = threshold_stats(values, embeddable, 4.0)
    assert loose["fp"] == 1 and loose["fn"] == 0
    strict = threshold_stats(values, embeddable, 2.0)
    assert strict["fp"] == 0 and strict["fn"] == 1
    groups = ["a", "a", "b", "b", "b"]
    assert 0.0 <= leave_one_group_out_accuracy(values, embeddable, groups) <= 1.0


def test_auc_is_half_without_information() -> None:
    assert auc_reject_score([1.0, 1.0, 1.0, 1.0], [True, False, True, False]) == 0.5


def test_parse_arm_and_run_id() -> None:
    arm = parse_arm("treewidth:130")
    assert arm == Arm(metric="treewidth", threshold=130)
    assert arm.name == "treewidth130"
    assert e2e_run_id("v100_s0_p0", "h2+3", True, arm) == (
        "v100_s0_p0__h2+3__red__treewidth130"
    )
    assert parse_arm("couplings:5194").metric == "couplings"
    assert (
        CouplingsPruningFaceCyclePartitionStrategy._estimate_degeneracy(
            nx.complete_graph(5)
        )
        == 10
    )
    with pytest.raises(ValueError):
        parse_arm("maxdegree:15")


def _job(candidate_id: str, n_couplings: int) -> CandidateJob:
    return CandidateJob(
        candidate_id=candidate_id,
        graph_id="v100_s0_p0",
        vertices=100,
        use_reduction=False,
        hop_key="h2",
        target_k=None,
        rank="whole",
        n_vertices=3,
        n_edges=3,
        n_directed=0,
        variables=("a", "b", "c"),
        couplings=tuple(("a", f"x{i}") for i in range(n_couplings)),
    )


def test_select_pending_rechecks_only_failed_small_candidates(tmp_path) -> None:
    jobs = [
        _job("new", 1),
        _job("failed_small", 2),
        _job("failed_big", 9),
        _job("ok", 1),
    ]
    (tmp_path / "failed_small.json").write_text('{"embeddable": false}')
    (tmp_path / "failed_big.json").write_text('{"embeddable": false}')
    (tmp_path / "ok.json").write_text('{"embeddable": true, "timeout_sec": 60}')
    base = {
        "graph_dir": tmp_path,
        "k_grid": (2,),
        "per_k": 1,
        "partition_seed": 0,
        "fill_in_max_vars": 0,
        "embed_timeout": 1,
        "embed_threads": 1,
        "embed_seed": 0,
        "embed_retries": 0,
    }
    pending, previous = _select_pending(jobs, tmp_path, CandidateOptions(**base))
    assert [job.candidate_id for job in pending] == ["new"]
    assert previous == {}
    pending, previous = _select_pending(
        jobs,
        tmp_path,
        CandidateOptions(**base, recheck_failed=True, recheck_max_couplings=5),
    )
    assert [job.candidate_id for job in pending] == ["failed_small"]
    assert previous["ok"]["timeout_sec"] == 60


def test_summarize_e2e_pairs_against_baseline() -> None:
    def row(arm: str, apsp: float, elapsed: float, embeddable: bool) -> dict:
        return {
            "graph_id": "v100_s0_p0",
            "hop_key": "h2",
            "use_reduction": False,
            "arm": arm,
            "status": "ok",
            "n_subgraphs": 1.0,
            "qubo_vars_max": 10.0,
            "elapsed_sec": elapsed,
            "apsp_sum": apsp,
            "strongly_connected": True,
            "partition_embeddable": embeddable,
            "verify_sec": 1.0,
        }

    rows = [
        row("degeneracy8", 1.30, 10.0, True),
        row("treewidth454", 1.35, 5.0, False),
    ]
    summary = {item["arm"]: item for item in summarize_e2e(rows, "degeneracy8")}
    assert summary["degeneracy8"]["d_apsp_vs_baseline"] == 0.0
    assert summary["treewidth454"]["d_apsp_vs_baseline"] == pytest.approx(0.05)
    assert summary["treewidth454"]["time_ratio_vs_baseline"] == 0.5
    assert summary["treewidth454"]["partition_embeddable_rate"] == 0.0
    assert summary["degeneracy8"]["n_paired"] == 1


def test_generate_candidates_dedupes_and_includes_whole(tmp_path) -> None:
    from experiments.graphs import save_graph

    record = build_record(30, 0, 0.0)
    save_graph(tmp_path / f"{record.graph_id}.json", record)
    opts = CandidateOptions(
        graph_dir=tmp_path,
        k_grid=(2, 3),
        per_k=2,
        partition_seed=0,
        fill_in_max_vars=0,
        embed_timeout=1,
        embed_threads=1,
        embed_seed=0,
        embed_retries=0,
    )
    jobs = generate_candidates(record.graph_id, "h2", False, opts)
    assert jobs[0].rank == "whole"
    assert jobs[0].target_k is None
    assert jobs[0].n_edges == record.n_edges
    assert len(jobs[0].variables) == record.n_edges
    ids = [job.candidate_id for job in jobs]
    assert len(ids) == len(set(ids))
    assert all(job.n_edges > 0 for job in jobs)
    assert any(job.target_k == 2 for job in jobs)
