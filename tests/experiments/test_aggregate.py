# pandas 3.0 타입 정보 부족으로 인한 pyright 오탐 억제 (experiments/aggregate.py 와 동일).
# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
import json
from pathlib import Path

import pytest

from experiments.aggregate import (
    best_by_graph,
    best_config_counts,
    best_of_k,
    collect_runs,
    estimate_total_hours,
    main,
    make_figures,
    paired_arms,
    paired_reduction,
    summarize_best_of,
    summarize_by_config,
    verify_solutions,
    wilcoxon_arms,
    wilcoxon_hops,
    wilcoxon_reduction,
    write_solutions_jsonl,
)
from experiments.config import RunSpec, iter_run_specs
from experiments.graphs import build_record, graph_path, save_graph
from experiments.solutions import encode_orientation, load_solution
from mr2s_module.solver.predefined import create_robbin_solver


def _synthetic_runs(runs_dir: Path, graph_dir: Path) -> None:
    """작은 그래프 2개 × hop {h2,h3} × 축약 on/off × rep 6 의 합성 결과. 해는 Robbin 배향."""
    runs_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    records = [build_record(14, seed, 0.0) for seed in (0, 1)]
    for record in records:
        save_graph(graph_path(graph_dir, record.graph_id), record)
    solver = create_robbin_solver(use_reduction=False)
    for record in records:
        solution = solver.run(load_solution(record, "0" * record.n_edges).graph)
        bits = encode_orientation(record, solution.edges.values())
        assert solution.score is not None
        for hop_key in ("h2", "h3"):
            for use_reduction in (True, False):
                for rep in range(6):
                    spec = RunSpec(
                        14, record.graph_seed, 0.0, hop_key, use_reduction, rep
                    )
                    bonus = (
                        0.01 * rep
                        + (0.0 if use_reduction else 0.05)
                        + (0.02 if hop_key == "h3" else 0.0)
                    )
                    status = "timeout" if (rep == 5 and hop_key == "h3") else "ok"
                    payload = {
                        "run_id": spec.run_id,
                        "graph_id": spec.graph_id,
                        "vertices": 14,
                        "graph_seed": record.graph_seed,
                        "remove_ratio_target": 0.0,
                        "n_edges": record.n_edges,
                        "hop_key": hop_key,
                        "use_reduction": use_reduction,
                        "rep": rep,
                        "run_seed": spec.run_seed,
                        "status": status,
                        "elapsed_sec": 1.0 + rep + (0.5 if use_reduction else 0.0),
                        "n_subgraphs": 1,
                        "qubo_vars_total": record.n_edges - (3 if use_reduction else 0),
                        "n_edges_solved": record.n_edges,
                        "subgraph_sizes": [record.n_edges],
                    }
                    if status == "ok":
                        payload.update(
                            {
                                "apsp_sum": solution.score.apsp_sum + bonus
                                if rep != 4
                                else None,
                                "strongly_connected": rep != 4,
                                "flow_score": solution.score.flow_score,
                                "orientation_bits": bits,
                                "solution": [list(e) for e in solution.edges.values()],
                            }
                        )
                    (runs_dir / f"{spec.run_id}.json").write_text(json.dumps(payload))


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("agg")
    _synthetic_runs(root / "results" / "runs", root / "graphs")
    return root / "results", root / "graphs"


def test_collect_and_summarize(results: tuple[Path, Path]) -> None:
    df = collect_runs(results[0] / "runs")
    assert len(df) == 2 * 2 * 2 * 6
    assert (
        df["apsp_sum"].isna().sum() == 2 * 2 * 2 * 1 + 2 * 2 * 1
    )  # rep4 비강연결 + h3 timeout
    summary = summarize_by_config(df)
    assert len(summary) == 4
    h3_on = summary[(summary["hop_key"] == "h3") & summary["use_reduction"]].iloc[0]
    assert h3_on["n_runs"] == 12 and h3_on["n_timeout"] == 2 and h3_on["n_sc"] == 8
    assert h3_on["apsp_mean"] > 0 and h3_on["sc_rate"] == pytest.approx(8 / 12)


def test_best_and_paired(results: tuple[Path, Path]) -> None:
    df = collect_runs(results[0] / "runs")
    best_of = summarize_best_of(df)
    assert set(best_of["n_graphs_with_sc"]) == {2}
    best = best_by_graph(df)
    assert len(best) == 2
    assert (best["hop_key"] == "h2").all() and best["use_reduction"].all()
    counts = best_config_counts(df)
    assert counts["n_best"].sum() == 2
    assert set(counts["n_graphs"]) == {2}

    paired = paired_reduction(df)
    assert len(paired) == 2 * 2 * 6 - 2 * 1  # timeout 쌍 제외
    sc_pairs = paired[paired["both_sc"]]
    assert (sc_pairs["diff_apsp"] < 0).all()
    wil = wilcoxon_reduction(paired)
    assert len(wil) == 2
    assert wil["p_apsp"].between(0, 1).all()
    hops = wilcoxon_hops(df)
    assert set(hops["hop_key"]) == {"h3"}
    assert hops["median_diff_apsp"].gt(0).all()


def test_estimate_solutions_verify_and_figures(
    results: tuple[Path, Path], tmp_path: Path
) -> None:
    results_dir, graph_dir = results
    df = collect_runs(results_dir / "runs")
    specs = iter_run_specs(
        vertex_counts=(14, 28),
        graph_seeds=(0,),
        remove_ratios=(0.0,),
        hop_keys=("h2", "h3"),
        reps=1,
    )
    est = estimate_total_hours(df, specs, workers=2)
    assert len(est) == 2 * 2 * 2
    assert (est["total_hours_parallel"] > 0).all()

    n = write_solutions_jsonl(df, tmp_path / "solutions")
    assert n == len(df)
    lines = (tmp_path / "solutions" / "v14_s0_p0.jsonl").read_text().splitlines()
    assert len(lines) == 24 and json.loads(lines[0])["run_id"].startswith("v14_s0_p0")

    # 합성 데이터는 rep 0·h2·축약 on 만 실제 점수를 담고 나머지는 가산값을 더했다.
    exact = df[(df["rep"] == 0) & df["use_reduction"] & (df["hop_key"] == "h2")]
    assert len(exact) == 2 and verify_solutions(exact, graph_dir).empty
    # 나머지 ok 행은 전부 불일치: 가산값 행 + 비강연결로 기록했지만 실제 배향은 강연결인 rep 4 행.
    mismatches = verify_solutions(df, graph_dir)
    assert len(mismatches) == int(df["ok"].sum()) - 2

    figures = make_figures(
        df, summarize_by_config(df), paired_reduction(df), tmp_path / "figs"
    )
    assert len(figures) == 8 and all(
        f.exists() and f.stat().st_size > 0 for f in figures
    )


def test_main_writes_outputs(results: tuple[Path, Path]) -> None:
    results_dir, graph_dir = results
    main(
        [
            "--results",
            str(results_dir),
            "--graph-dir",
            str(graph_dir),
            "--no-figures",
            "--verify",
        ]
    )
    for name in (
        "results.csv",
        "summary_by_config.csv",
        "paired_reduction.csv",
        "wilcoxon_reduction.csv",
        "best_by_graph.csv",
        "summary.md",
    ):
        assert (results_dir / name).exists()
    assert (
        "orientation_bits"
        not in (results_dir / "results.csv").read_text().splitlines()[0]
    )


def test_best_of_k_is_monotone_and_counts_reps(results: tuple[Path, Path]) -> None:
    df = collect_runs(results[0] / "runs")
    bok = best_of_k(df, max_k=6)
    assert set(bok["k"]) == set(range(1, 7))
    h2_on = bok[(bok["hop_key"] == "h2") & bok["use_reduction"]].sort_values("k")
    # 합성 데이터는 rep 0 이 가장 좋으므로 best-of-k 는 k 에 대해 일정, 단조 비증가.
    assert h2_on["best_mean"].is_monotonic_decreasing
    assert h2_on["best_mean"].iloc[0] == pytest.approx(h2_on["best_mean"].iloc[-1])
    assert (h2_on["n_graphs"] == 2).all() and (h2_on["sc_frac"] == 1.0).all()


def test_paired_arms_and_wilcoxon(results: tuple[Path, Path]) -> None:
    df = collect_runs(results[0] / "runs")
    whole = df.copy()
    whole["use_dnc"] = False
    whole["apsp_sum"] = whole["apsp_sum"] - 0.01  # 전체 그래프가 조금 더 좋다고 가정
    whole["elapsed_sec"] = whole["elapsed_sec"] * 0.5
    pair = paired_arms(df, whole)
    assert len(pair) == int(df["ok"].sum())
    assert (pair.loc[pair["both_sc"], "diff_apsp"] > 0).all()
    assert pair["ratio_time"].to_numpy() == pytest.approx(2.0)
    wil = wilcoxon_arms(pair)
    assert len(wil) == 2 and wil["p_apsp"].between(0, 1).all()
