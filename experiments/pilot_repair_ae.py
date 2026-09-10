"""경계 봉합 규칙 파일럿: A(legacy) 대 E(interior_merge) 를 같은 그래프·같은 seed 로 짝지어 비교.

본 실험 하네스(run_one/run_all)의 run_id 체계는 건드리지 않고, 고정 그래프 인스턴스
(experiments/data/graphs) 위에서 두 변형을 나란히 돌려 stretch·강연결·부분 그래프 수·
QUBO 변수 수·시간을 JSONL 로 남긴다. 결과 파일이 있으면 이미 끝난 실행은 건너뛴다(재개).

사용법은 docs/PILOT_REPAIR_AE.md 참고. 요약만 다시 보려면 `--summary` 를 준다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

from experiments import config
from experiments.config import RunSpec
from experiments.graphs import (
    DEFAULT_GRAPH_DIR,
    graph_path,
    load_graph,
    to_domain_graph,
)
from experiments.solvers import REPAIR_OPTIONS, build_solver
from mr2s_module.domain import Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.solver.dnc_mr2s_solver import DnCSolution

VARIANTS = ("legacy", "interior_merge")
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "pilot_repair_ae"


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
    return out


def _is_strongly_connected(solution: Solution) -> bool:
    digraph = nx.DiGraph()
    digraph.add_nodes_from(solution.graph.get_vertices())
    digraph.add_edges_from(solution.edges.values())
    return nx.is_strongly_connected(digraph)


def _dnc_metadata(inner: Solution | None) -> dict[str, Any]:
    if not isinstance(inner, DnCSolution):
        return {
            "n_subgraphs": None,
            "partition_target_k": None,
            "subgraph_sizes": None,
            "qubo_vars_total": None,
            "qubo_vars_max": None,
            "qubo_couplings_total": None,
        }
    bqms = [
        context.bqm
        for context in inner.solve_contexts
        if context is not None and getattr(context, "bqm", None) is not None
    ]
    var_counts = [len(bqm.variables) for bqm in bqms]
    return {
        "n_subgraphs": len(inner.sub_graphs),
        "partition_target_k": inner.partition_target_k,
        "subgraph_sizes": [len(sub.edges) for sub in inner.sub_graphs],
        "qubo_vars_total": sum(var_counts) if var_counts else None,
        "qubo_vars_max": max(var_counts) if var_counts else None,
        "qubo_couplings_total": sum(len(bqm.quadratic) for bqm in bqms)
        if bqms
        else None,
    }


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def run_key(spec: RunSpec, variant: str) -> str:
    return f"{spec.run_id}__{variant}"


def execute_one(
    spec: RunSpec, variant: str, graph_dir: Path, *, num_reads: int
) -> dict[str, Any]:
    np.random.seed(spec.run_seed)
    random.seed(spec.run_seed)
    record = load_graph(graph_path(graph_dir, spec.graph_id))
    graph = to_domain_graph(record)
    row: dict[str, Any] = {
        "key": run_key(spec, variant),
        "run_id": spec.run_id,
        "variant": variant,
        "graph_id": spec.graph_id,
        "vertices": spec.vertices,
        "graph_seed": spec.graph_seed,
        "remove_ratio": spec.remove_ratio,
        "n_vertices": record.n_vertices,
        "n_edges": record.n_edges,
        "hop_key": spec.hop_key,
        "use_reduction": spec.use_reduction,
        "rep": spec.rep,
        "run_seed": spec.run_seed,
        "num_reads": num_reads,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    solver, recorder = build_solver(
        spec.hops,
        spec.use_reduction,
        seed=spec.run_seed,
        num_reads=num_reads,
        repair=variant,
    )
    started = perf_counter()
    try:
        solution = solver.run(graph)
    except Exception as exc:  # 분할 실패 등은 기록하고 계속
        row.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": "".join(traceback.format_exception_only(exc))[:500],
                "elapsed_sec": perf_counter() - started,
            }
        )
        row.update(_dnc_metadata(recorder.last_solution))
        return row
    elapsed = perf_counter() - started
    score = solution.score if solution.score is not None else Evaluator().run(solution)
    row.update(_dnc_metadata(recorder.last_solution))
    row.update(
        {
            "status": "ok",
            "elapsed_sec": elapsed,
            "stretch": _finite_or_none(score.apsp_sum),
            "strong_connect_rate": score.strong_connect_rate,
            "flow_score": score.flow_score,
            "strongly_connected": _is_strongly_connected(solution),
            "solution": [[tail, head] for tail, head in solution.edges.values()],
        }
    )
    return row


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["key"]] = row
    return rows


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for vertices in args.vertices:
        for seed in args.seeds:
            for remove in args.remove:
                for hop_key in args.hops:
                    for rep in range(args.reps):
                        specs.append(
                            RunSpec(
                                vertices=vertices,
                                graph_seed=seed,
                                remove_ratio=remove / 100.0,
                                hop_key=hop_key,
                                use_reduction=not args.no_reduction,
                                rep=rep,
                            )
                        )
    return specs


# ------------------------------------------------------------------ 요약


def _wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    if len(a) < 6:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    if all(d == 0 for d in diffs):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def summarize(rows: dict[str, dict[str, Any]]) -> str:
    by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows.values():
        by_run[row["run_id"]][row["variant"]] = row
    groups: dict[tuple[int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for variants in by_run.values():
        if all(v in variants for v in VARIANTS):
            a, e = variants["legacy"], variants["interior_merge"]
            groups[(a["vertices"], a["hop_key"])].append((a, e))

    lines = [
        "| 정점 | hop | 쌍 | 오류 A/E | 강연결 A/E | stretch A | stretch E | Δ(E−A) | E 우세/동률/열세 | p(Wilcoxon) | 부분그래프 A/E | QUBO 최대변수 A/E | 시간 A/E (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (vertices, hop_key), pairs in sorted(groups.items()):
        n = len(pairs)
        err_a = sum(1 for a, _ in pairs if a["status"] != "ok")
        err_e = sum(1 for _, e in pairs if e["status"] != "ok")
        ok_pairs = [
            (a, e) for a, e in pairs if a["status"] == "ok" and e["status"] == "ok"
        ]
        sc_a = sum(1 for a, _ in ok_pairs if a["strongly_connected"])
        sc_e = sum(1 for _, e in ok_pairs if e["strongly_connected"])
        both = [
            (a, e)
            for a, e in ok_pairs
            if a["strongly_connected"]
            and e["strongly_connected"]
            and a["stretch"] is not None
            and e["stretch"] is not None
        ]
        sa = [a["stretch"] for a, _ in both]
        se = [e["stretch"] for _, e in both]
        mean_a = st.mean(sa) if sa else float("nan")
        mean_e = st.mean(se) if se else float("nan")
        wins = sum(1 for x, y in zip(sa, se, strict=True) if y < x - 1e-9)
        ties = sum(1 for x, y in zip(sa, se, strict=True) if abs(y - x) <= 1e-9)
        losses = len(both) - wins - ties
        p = _wilcoxon_p(sa, se)

        def m(key: str, which: int, pairs_ok=ok_pairs) -> str:
            vals = [
                pair[which][key]
                for pair in pairs_ok
                if pair[which].get(key) is not None
            ]
            return f"{st.mean(vals):.1f}" if vals else "-"

        lines.append(
            f"| {vertices} | {hop_key} | {n} | {err_a}/{err_e} | {sc_a}/{sc_e} "
            f"| {mean_a:.4f} | {mean_e:.4f} | {mean_e - mean_a:+.4f} | {wins}/{ties}/{losses} "
            f"| {'-' if p is None else f'{p:.3g}'} "
            f"| {m('n_subgraphs', 0)}/{m('n_subgraphs', 1)} "
            f"| {m('qubo_vars_max', 0)}/{m('qubo_vars_max', 1)} "
            f"| {m('elapsed_sec', 0)}/{m('elapsed_sec', 1)} |"
        )
    lines.append("")
    lines.append(
        "stretch 는 두 변형 모두 강연결인 쌍에서만 평균했다(낮을수록 좋음). "
        "Δ<0 이면 E 가 stretch 를 줄인 것이다. p 는 scipy 가 있을 때만 계산한다."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="경계 봉합 A/E 파일럿")
    parser.add_argument("--vertices", nargs="+", default=["100", "200"])
    parser.add_argument("--seeds", nargs="+", default=["0-9"])
    parser.add_argument(
        "--remove", nargs="+", default=["0", "30"], help="간선 제거 비율(%%)"
    )
    parser.add_argument(
        "--hops",
        nargs="+",
        default=["h2", "h3"],
        help=f"hop key: {', '.join(config.HOP_SETS)}",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--num-reads", type=int, default=config.NUM_READS)
    parser.add_argument(
        "--no-reduction", action="store_true", help="체인 축약 presolve 끄기"
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(REPAIR_OPTIONS)
    )
    parser.add_argument(
        "--summary", action="store_true", help="실행 없이 기존 결과만 요약"
    )
    args = parser.parse_args(argv)
    args.vertices = _parse_int_list(args.vertices)
    args.seeds = _parse_int_list(args.seeds)
    args.remove = _parse_int_list(args.remove)

    runs_path = args.out / "runs.jsonl"
    summary_path = args.out / "summary.md"
    done = load_done(runs_path)

    if not args.summary:
        specs = build_specs(args)
        todo = [
            (spec, variant)
            for spec in specs
            for variant in args.variants
            if run_key(spec, variant) not in done
        ]
        print(
            f"specs={len(specs)} variants={args.variants} todo={len(todo)} done={len(done)}",
            flush=True,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        for i, (spec, variant) in enumerate(todo, start=1):
            row = execute_one(spec, variant, args.graph_dir, num_reads=args.num_reads)
            with runs_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
                )
            done[row["key"]] = row
            print(
                f"[{i}/{len(todo)}] {row['key']}: status={row['status']} "
                f"elapsed={row['elapsed_sec']:.1f}s sc={row.get('strongly_connected')} "
                f"stretch={row.get('stretch')} subgraphs={row.get('n_subgraphs')}",
                flush=True,
            )

    summary = summarize(done)
    print(summary)
    if done:
        args.out.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary + "\n", encoding="utf-8")
        print(f"summary written: {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
