"""단일 실행 워커: 프로세스 1개 = (그래프, hop 조합, 축약, rep) 1회.

결과는 runs/<run_id>.json 에 원자적으로 쓴다(.tmp 후 rename). 해는 방향 비트열과
(tail, head) 목록 두 형식으로 모두 남긴다. 솔버 예외(DnC 분할 실패 등)는 잡아서
status=error 로 기록하고 정상 종료한다 — 드라이버가 재시작 시 건너뛸 수 있도록.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import socket
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

from experiments import config
from experiments.config import RunSpec, parse_run_id
from experiments.graphs import GraphRecord, graph_path, load_graph, to_domain_graph
from experiments.solutions import encode_orientation
from experiments.solvers import SAMPLER_NAME, RecordingSolver, build_solver
from mr2s_module.domain import Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.reduction import contract_chains
from mr2s_module.solver.dnc_mr2s_solver import DnCSolution
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS 는 bytes, Linux 는 KiB 를 돌려준다.
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _is_strongly_connected(solution: Solution) -> bool:
    digraph = nx.DiGraph()
    digraph.add_nodes_from(solution.graph.get_vertices())
    digraph.add_edges_from(solution.edges.values())
    return nx.is_strongly_connected(digraph)


def _whole_graph_metadata(recorder: RecordingSolver) -> dict[str, Any]:
    """DnC 없는 대조군: inner QuboMR2SSolver 가 푼 그래프의 BQM 크기를 다시 계산한다."""
    inner = recorder.inner
    graph = recorder.last_graph
    if graph is None or not isinstance(inner, QuboMR2SSolver):
        return _dnc_metadata(None)
    bqm = inner.build_bqm(graph)
    return {
        "n_subgraphs": 1,
        "partition_target_k": None,
        "subgraph_sizes": [len(graph.edges)],
        "qubo_vars_total": len(bqm.variables),
        "qubo_vars_max": len(bqm.variables),
        "qubo_couplings_total": len(bqm.quadratic),
        "n_edges_solved": len(graph.edges),
    }


def _dnc_metadata(inner: Solution | None) -> dict[str, Any]:
    if not isinstance(inner, DnCSolution):
        return {
            "n_subgraphs": None,
            "partition_target_k": None,
            "subgraph_sizes": None,
            "qubo_vars_total": None,
            "qubo_vars_max": None,
            "qubo_couplings_total": None,
            "n_edges_solved": len(inner.graph.edges) if inner is not None else None,
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
        "n_edges_solved": len(inner.graph.edges),
    }


def base_result(spec: RunSpec, record: GraphRecord, num_reads: int) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "graph_id": spec.graph_id,
        "vertices": spec.vertices,
        "graph_seed": spec.graph_seed,
        "remove_ratio_target": spec.remove_ratio,
        "remove_ratio_actual": record.remove_ratio_actual,
        "n_vertices": record.n_vertices,
        "n_edges": record.n_edges,
        "hop_key": spec.hop_key,
        "hops": "+".join(str(h) for h in spec.hops),
        "use_reduction": spec.use_reduction,
        "use_dnc": spec.use_dnc,
        "rep": spec.rep,
        "run_seed": spec.run_seed,
        "status": None,
        "error_type": None,
        "error_message": None,
        "elapsed_sec": None,
        "timeout_sec": None,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "git_sha": _git_sha(),
        "sampler": SAMPLER_NAME,
        "num_reads": num_reads,
    }


def execute(spec: RunSpec, graph_dir: Path, *, num_reads: int) -> dict[str, Any]:
    np.random.seed(spec.run_seed)
    random.seed(spec.run_seed)

    record = load_graph(graph_path(graph_dir, spec.graph_id))
    graph = to_domain_graph(record)
    result = base_result(spec, record, num_reads)

    contraction_started = perf_counter()
    contraction = contract_chains(graph)
    result.update(
        {
            "contraction_sec": perf_counter() - contraction_started,
            "n_edges_contracted": len(contraction.contracted_graph.edges),
            "n_super_edges": len(contraction.chain_by_super_id),
            "n_cycle_chains": len(contraction.cycle_chains),
        }
    )

    solver, recorder = build_solver(
        spec.hops,
        spec.use_reduction,
        seed=spec.run_seed,
        num_reads=num_reads,
        use_dnc=spec.use_dnc,
    )
    started = perf_counter()
    try:
        solution = solver.run(graph)
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": "".join(traceback.format_exception_only(exc))[:500],
                "elapsed_sec": perf_counter() - started,
                "peak_rss_mb": _peak_rss_mb(),
            }
        )
        result.update(_dnc_metadata(recorder.last_solution))
        return result
    elapsed = perf_counter() - started

    score = solution.score if solution.score is not None else Evaluator().run(solution)
    directed = list(solution.edges.values())
    bits = encode_orientation(record, directed)  # 간선 집합 일치도 여기서 검증된다
    result.update(
        _dnc_metadata(recorder.last_solution)
        if spec.use_dnc
        else _whole_graph_metadata(recorder)
    )
    result.update(
        {
            "status": "ok",
            "elapsed_sec": elapsed,
            "peak_rss_mb": _peak_rss_mb(),
            "apsp_sum": _finite_or_none(score.apsp_sum),
            "strong_connect_rate": score.strong_connect_rate,
            "flow_score": score.flow_score,
            "sample_score": _finite_or_none(score.sample_score),
            "strongly_connected": _is_strongly_connected(solution),
            "orientation_bits": bits,
            "solution": [[tail, head] for tail, head in directed],
        }
    )
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MR2S 실험 단일 실행 워커")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="runs/ 디렉터리")
    parser.add_argument("--num-reads", type=int, default=config.NUM_READS)
    args = parser.parse_args(argv)

    spec = parse_run_id(args.run_id)
    result = execute(spec, args.graph_dir, num_reads=args.num_reads)
    write_result(args.out / f"{spec.run_id}.json", result)
    print(
        f"{spec.run_id}: status={result['status']} elapsed={result['elapsed_sec']:.1f}s "
        f"sc={result.get('strongly_connected')} stretch={result.get('apsp_sum')}"
    )


if __name__ == "__main__":
    main()
