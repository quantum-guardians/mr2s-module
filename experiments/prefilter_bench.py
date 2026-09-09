"""DnC 임베딩 prefilter 지표 비교 벤치 (#94).

`DegeneracyPruningFaceCyclePartitionStrategy` 의 가짜 임베딩 판정(degeneracy <= 타깃
degeneracy)을 treewidth 상계 기준으로 바꾸면 나아지는지 실측한다. 세 서브커맨드:

- ``candidates``: 면 분할로 크기가 다양한 서브그래프 BQM 을 만들고, 상호작용 그래프 지표
  (변수, 커플링, 최대 차수, degeneracy, treewidth 상계)와 minorminer 실측(이상형 Pegasus
  P16)을 기록한다. 산출: ``candidates.csv``.
- ``analyze``: 지표별 AUC 와 임계값 정확도(타깃 유도, 최적, leave-one-graph-out)를 계산하고
  종단 비교용 보정 임계값을 ``thresholds.json`` 에 쓴다. 산출: ``metrics.csv``.
- ``e2e``: prefilter 전략별로 DnC + QUBO-SA 를 종단 실행하고(``experiments.solvers``
  의 구성 그대로), 최종 분할 서브그래프 전부의 실제 임베딩 가능 여부를 minorminer 로
  확인한다. 산출: ``e2e.csv``. ``e2e-one`` 은 드라이버가 띄우는 워커다.

라이브러리는 수정하지 않는다. treewidth 전략은 ``DegeneracyPruningFaceCyclePartitionStrategy``
를 상속해 ``_estimate_degeneracy`` 만 바꾼 하네스 내부 서브클래스이고, 임계값은 부모의
``max_degeneracy`` 로 준다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import traceback
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import dwave_networkx as dnx
import minorminer
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import (
    treewidth_min_degree,
    treewidth_min_fill_in,
)

from experiments import config
from experiments.config import RunSpec, graph_id, parse_graph_id
from experiments.graphs import (
    DEFAULT_GRAPH_DIR,
    graph_path,
    load_graph,
    to_domain_graph,
)
from experiments.solvers import build_qubo_solver, build_solver
from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Graph, Solution
from mr2s_module.evaluator import Evaluator
from mr2s_module.reduction import contract_chains
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver, DnCSolution
from mr2s_module.solver.partition.degeneracy_pruning import (
    DegeneracyPruningFaceCyclePartitionStrategy,
)
from mr2s_module.solver.qubo_mr2s_solver import QuboMR2SSolver

DEFAULT_RESULTS_DIR = Path("experiments/results/prefilter")
PEGASUS_SIZE = 16
PEGASUS_NODES = 5640
PEGASUS_COUPLERS = 40484
# 사전 계산(2026-09-09): P16 의 nx.core_number 최댓값과 treewidth_min_degree 상계(20 초 소요).
PEGASUS_DEGENERACY = 8
PEGASUS_TREEWIDTH_UB = 454
DEFAULT_K_GRID: tuple[int, ...] = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
DEFAULT_VERTICES: tuple[int, ...] = (100, 200, 300, 500)
DEFAULT_SEEDS: tuple[int, ...] = (0,)
DEFAULT_REMOVE_RATIOS: tuple[float, ...] = (0.0, 0.3)
DEFAULT_HOP_KEYS: tuple[str, ...] = ("h2", "h2+3")
RANKS: tuple[str, ...] = ("max", "mid", "min")
METRICS: tuple[str, ...] = (
    "degeneracy",
    "tw_min_degree",
    "tw_min_fill_in",
    "n_vars",
    "n_couplings",
    "max_degree",
)
# 타깃 토폴로지에서 유도되는 임계값. prefilter 가 "지표 <= 임계값" 이면 통과시킨다고 볼 때.
TARGET_THRESHOLDS: dict[str, int] = {
    "degeneracy": PEGASUS_DEGENERACY,
    "tw_min_degree": PEGASUS_TREEWIDTH_UB,
    "tw_min_fill_in": PEGASUS_TREEWIDTH_UB,
    "n_vars": PEGASUS_NODES,
    "n_couplings": PEGASUS_COUPLERS,
    "max_degree": 15,
}


# ---------------------------------------------------------------------------
# 전략
# ---------------------------------------------------------------------------


@dataclass
class TreewidthPruningFaceCyclePartitionStrategy(
    DegeneracyPruningFaceCyclePartitionStrategy,
):
    """degeneracy 대신 treewidth 상계(min_degree 휴리스틱)로 prefilter 하는 실험용 전략.

    부모의 ``max_degeneracy`` 가 treewidth 임계값 역할을 한다. None 이면 부모가 타깃
    그래프에 같은 함수를 적용해 상한을 구하므로(P16 은 20 초) 항상 명시하는 편이 낫다.
    """

    @staticmethod
    def _estimate_degeneracy(graph: nx.Graph) -> int:
        return treewidth_upper_bound(graph)


@dataclass
class CouplingsPruningFaceCyclePartitionStrategy(
    DegeneracyPruningFaceCyclePartitionStrategy,
):
    """상호작용 그래프의 커플링(2차 항) 수로 prefilter 하는 대조용 전략.

    후보 실측에서 커플링 수가 가장 판별력이 높게 나와 종단 비교에 넣는다. 부모의
    ``max_degeneracy`` 가 커플링 수 임계값이다.
    """

    @staticmethod
    def _estimate_degeneracy(graph: nx.Graph) -> int:
        return graph.number_of_edges()


ARM_STRATEGIES: dict[str, type[DegeneracyPruningFaceCyclePartitionStrategy]] = {
    "degeneracy": DegeneracyPruningFaceCyclePartitionStrategy,
    "treewidth": TreewidthPruningFaceCyclePartitionStrategy,
    "couplings": CouplingsPruningFaceCyclePartitionStrategy,
}


@dataclass(frozen=True)
class Arm:
    metric: str  # degeneracy | treewidth | couplings
    threshold: int

    @property
    def name(self) -> str:
        return f"{self.metric}{self.threshold}"


def parse_arm(text: str) -> Arm:
    """``degeneracy:8`` / ``treewidth:454`` / ``couplings:5194`` 형식."""
    metric, _, threshold = text.partition(":")
    if metric not in ARM_STRATEGIES or not threshold.isdigit():
        raise ValueError(f"invalid arm: {text!r} (expected metric:threshold)")
    return Arm(metric=metric, threshold=int(threshold))


def make_strategy(
    arm: Arm, dnc: DnCMr2sSolver
) -> DegeneracyPruningFaceCyclePartitionStrategy:
    qubo = dnc.mr2s_solver
    if not isinstance(qubo, QuboMR2SSolver):
        raise TypeError("prefilter arms require a QuboMR2SSolver inside DnC")
    strategy_type = ARM_STRATEGIES[arm.metric]
    return strategy_type(
        mr2s_solver=qubo,
        face_cycle=dnc.face_cycle,
        target_graph=None,
        max_degeneracy=arm.threshold,
    )


# ---------------------------------------------------------------------------
# 지표와 실측
# ---------------------------------------------------------------------------


def treewidth_upper_bound(graph: nx.Graph) -> int:
    if graph.number_of_edges() == 0:
        return 0
    width, _ = treewidth_min_degree(graph)
    return int(width)


def degeneracy(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    return max(nx.core_number(graph).values(), default=0)


def interaction_graph(bqm: Any) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(bqm.variables)
    graph.add_edges_from(bqm.quadratic)
    return graph


def interaction_metrics(graph: nx.Graph, *, fill_in_max_vars: int) -> dict[str, Any]:
    n_vars = graph.number_of_nodes()
    started = perf_counter()
    degeneracy_value = degeneracy(graph)
    degeneracy_sec = perf_counter() - started

    started = perf_counter()
    tw_min_degree_value = treewidth_upper_bound(graph)
    tw_min_degree_sec = perf_counter() - started

    tw_min_fill_in_value: int | None = None
    tw_min_fill_in_sec: float | None = None
    if 0 < n_vars <= fill_in_max_vars and graph.number_of_edges() > 0:
        started = perf_counter()
        width, _ = treewidth_min_fill_in(graph)
        tw_min_fill_in_value = int(width)
        tw_min_fill_in_sec = perf_counter() - started

    return {
        "n_vars": n_vars,
        "n_couplings": graph.number_of_edges(),
        "max_degree": max((int(d) for _, d in graph.degree()), default=0),
        "degeneracy": degeneracy_value,
        "degeneracy_sec": degeneracy_sec,
        "tw_min_degree": tw_min_degree_value,
        "tw_min_degree_sec": tw_min_degree_sec,
        "tw_min_fill_in": tw_min_fill_in_value,
        "tw_min_fill_in_sec": tw_min_fill_in_sec,
    }


_PEGASUS_EDGES: list[tuple[Any, Any]] | None = None


def pegasus_edges() -> list[tuple[Any, Any]]:
    global _PEGASUS_EDGES
    if _PEGASUS_EDGES is None:
        target = cast(nx.Graph, dnx.pegasus_graph(PEGASUS_SIZE))
        _PEGASUS_EDGES = list(target.edges())
    return _PEGASUS_EDGES


def embed_in_pegasus(
    graph: nx.Graph,
    *,
    timeout: int,
    threads: int,
    seed: int,
    retries: int,
) -> dict[str, Any]:
    """minorminer 로 P16 임베딩을 찾는다. 시간 제한 실패는 임베딩 불가로 본다(#93 기준).

    실패하면 seed 를 바꿔 ``retries`` 번 더 시도해 탐색 실패를 일부 걸러낸다.
    """
    started = perf_counter()
    n_vars = graph.number_of_nodes()
    if n_vars == 0:
        return _embed_record(True, 0, 0, started, 0)
    if graph.number_of_edges() == 0:
        embeddable = n_vars <= PEGASUS_NODES
        return _embed_record(embeddable, n_vars, 1, started, 0)

    source_edges = list(graph.edges())
    attempts = 0
    embedding: dict[Any, list[Any]] = {}
    while attempts <= retries and not embedding:
        embedding = minorminer.find_embedding(
            source_edges,
            pegasus_edges(),
            timeout=timeout,
            threads=threads,
            random_seed=seed + attempts,
        )
        attempts += 1
    if not embedding:
        return _embed_record(False, None, None, started, attempts)
    return _embed_record(
        True,
        sum(len(chain) for chain in embedding.values()),
        max(len(chain) for chain in embedding.values()),
        started,
        attempts,
    )


def _embed_record(
    embeddable: bool,
    physical_qubits: int | None,
    max_chain_length: int | None,
    started: float,
    attempts: int,
) -> dict[str, Any]:
    return {
        "embeddable": embeddable,
        "physical_qubits": physical_qubits,
        "max_chain_length": max_chain_length,
        "embed_sec": perf_counter() - started,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateJob:
    candidate_id: str
    graph_id: str
    vertices: int
    use_reduction: bool
    hop_key: str
    target_k: int | None
    rank: str
    n_vertices: int
    n_edges: int  # 무방향 간선 = QUBO 변수가 되는 간선
    n_directed: int
    variables: tuple[Any, ...]
    couplings: tuple[tuple[Any, Any], ...]

    @property
    def meta(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("variables")
        record.pop("couplings")
        return record


@dataclass(frozen=True)
class CandidateOptions:
    graph_dir: Path
    k_grid: tuple[int, ...]
    per_k: int
    partition_seed: int
    fill_in_max_vars: int
    embed_timeout: int
    embed_threads: int
    embed_seed: int
    embed_retries: int
    # 이미 실측된 후보 중 임베딩 실패로 기록된 것만 더 긴 예산으로 다시 실측한다.
    recheck_failed: bool = False
    recheck_max_couplings: int | None = None


def _undirected_ids(graph: Graph) -> tuple[int, ...]:
    return tuple(sorted(edge.id for edge in graph.edges.values() if not edge.directed))


def generate_candidates(
    graph_id_value: str,
    hop_key: str,
    use_reduction: bool,
    opts: CandidateOptions,
) -> list[CandidateJob]:
    """면 분할 target_k 격자를 돌며 크기가 다른 서브그래프 후보를 뽑는다.

    k 마다 무방향 간선 수 기준 최대/중앙/최소(``per_k`` 개)를 고르고, 전체 그래프도 하나의
    후보(``whole``)로 넣는다. 같은 무방향 간선 집합은 한 번만 센다.
    """
    vertices, _, _ = parse_graph_id(graph_id_value)
    record = load_graph(graph_path(opts.graph_dir, graph_id_value))
    graph = to_domain_graph(record)
    if use_reduction:
        graph = contract_chains(graph).contracted_graph
    qubo = build_qubo_solver(config.hops_of(hop_key), seed=0, num_reads=1)
    reduction_tag = "red" if use_reduction else "nored"

    jobs: list[CandidateJob] = []
    seen: set[tuple[int, ...]] = set()

    def add(sub_graph: Graph, target_k: int | None, rank: str) -> None:
        signature = _undirected_ids(sub_graph)
        if not signature or signature in seen:
            return
        seen.add(signature)
        bqm = qubo.build_bqm(sub_graph)
        k_tag = "whole" if target_k is None else f"k{target_k}"
        jobs.append(
            CandidateJob(
                candidate_id=(
                    f"{graph_id_value}__{hop_key}__{reduction_tag}__{k_tag}__{rank}"
                ),
                graph_id=graph_id_value,
                vertices=vertices,
                use_reduction=use_reduction,
                hop_key=hop_key,
                target_k=target_k,
                rank=rank,
                n_vertices=len(sub_graph.get_vertices()),
                n_edges=len(signature),
                n_directed=sum(1 for e in sub_graph.edges.values() if e.directed),
                variables=tuple(bqm.variables),
                couplings=tuple((u, v) for u, v in bqm.quadratic),
            )
        )

    add(graph, None, "whole")
    for target_k in opts.k_grid:
        np.random.seed(opts.partition_seed)
        partition = FaceClusterPartition(
            target_k=target_k, clusterer=KMeansFaceClusterer()
        ).run(graph)
        sub_graphs = [sg for sg in partition.sub_graphs if _undirected_ids(sg)]
        if not sub_graphs:
            continue
        sub_graphs.sort(key=lambda sg: len(_undirected_ids(sg)))
        picks = {
            "max": sub_graphs[-1],
            "mid": sub_graphs[len(sub_graphs) // 2],
            "min": sub_graphs[0],
        }
        for rank in RANKS[: opts.per_k]:
            add(picks[rank], target_k, rank)
    return jobs


def evaluate_candidate(job: CandidateJob, opts: CandidateOptions) -> dict[str, Any]:
    graph = nx.Graph()
    graph.add_nodes_from(job.variables)
    graph.add_edges_from(job.couplings)
    result = job.meta
    result.update(interaction_metrics(graph, fill_in_max_vars=opts.fill_in_max_vars))
    result.update(
        embed_in_pegasus(
            graph,
            timeout=opts.embed_timeout,
            threads=opts.embed_threads,
            seed=opts.embed_seed,
            retries=opts.embed_retries,
        )
    )
    result["timeout_sec"] = opts.embed_timeout
    return result


def _generate_task(
    args: tuple[str, str, bool, CandidateOptions],
) -> list[CandidateJob]:
    try:
        return generate_candidates(*args)
    except Exception as exc:
        graph_id_value, hop_key, use_reduction, _ = args
        print(
            f"candidate generation failed graph={graph_id_value} hops={hop_key} "
            f"reduction={use_reduction}: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        return []


def _evaluate_task(args: tuple[CandidateJob, CandidateOptions]) -> dict[str, Any]:
    return evaluate_candidate(*args)


def iter_graph_ids(
    vertex_counts: Iterable[int],
    seeds: Iterable[int],
    remove_ratios: Iterable[float],
) -> list[str]:
    return [
        graph_id(vertices, seed, ratio)
        for vertices in vertex_counts
        for seed in seeds
        for ratio in remove_ratios
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_jsons(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


CANDIDATE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "graph_id",
    "vertices",
    "use_reduction",
    "hop_key",
    "target_k",
    "rank",
    "n_vertices",
    "n_edges",
    "n_directed",
    "n_vars",
    "n_couplings",
    "max_degree",
    "degeneracy",
    "degeneracy_sec",
    "tw_min_degree",
    "tw_min_degree_sec",
    "tw_min_fill_in",
    "tw_min_fill_in_sec",
    "embeddable",
    "physical_qubits",
    "max_chain_length",
    "embed_sec",
    "attempts",
    "timeout_sec",
    "rechecked",
    "first_timeout_sec",
)


def _select_pending(
    jobs: list[CandidateJob], runs_dir: Path, opts: CandidateOptions
) -> tuple[list[CandidateJob], dict[str, dict[str, Any]]]:
    """실측할 후보와, 재실측 시 덮어쓸 이전 기록을 고른다."""
    existing = {path.stem: path for path in runs_dir.glob("*.json")}
    if not opts.recheck_failed:
        return [job for job in jobs if job.candidate_id not in existing], {}
    previous = {
        stem: json.loads(path.read_text(encoding="utf-8"))
        for stem, path in existing.items()
    }
    pending = [
        job
        for job in jobs
        if previous.get(job.candidate_id, {}).get("embeddable") is False
        and (
            opts.recheck_max_couplings is None
            or len(job.couplings) <= opts.recheck_max_couplings
        )
    ]
    return pending, previous


def run_candidates(args: argparse.Namespace) -> None:
    opts = CandidateOptions(
        graph_dir=args.graph_dir,
        k_grid=tuple(args.k_grid),
        per_k=args.per_k,
        partition_seed=args.partition_seed,
        fill_in_max_vars=args.fill_in_max_vars,
        embed_timeout=args.timeout,
        embed_threads=args.threads,
        embed_seed=args.embed_seed,
        embed_retries=args.retries,
        recheck_failed=args.recheck_failed,
        recheck_max_couplings=args.recheck_max_couplings,
    )
    runs_dir = args.results / "runs" / "candidates"
    runs_dir.mkdir(parents=True, exist_ok=True)
    combos = [
        (gid, hop_key, use_reduction, opts)
        for gid in iter_graph_ids(args.vertices, args.seeds, args.remove_ratios)
        for hop_key in args.hops
        for use_reduction in (False, True)
    ]
    context = get_context("spawn")
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        jobs = [job for jobs in pool.map(_generate_task, combos) for job in jobs]
        pending, previous = _select_pending(jobs, runs_dir, opts)
        print(
            f"candidates total={len(jobs)} pending={len(pending)} "
            f"recheck={opts.recheck_failed} "
            f"generated_in={perf_counter() - started:.0f}s workers={args.workers}",
            flush=True,
        )
        # 큰 후보(시간 제한까지 갈 수 있음)를 먼저 넣어 꼬리 지연을 줄인다.
        pending.sort(key=lambda job: -len(job.couplings))
        futures = {pool.submit(_evaluate_task, (job, opts)): job for job in pending}
        for finished, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = job.meta
                result["error"] = "".join(traceback.format_exception_only(exc))[:500]
            if opts.recheck_failed:
                result["rechecked"] = True
                result["first_timeout_sec"] = previous.get(job.candidate_id, {}).get(
                    "timeout_sec"
                )
            write_json(runs_dir / f"{job.candidate_id}.json", result)
            print(
                f"[{finished}/{len(pending)}] {job.candidate_id} "
                f"vars={result.get('n_vars')} embeddable={result.get('embeddable')} "
                f"embed={result.get('embed_sec', 0.0):.1f}s "
                f"elapsed={(perf_counter() - started) / 60:.1f}m",
                flush=True,
            )
    rows = [row for row in load_jsons(runs_dir) if "embeddable" in row]
    rows.sort(key=lambda row: str(row["candidate_id"]))
    write_csv(args.results / "candidates.csv", rows, CANDIDATE_COLUMNS)
    print(f"wrote {args.results / 'candidates.csv'} rows={len(rows)}", flush=True)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def auc_reject_score(values: Sequence[float], embeddable: Sequence[bool]) -> float:
    """지표가 클수록 '임베딩 불가' 쪽이라고 볼 때의 AUC (Mann-Whitney, 동점 0.5).

    1.0 이면 모든 불가 후보의 지표가 모든 가능 후보보다 크다. 0.5 는 무정보.
    """
    positives = [v for v, ok in zip(values, embeddable, strict=True) if not ok]
    negatives = [v for v, ok in zip(values, embeddable, strict=True) if ok]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def threshold_stats(
    values: Sequence[float], embeddable: Sequence[bool], threshold: float
) -> dict[str, Any]:
    """규칙 '지표 <= threshold 면 통과' 의 정확도. FP = 통과했지만 불가, FN = 거절했지만 가능."""
    accepted = [v <= threshold for v in values]
    fp = sum(1 for a, ok in zip(accepted, embeddable, strict=True) if a and not ok)
    fn = sum(1 for a, ok in zip(accepted, embeddable, strict=True) if not a and ok)
    total = len(values)
    return {
        "threshold": threshold,
        "accuracy": (total - fp - fn) / total if total else float("nan"),
        "fp": fp,
        "fn": fn,
        "accepted": sum(accepted),
    }


def best_threshold(
    values: Sequence[float], embeddable: Sequence[bool]
) -> dict[str, Any]:
    """정확도 최대 임계값. 동률이면 FP 가 적은(더 엄격한) 쪽, 그다음 큰 임계값."""
    candidates = sorted(set(values))
    if not candidates:
        return threshold_stats(values, embeddable, float("nan"))
    candidates.insert(0, candidates[0] - 1)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        stats = threshold_stats(values, embeddable, threshold)
        if best is None or (stats["accuracy"], -stats["fp"], threshold) > (
            best["accuracy"],
            -best["fp"],
            best["threshold"],
        ):
            best = stats
    assert best is not None
    return best


def leave_one_group_out_accuracy(
    values: Sequence[float], embeddable: Sequence[bool], groups: Sequence[str]
) -> float:
    """그룹(그래프) 하나를 빼고 고른 임계값을 그 그룹에 적용한 정확도의 가중 평균."""
    correct = 0
    total = 0
    for held_out in sorted(set(groups)):
        train = [i for i, g in enumerate(groups) if g != held_out]
        test = [i for i, g in enumerate(groups) if g == held_out]
        if not train:
            continue
        threshold = best_threshold(
            [values[i] for i in train], [embeddable[i] for i in train]
        )["threshold"]
        stats = threshold_stats(
            [values[i] for i in test], [embeddable[i] for i in test], threshold
        )
        correct += len(test) - stats["fp"] - stats["fn"]
        total += len(test)
    return correct / total if total else float("nan")


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def load_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["embeddable"] = _parse_bool(row["embeddable"])
        row["use_reduction"] = _parse_bool(row["use_reduction"])
        for metric in METRICS:
            raw = row.get(metric)
            row[metric] = float(raw) if raw not in ("", None) else None
    return rows


METRIC_COLUMNS: tuple[str, ...] = (
    "stratum",
    "metric",
    "n",
    "n_embeddable",
    "auc",
    "target_threshold",
    "target_accuracy",
    "target_fp",
    "target_fn",
    "best_threshold",
    "best_accuracy",
    "best_fp",
    "best_fn",
    "loo_accuracy",
)


def analyze_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        strata.setdefault(str(row["hop_key"]), []).append(row)
    output: list[dict[str, Any]] = []
    for stratum, subset in strata.items():
        for metric in METRICS:
            usable = [row for row in subset if row.get(metric) is not None]
            if not usable:
                continue
            values = [float(row[metric]) for row in usable]
            labels = [bool(row["embeddable"]) for row in usable]
            groups = [str(row["graph_id"]) for row in usable]
            target = threshold_stats(values, labels, TARGET_THRESHOLDS[metric])
            best = best_threshold(values, labels)
            output.append(
                {
                    "stratum": stratum,
                    "metric": metric,
                    "n": len(usable),
                    "n_embeddable": sum(labels),
                    "auc": auc_reject_score(values, labels),
                    "target_threshold": target["threshold"],
                    "target_accuracy": target["accuracy"],
                    "target_fp": target["fp"],
                    "target_fn": target["fn"],
                    "best_threshold": best["threshold"],
                    "best_accuracy": best["accuracy"],
                    "best_fp": best["fp"],
                    "best_fn": best["fn"],
                    "loo_accuracy": leave_one_group_out_accuracy(
                        values, labels, groups
                    ),
                }
            )
    return output


def format_metrics_table(metrics: list[dict[str, Any]]) -> str:
    header = (
        "| stratum | metric | n | embeddable | AUC | target thr | acc (FP/FN) | "
        "best thr | acc (FP/FN) | LOO acc |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|"]
    for row in metrics:
        lines.append(
            f"| {row['stratum']} | {row['metric']} | {row['n']} | {row['n_embeddable']} "
            f"| {row['auc']:.3f} | {row['target_threshold']:g} "
            f"| {row['target_accuracy']:.3f} ({row['target_fp']}/{row['target_fn']}) "
            f"| {row['best_threshold']:g} "
            f"| {row['best_accuracy']:.3f} ({row['best_fp']}/{row['best_fn']}) "
            f"| {row['loo_accuracy']:.3f} |"
        )
    return "\n".join(lines)


def run_analyze(args: argparse.Namespace) -> None:
    rows = load_candidates(args.results / "candidates.csv")
    metrics = analyze_rows(rows)
    write_csv(args.results / "metrics.csv", metrics, METRIC_COLUMNS)
    thresholds: dict[str, dict[str, float]] = {}
    for row in metrics:
        if row["metric"] in ("degeneracy", "tw_min_degree"):
            key = "treewidth" if row["metric"] == "tw_min_degree" else "degeneracy"
            thresholds.setdefault(key, {})[row["stratum"]] = row["best_threshold"]
    (args.results / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8"
    )
    print(format_metrics_table(metrics))
    print(f"\nthresholds: {json.dumps(thresholds)}", flush=True)


# ---------------------------------------------------------------------------
# e2e
# ---------------------------------------------------------------------------


E2E_COLUMNS: tuple[str, ...] = (
    "run_id",
    "graph_id",
    "vertices",
    "remove_ratio",
    "hop_key",
    "use_reduction",
    "arm",
    "metric",
    "threshold",
    "run_seed",
    "status",
    "error",
    "elapsed_sec",
    "n_subgraphs",
    "partition_target_k",
    "qubo_vars_total",
    "qubo_vars_max",
    "qubo_couplings_total",
    "apsp_sum",
    "strong_connect_rate",
    "flow_score",
    "strongly_connected",
    "n_subgraphs_checked",
    "n_subgraphs_embeddable",
    "partition_embeddable",
    "physical_qubits_total",
    "max_chain_length",
    "verify_sec",
)


def e2e_run_id(graph_id_value: str, hop_key: str, use_reduction: bool, arm: Arm) -> str:
    reduction_tag = "red" if use_reduction else "nored"
    return f"{graph_id_value}__{hop_key}__{reduction_tag}__{arm.name}"


def is_strongly_connected(solution: Solution) -> bool:
    digraph = nx.DiGraph()
    digraph.add_nodes_from(solution.graph.get_vertices())
    digraph.add_edges_from(solution.edges.values())
    return nx.is_strongly_connected(digraph)


def dnc_metadata(inner: Solution | None) -> dict[str, Any]:
    if not isinstance(inner, DnCSolution):
        return {
            "n_subgraphs": None,
            "partition_target_k": None,
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
        "qubo_vars_total": sum(var_counts) if var_counts else None,
        "qubo_vars_max": max(var_counts) if var_counts else None,
        "qubo_couplings_total": sum(len(bqm.quadratic) for bqm in bqms)
        if bqms
        else None,
    }


def verify_partition(
    inner: Solution | None, *, timeout: int, threads: int, seed: int, retries: int
) -> dict[str, Any]:
    """최종 분할의 서브그래프 BQM 전부를 minorminer 로 실측한다."""
    started = perf_counter()
    if not isinstance(inner, DnCSolution):
        return {
            "n_subgraphs_checked": 0,
            "n_subgraphs_embeddable": 0,
            "partition_embeddable": None,
            "physical_qubits_total": None,
            "max_chain_length": None,
            "verify_sec": 0.0,
        }
    checked = 0
    embeddable = 0
    qubits = 0
    max_chain = 0
    for context in inner.solve_contexts:
        bqm = getattr(context, "bqm", None)
        if bqm is None:
            continue
        checked += 1
        record = embed_in_pegasus(
            interaction_graph(bqm),
            timeout=timeout,
            threads=threads,
            seed=seed,
            retries=retries,
        )
        if record["embeddable"]:
            embeddable += 1
            qubits += record["physical_qubits"] or 0
            max_chain = max(max_chain, record["max_chain_length"] or 0)
    return {
        "n_subgraphs_checked": checked,
        "n_subgraphs_embeddable": embeddable,
        "partition_embeddable": checked > 0 and checked == embeddable,
        "physical_qubits_total": qubits,
        "max_chain_length": max_chain,
        "verify_sec": perf_counter() - started,
    }


def e2e_one(
    graph_id_value: str,
    hop_key: str,
    use_reduction: bool,
    arm: Arm,
    *,
    graph_dir: Path,
    num_reads: int,
    embed_timeout: int,
    embed_threads: int,
    embed_retries: int,
) -> dict[str, Any]:
    vertices, seed, ratio = parse_graph_id(graph_id_value)
    spec = RunSpec(
        vertices=vertices,
        graph_seed=seed,
        remove_ratio=ratio,
        hop_key=hop_key,
        use_reduction=use_reduction,
        rep=0,
    )
    result: dict[str, Any] = {
        "run_id": e2e_run_id(graph_id_value, hop_key, use_reduction, arm),
        "graph_id": graph_id_value,
        "vertices": vertices,
        "remove_ratio": ratio,
        "hop_key": hop_key,
        "use_reduction": use_reduction,
        "arm": arm.name,
        "metric": arm.metric,
        "threshold": arm.threshold,
        "run_seed": spec.run_seed,
        "status": "ok",
        "error": None,
    }
    np.random.seed(spec.run_seed)
    random.seed(spec.run_seed)
    record = load_graph(graph_path(graph_dir, graph_id_value))
    graph = to_domain_graph(record)
    solver, recorder = build_solver(
        spec.hops, use_reduction, seed=spec.run_seed, num_reads=num_reads
    )
    dnc = recorder.inner
    if not isinstance(dnc, DnCMr2sSolver):
        raise TypeError("build_solver did not return a DnC solver")
    dnc.graph_partition_strategy = make_strategy(arm, dnc)

    started = perf_counter()
    try:
        solution = solver.run(graph)
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": "".join(traceback.format_exception_only(exc))[:500],
                "elapsed_sec": perf_counter() - started,
            }
        )
        result.update(dnc_metadata(recorder.last_solution))
        return result
    result["elapsed_sec"] = perf_counter() - started

    score = solution.score if solution.score is not None else Evaluator().run(solution)
    result.update(dnc_metadata(recorder.last_solution))
    result.update(
        {
            "apsp_sum": score.apsp_sum,
            "strong_connect_rate": score.strong_connect_rate,
            "flow_score": score.flow_score,
            "strongly_connected": is_strongly_connected(solution),
        }
    )
    result.update(
        verify_partition(
            recorder.last_solution,
            timeout=embed_timeout,
            threads=embed_threads,
            seed=0,
            retries=embed_retries,
        )
    )
    return result


def run_e2e_one(args: argparse.Namespace) -> None:
    arm = parse_arm(args.arm)
    result = e2e_one(
        args.graph_id,
        args.hops,
        args.reduction,
        arm,
        graph_dir=args.graph_dir,
        num_reads=args.num_reads,
        embed_timeout=args.timeout,
        embed_threads=args.threads,
        embed_retries=args.retries,
    )
    write_json(args.out, result)
    print(
        f"{result['run_id']}: status={result['status']} "
        f"elapsed={result.get('elapsed_sec', 0.0):.1f}s "
        f"subgraphs={result.get('n_subgraphs')} vars_max={result.get('qubo_vars_max')} "
        f"apsp={result.get('apsp_sum')} embeddable={result.get('partition_embeddable')}",
        flush=True,
    )


@dataclass(frozen=True)
class E2ESpec:
    graph_id: str
    hop_key: str
    use_reduction: bool
    arm: Arm

    @property
    def run_id(self) -> str:
        return e2e_run_id(self.graph_id, self.hop_key, self.use_reduction, self.arm)


def _e2e_worker_command(
    spec: E2ESpec, args: argparse.Namespace, out: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.prefilter_bench",
        "e2e-one",
        "--graph-id",
        spec.graph_id,
        "--hops",
        spec.hop_key,
        "--reduction" if spec.use_reduction else "--no-reduction",
        "--arm",
        f"{spec.arm.metric}:{spec.arm.threshold}",
        "--graph-dir",
        str(args.graph_dir),
        "--num-reads",
        str(args.num_reads),
        "--timeout",
        str(args.timeout),
        "--threads",
        str(args.threads),
        "--retries",
        str(args.retries),
        "--out",
        str(out),
    ]


def _run_e2e_spec(spec: E2ESpec, args: argparse.Namespace, runs_dir: Path) -> str:
    out = runs_dir / f"{spec.run_id}.json"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    started = perf_counter()
    process = subprocess.Popen(
        _e2e_worker_command(spec, args, out),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, stderr = process.communicate(timeout=args.run_timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        write_json(
            out,
            _e2e_failure(spec, "timeout", f"killed after {args.run_timeout}s", started),
        )
        return "timeout"
    if process.returncode != 0:
        write_json(
            out,
            _e2e_failure(
                spec, "error", f"worker exit {process.returncode}: {stderr}", started
            ),
        )
        return "error"
    try:
        return str(json.loads(out.read_text(encoding="utf-8")).get("status"))
    except (OSError, ValueError):
        return "missing"


def _e2e_failure(
    spec: E2ESpec, status: str, message: str, started: float
) -> dict[str, Any]:
    vertices, _, ratio = parse_graph_id(spec.graph_id)
    return {
        "run_id": spec.run_id,
        "graph_id": spec.graph_id,
        "vertices": vertices,
        "remove_ratio": ratio,
        "hop_key": spec.hop_key,
        "use_reduction": spec.use_reduction,
        "arm": spec.arm.name,
        "metric": spec.arm.metric,
        "threshold": spec.arm.threshold,
        "status": status,
        "error": message[-500:],
        "elapsed_sec": perf_counter() - started,
    }


def run_e2e(args: argparse.Namespace) -> None:
    arms = [parse_arm(text) for text in args.arms]
    runs_dir = args.results / "runs" / "e2e"
    runs_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        E2ESpec(gid, hop_key, use_reduction, arm)
        for gid in iter_graph_ids(args.vertices, args.seeds, args.remove_ratios)
        for hop_key in args.hops
        for use_reduction in (False, True)
        for arm in arms
    ]
    done = {path.stem for path in runs_dir.glob("*.json")}
    pending = [spec for spec in specs if spec.run_id not in done]
    print(
        f"e2e total={len(specs)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_e2e_spec, spec, args, runs_dir): spec for spec in pending
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            print(
                f"[{finished}/{len(pending)}] {spec.run_id} {future.result()} "
                f"elapsed={(perf_counter() - started) / 60:.1f}m",
                flush=True,
            )
    rows = load_jsons(runs_dir)
    rows.sort(key=lambda row: str(row["run_id"]))
    write_csv(args.results / "e2e.csv", rows, E2E_COLUMNS)
    print(f"wrote {args.results / 'e2e.csv'} rows={len(rows)}", flush=True)


# ---------------------------------------------------------------------------
# summarize (e2e)
# ---------------------------------------------------------------------------


E2E_NUMERIC: tuple[str, ...] = (
    "elapsed_sec",
    "n_subgraphs",
    "qubo_vars_max",
    "qubo_vars_total",
    "apsp_sum",
    "strong_connect_rate",
    "verify_sec",
    "physical_qubits_total",
    "max_chain_length",
)
E2E_SUMMARY_COLUMNS: tuple[str, ...] = (
    "hop_key",
    "use_reduction",
    "arm",
    "n",
    "n_ok",
    "n_subgraphs",
    "qubo_vars_max",
    "elapsed_sec",
    "apsp_sum",
    "strongly_connected_rate",
    "partition_embeddable_rate",
    "verify_sec",
    "n_paired",
    "d_apsp_vs_baseline",
    "time_ratio_vs_baseline",
)


def load_e2e(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["use_reduction"] = _parse_bool(row["use_reduction"])
        row["strongly_connected"] = _parse_bool(row["strongly_connected"])
        row["partition_embeddable"] = _parse_bool(row["partition_embeddable"])
        for column in E2E_NUMERIC:
            raw = row.get(column)
            row[column] = float(raw) if raw not in ("", None) else None
    return rows


def _mean(values: Iterable[float | None]) -> float:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else float("nan")


def summarize_e2e(
    rows: list[dict[str, Any]], baseline_arm: str
) -> list[dict[str, Any]]:
    """(hop, 축약, 전략)별 평균과, 같은 그래프의 baseline 전략 대비 짝 비교."""
    ok_rows = [row for row in rows if row["status"] == "ok"]
    baseline = {
        (row["graph_id"], row["hop_key"], row["use_reduction"]): row
        for row in ok_rows
        if row["arm"] == baseline_arm
    }
    groups: dict[tuple[str, bool, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["hop_key"]), bool(row["use_reduction"]), str(row["arm"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (hop_key, use_reduction, arm), members in sorted(groups.items()):
        ok = [row for row in members if row["status"] == "ok"]
        paired = [
            (row, baseline[(row["graph_id"], hop_key, use_reduction)])
            for row in ok
            if (row["graph_id"], hop_key, use_reduction) in baseline
        ]
        output.append(
            {
                "hop_key": hop_key,
                "use_reduction": use_reduction,
                "arm": arm,
                "n": len(members),
                "n_ok": len(ok),
                "n_subgraphs": _mean(row["n_subgraphs"] for row in ok),
                "qubo_vars_max": _mean(row["qubo_vars_max"] for row in ok),
                "elapsed_sec": _mean(row["elapsed_sec"] for row in ok),
                "apsp_sum": _mean(row["apsp_sum"] for row in ok),
                "strongly_connected_rate": _mean(
                    float(row["strongly_connected"]) for row in ok
                ),
                "partition_embeddable_rate": _mean(
                    float(row["partition_embeddable"]) for row in ok
                ),
                "verify_sec": _mean(row["verify_sec"] for row in ok),
                "n_paired": len(paired),
                "d_apsp_vs_baseline": _mean(
                    row["apsp_sum"] - base["apsp_sum"]
                    for row, base in paired
                    if row["apsp_sum"] is not None and base["apsp_sum"] is not None
                ),
                "time_ratio_vs_baseline": _mean(
                    row["elapsed_sec"] / base["elapsed_sec"]
                    for row, base in paired
                    if row["elapsed_sec"] and base["elapsed_sec"]
                ),
            }
        )
    return output


def format_e2e_table(summary: list[dict[str, Any]]) -> str:
    header = (
        "| hop | red | arm | ok/n | subgraphs | vars max | elapsed s | apsp | "
        "SC | embeddable | verify s | d apsp | time ratio |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary:
        lines.append(
            f"| {row['hop_key']} | {row['use_reduction']} | {row['arm']} "
            f"| {row['n_ok']}/{row['n']} | {row['n_subgraphs']:.1f} "
            f"| {row['qubo_vars_max']:.0f} | {row['elapsed_sec']:.0f} "
            f"| {row['apsp_sum']:.4f} | {row['strongly_connected_rate']:.2f} "
            f"| {row['partition_embeddable_rate']:.2f} | {row['verify_sec']:.0f} "
            f"| {row['d_apsp_vs_baseline']:+.4f} | {row['time_ratio_vs_baseline']:.2f} |"
        )
    return "\n".join(lines)


def run_summarize(args: argparse.Namespace) -> None:
    rows = load_e2e(args.results / "e2e.csv")
    summary = summarize_e2e(rows, args.baseline_arm)
    write_csv(args.results / "e2e_summary.csv", summary, E2E_SUMMARY_COLUMNS)
    print(format_e2e_table(summary), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def _parse_strs(text: str) -> list[str]:
    return [part for part in text.split(",") if part]


def _add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--vertices", type=_parse_ints, default=list(DEFAULT_VERTICES))
    parser.add_argument("--seeds", type=_parse_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--remove-ratios", type=_parse_floats, default=list(DEFAULT_REMOVE_RATIOS)
    )
    parser.add_argument("--hops", type=_parse_strs, default=list(DEFAULT_HOP_KEYS))


def _add_embed_arguments(parser: argparse.ArgumentParser, *, threads: int) -> None:
    parser.add_argument("--timeout", type=int, default=60, help="minorminer 초")
    parser.add_argument("--threads", type=int, default=threads)
    parser.add_argument("--retries", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DnC 임베딩 prefilter 지표 비교 (#94)")
    sub = parser.add_subparsers(dest="command", required=True)

    candidates = sub.add_parser(
        "candidates", help="후보 서브그래프 지표 + minorminer 실측"
    )
    _add_matrix_arguments(candidates)
    _add_embed_arguments(candidates, threads=2)
    candidates.add_argument("--k-grid", type=_parse_ints, default=list(DEFAULT_K_GRID))
    candidates.add_argument("--per-k", type=int, default=3, choices=(1, 2, 3))
    candidates.add_argument("--partition-seed", type=int, default=0)
    candidates.add_argument("--fill-in-max-vars", type=int, default=600)
    candidates.add_argument("--embed-seed", type=int, default=0)
    candidates.add_argument("--workers", type=int, default=8)
    candidates.add_argument(
        "--recheck-failed",
        action="store_true",
        help="임베딩 실패로 기록된 후보만 현재 --timeout 으로 다시 실측해 덮어쓴다",
    )
    candidates.add_argument(
        "--recheck-max-couplings",
        type=int,
        default=None,
        help="재실측 대상 커플링 수 상한 (더 큰 후보는 실패 확정으로 둔다)",
    )
    candidates.set_defaults(func=run_candidates)

    analyze = sub.add_parser("analyze", help="지표별 AUC 와 임계값 정확도")
    analyze.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    analyze.set_defaults(func=run_analyze)

    summarize = sub.add_parser("summarize", help="e2e.csv 를 전략별로 집계")
    summarize.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    summarize.add_argument("--baseline-arm", default="degeneracy8")
    summarize.set_defaults(func=run_summarize)

    e2e = sub.add_parser("e2e", help="prefilter 전략별 DnC + QUBO-SA 종단 비교")
    _add_matrix_arguments(e2e)
    _add_embed_arguments(e2e, threads=1)
    e2e.add_argument(
        "--arms",
        type=_parse_strs,
        default=["degeneracy:8", "treewidth:454"],
        help="metric:threshold 목록 (예: degeneracy:8,treewidth:454,treewidth:130)",
    )
    e2e.add_argument("--num-reads", type=int, default=config.NUM_READS)
    e2e.add_argument("--workers", type=int, default=4)
    e2e.add_argument("--run-timeout", type=int, default=3600, help="실행 1회 초")
    e2e.set_defaults(func=run_e2e)

    one = sub.add_parser("e2e-one", help="e2e 워커 (드라이버가 호출)")
    one.add_argument("--graph-id", required=True)
    one.add_argument("--hops", required=True, help="hop_key (h2, h2+3, ...)")
    one.add_argument("--reduction", dest="reduction", action="store_true")
    one.add_argument("--no-reduction", dest="reduction", action="store_false")
    one.set_defaults(reduction=True)
    one.add_argument("--arm", required=True)
    one.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    one.add_argument("--num-reads", type=int, default=config.NUM_READS)
    _add_embed_arguments(one, threads=1)
    one.add_argument("--out", type=Path, required=True)
    one.set_defaults(func=run_e2e_one)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
