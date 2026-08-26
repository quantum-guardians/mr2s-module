"""실험 드라이버: 매트릭스를 순회하며 실행마다 워커 프로세스를 띄운다.

- 완료 판정 = runs/<run_id>.json 존재. --retry 로 지정한 status 의 파일은 완료로 보지 않는다.
- 타임아웃 시 프로세스 그룹을 통째로 죽이고 status=timeout 레코드를 드라이버가 쓴다.
- 워커가 비정상 종료하면 status=error 레코드를 드라이버가 쓴다 (워커 내부 예외는 워커가 기록).
- 실행 단위 병렬(스레드 × subprocess). DnC 내부 병렬은 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from experiments import config
from experiments.config import RunSpec, iter_run_specs
from experiments.graphs import DEFAULT_GRAPH_DIR
from experiments.run_one import write_result


@dataclass(frozen=True)
class DriverOptions:
    graph_dir: Path
    results_dir: Path
    workers: int
    num_reads: int
    timeout_by_vertices: dict[int, int]
    retry_statuses: frozenset[str] = frozenset()
    dry_run: bool = False
    python: str = field(default=sys.executable)

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    def timeout_for(self, vertices: int) -> int:
        if vertices in self.timeout_by_vertices:
            return self.timeout_by_vertices[vertices]
        return max(self.timeout_by_vertices.values())


def load_done(runs_dir: Path, retry_statuses: Iterable[str] = ()) -> set[str]:
    retry = set(retry_statuses)
    done: set[str] = set()
    for path in runs_dir.glob("*.json"):
        if retry:
            try:
                status = json.loads(path.read_text()).get("status")
            except (OSError, ValueError):
                continue
            if status in retry:
                continue
        done.add(path.stem)
    return done


def worker_command(spec: RunSpec, opts: DriverOptions) -> list[str]:
    return [
        opts.python,
        "-m",
        "experiments.run_one",
        "--run-id",
        spec.run_id,
        "--graph-dir",
        str(opts.graph_dir),
        "--out",
        str(opts.runs_dir),
        "--num-reads",
        str(opts.num_reads),
    ]


def _failure_record(
    spec: RunSpec, status: str, message: str, elapsed: float, timeout: int
) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "graph_id": spec.graph_id,
        "vertices": spec.vertices,
        "graph_seed": spec.graph_seed,
        "remove_ratio_target": spec.remove_ratio,
        "hop_key": spec.hop_key,
        "hops": "+".join(str(h) for h in spec.hops),
        "use_reduction": spec.use_reduction,
        "rep": spec.rep,
        "run_seed": spec.run_seed,
        "status": status,
        "error_type": status,
        "error_message": message[-500:],
        "elapsed_sec": elapsed,
        "timeout_sec": timeout,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_spec(spec: RunSpec, opts: DriverOptions) -> str:
    """워커를 실행하고 최종 status 를 돌려준다."""
    timeout = opts.timeout_for(spec.vertices)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    started = perf_counter()
    process = subprocess.Popen(
        worker_command(spec, opts),
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        write_result(
            opts.runs_dir / f"{spec.run_id}.json",
            _failure_record(
                spec,
                "timeout",
                f"killed after {timeout}s",
                perf_counter() - started,
                timeout,
            ),
        )
        return "timeout"

    if process.returncode != 0:
        write_result(
            opts.runs_dir / f"{spec.run_id}.json",
            _failure_record(
                spec,
                "error",
                f"worker exit {process.returncode}: {stderr}",
                perf_counter() - started,
                timeout,
            ),
        )
        return "error"

    result_path = opts.runs_dir / f"{spec.run_id}.json"
    try:
        return str(json.loads(result_path.read_text()).get("status"))
    except (OSError, ValueError):
        return "missing"


def run_matrix(specs: list[RunSpec], opts: DriverOptions) -> dict[str, int]:
    opts.runs_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(opts.runs_dir, opts.retry_statuses)
    pending = [spec for spec in specs if spec.run_id not in done]
    counts: dict[str, int] = {"skipped": len(specs) - len(pending)}
    print(
        f"total={len(specs)} pending={len(pending)} skipped={counts['skipped']} "
        f"workers={opts.workers}",
        flush=True,
    )
    if opts.dry_run or not pending:
        return counts

    started = perf_counter()
    finished = 0
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures = {pool.submit(run_spec, spec, opts): spec for spec in pending}
        for future in as_completed(futures):
            spec = futures[future]
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
            finished += 1
            elapsed = perf_counter() - started
            eta = elapsed / finished * (len(pending) - finished)
            print(
                f"[{finished}/{len(pending)}] {spec.run_id} {status} "
                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                flush=True,
            )
    return counts


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def _parse_strs(text: str) -> list[str]:
    return [part for part in text.split(",") if part]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MR2S 실험 매트릭스 드라이버")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--num-reads", type=int, default=config.NUM_READS)
    parser.add_argument(
        "--vertices", type=_parse_ints, default=list(config.VERTEX_COUNTS)
    )
    parser.add_argument("--seeds", type=_parse_ints, default=list(config.GRAPH_SEEDS))
    parser.add_argument(
        "--remove-ratios", type=_parse_floats, default=list(config.REMOVE_RATIOS)
    )
    parser.add_argument("--hops", type=_parse_strs, default=list(config.HOP_SETS))
    parser.add_argument("--reps", type=int, default=config.REPS)
    parser.add_argument(
        "--hop4-max-vertices", type=int, default=config.HOP4_MAX_VERTICES
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="정점 수별 기본표 대신 단일 값 사용",
    )
    parser.add_argument("--retry", type=_parse_strs, default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    timeout_table: dict[int, int] = dict(config.TIMEOUT_SEC_BY_VERTICES)
    if args.timeout_sec is not None:
        timeout_table = dict.fromkeys(timeout_table, int(args.timeout_sec))
    opts = DriverOptions(
        graph_dir=args.graph_dir,
        results_dir=args.results,
        workers=args.workers,
        num_reads=args.num_reads,
        timeout_by_vertices=timeout_table,
        retry_statuses=frozenset(args.retry),
        dry_run=args.dry_run,
    )
    specs = iter_run_specs(
        vertex_counts=args.vertices,
        graph_seeds=args.seeds,
        remove_ratios=args.remove_ratios,
        hop_keys=args.hops,
        reps=args.reps,
        hop4_max_vertices=args.hop4_max_vertices,
    )
    counts = run_matrix(specs, opts)
    print(f"done: {counts}", flush=True)


if __name__ == "__main__":
    main()
