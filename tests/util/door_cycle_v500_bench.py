"""V=500 독립 검증: 무작위 door 사이클 반전이 face-cycle 기준선보다 나은가.

V=200 스캔에서 얻은 관찰(좋은 해가 door 사이클 반전 근처에 뭉친다)이 우연인지
확인한다. 체리피킹을 피하려고 **사이클공간에서 feasible 원소 하나를 무작위로**
뽑아 쓴다(최고해를 고르지 않는다).

팔 3개 (모두 같은 DnC(max_vertices=40)+SA 솔버):
  FC     : FaceClusterPartition — 프로덕션 기준선(door+외곽 전부 directed)
  DO-can : door-only, 2색칠 기본 방향 그대로
  DO-rnd : door-only, 무작위 feasible 사이클 반전 적용

    PYTHONHASHSEED=0 python tests/util/door_cycle_v500_bench.py [seeds]

출력: tests/util/door_cycle_v500.json
실험 전용 — 프로덕션(mr2s_module/)에 두지 않는다.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Edge, Graph
from mr2s_module.evaluator import Evaluator
from mr2s_module.reduction import ReductionMr2sSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver
from tests.util import door_combo_scan as scan
from tests.util.fixed_door_partition import FixedDoorFaceClusterPartition

N_VERTICES = 500
REMOVE_RATIO = 0.4
TARGET_K = 10
MAX_VERTICES = 40
N_PROCS = 12
SEEDS = list(range(10))
OUT = "tests/util/door_cycle_v500.json"


def pick_random_cycle_combo(case: dict, seed: int) -> tuple[int, dict]:
    """사이클공간 전수 → feasible 중 하나를 무작위 선택 (bits=0 제외)."""
    basis = scan.cycle_space_basis(case)
    feasibility = scan.Feasibility(case)
    feasible: list[int] = []
    for mask in range(1 << len(basis)):
        bits = 0
        for i, cycle in enumerate(basis):
            if (mask >> i) & 1:
                bits ^= cycle
        if bits and feasibility.first_failure(bits) < 0:
            feasible.append(bits)
    stats = {
        "cycle_dim": len(basis),
        "cycle_total": 1 << len(basis),
        "cycle_feasible": len(feasible),
        "doors": len(case["door_ids"]),
        "macros": len(case["macros"]),
    }
    if not feasible:
        return 0, stats
    return random.Random(seed).choice(sorted(feasible)), stats


_WORKER: dict = {}


def _init(cases: dict) -> None:
    _WORKER["cases"] = cases


def build_solver(face_cycle_cls, seed: int, cap: int) -> ReductionMr2sSolver:
    sa = SAMR2SSolver(
        sweeps_per_temperature=2,
        num_restarts=4,
        random_seed=seed,
    )
    face_cycle = face_cycle_cls(target_k=2, clusterer=KMeansFaceClusterer())
    dnc = DnCMr2sSolver(
        mr2s_solver=sa,
        face_cycle=face_cycle,
        graph_partition_strategy=VertexCountPartitionStrategy(
            face_cycle=face_cycle,
            max_vertices=cap,
        ),
        subgraph_processes=1,
    )
    return ReductionMr2sSolver(mr2s_solver=dnc, evaluator=Evaluator())


def run_arm(args: tuple[int, str, int]) -> dict:
    """(seed, arm, bits) → 점수. arm: 'FC' | 'DO-can' | 'DO-rnd'"""
    seed, arm, bits = args
    case = _WORKER["cases"][seed]

    started = time.perf_counter()
    error = ""
    for cap in (MAX_VERTICES, 60, 80):
        # 조합마다 전역 Edge id 카운터를 같은 값으로 리셋 (재현성).
        Edge._id_counter = itertools.count(
            max(eid for eid, _u, _v in case["edges"]) + 1000
        )
        graph = scan.rebuild_graph(case)
        if arm != "FC":
            for eid in case["door_ids"]:
                tail, head = case["canon"][eid]
                if (bits >> case["bit_of"][eid]) & 1:
                    tail, head = head, tail
                graph.edges[eid].set_direction(tail, head)
        cls = FaceClusterPartition if arm == "FC" else FixedDoorFaceClusterPartition
        np.random.seed(seed)
        random.seed(seed)
        try:
            solution = build_solver(cls, seed, cap).run(graph)
        except RuntimeError as exc:
            error = str(exc)
            continue
        score = solution.score
        return {
            "seed": seed,
            "arm": arm,
            "bits": str(bits),
            "cap": cap,
            "failed": False,
            "apsp_sum": float(score.apsp_sum),
            "flow_score": float(score.flow_score),
            "strong_connect_rate": float(score.strong_connect_rate),
            "sec": time.perf_counter() - started,
        }
    return {
        "seed": seed,
        "arm": arm,
        "bits": str(bits),
        "failed": True,
        "error": error,
        "sec": time.perf_counter() - started,
    }


def main() -> None:
    seeds = SEEDS
    if len(sys.argv) > 1:
        seeds = [int(x) for x in sys.argv[1].split(",")]

    scan.N_VERTICES = N_VERTICES
    scan.REMOVE_RATIO = REMOVE_RATIO
    scan.TARGET_K = TARGET_K

    cases: dict[int, dict] = {}
    chosen: dict[int, int] = {}
    stats: dict[int, dict] = {}
    tasks: list[tuple[int, str, int]] = []
    for seed in seeds:
        case = scan.build_case(seed)
        bits, stat = pick_random_cycle_combo(case, seed)
        cases[seed] = case
        chosen[seed] = bits
        stats[seed] = stat
        flipped = sum(
            1 for eid in case["door_ids"] if (bits >> case["bit_of"][eid]) & 1
        )
        stat["flipped"] = flipped
        print(
            f"seed {seed}: macros={stat['macros']} doors={stat['doors']} "
            f"cycle_dim={stat['cycle_dim']} feasible={stat['cycle_feasible']}"
            f"/{stat['cycle_total']} → 무작위 선택 조합의 반전 door {flipped}개",
            flush=True,
        )
        tasks += [(seed, "FC", 0), (seed, "DO-can", 0), (seed, "DO-rnd", bits)]

    print(f"\n실행 {len(tasks)}회 (procs {N_PROCS})", flush=True)
    rows: list[dict] = []
    started = time.perf_counter()
    with Pool(N_PROCS, initializer=_init, initargs=(cases,)) as pool:
        for row in pool.imap_unordered(run_arm, tasks, 1):
            rows.append(row)
            done = len(rows)
            rate = done / max(1e-9, time.perf_counter() - started)
            print(
                f"  {done}/{len(tasks)} seed {row['seed']} {row['arm']:<6} "
                + ("실패" if row["failed"] else
                   f"stretch={row['apsp_sum']:.4f} sc={row['strong_connect_rate']:.2f}")
                + f" ({row['sec']:.0f}s, eta {(len(tasks)-done)/max(rate,1e-9)/60:.1f}m)",
                flush=True,
            )

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({
            "config": {
                "n_vertices": N_VERTICES,
                "remove_ratio": REMOVE_RATIO,
                "target_k": TARGET_K,
                "max_vertices": MAX_VERTICES,
                "seeds": seeds,
            },
            "stats": {str(k): v for k, v in stats.items()},
            "rows": rows,
        }, handle)
    print(f"\nwrote {OUT}", flush=True)

    by = {(r["seed"], r["arm"]): r for r in rows}
    ok_seeds = [
        s for s in seeds
        if all(
            not by[(s, a)]["failed"] and by[(s, a)]["strong_connect_rate"] >= 1.0
            for a in ("FC", "DO-can", "DO-rnd")
        )
    ]
    print(f"\n=== 요약 (3팔 모두 SC 성공한 seed {len(ok_seeds)}/{len(seeds)}개) ===")
    print(f"{'seed':>4}  {'FC':>9}  {'DO-can':>9}  {'DO-rnd':>9}  {'rnd vs FC':>10}")
    for s in ok_seeds:
        fc = by[(s, "FC")]["apsp_sum"]
        dc = by[(s, "DO-can")]["apsp_sum"]
        dr = by[(s, "DO-rnd")]["apsp_sum"]
        print(f"{s:>4}  {fc:>9.4f}  {dc:>9.4f}  {dr:>9.4f}  "
              f"{100 * (dr - fc) / fc:>+9.2f}%")
    if ok_seeds:
        def avg(arm: str) -> float:
            return sum(by[(s, arm)]["apsp_sum"] for s in ok_seeds) / len(ok_seeds)
        fc_a, dc_a, dr_a = avg("FC"), avg("DO-can"), avg("DO-rnd")
        wins = sum(
            1 for s in ok_seeds
            if by[(s, "DO-rnd")]["apsp_sum"] < by[(s, "FC")]["apsp_sum"]
        )
        print(f"\n평균  FC {fc_a:.4f}  DO-can {dc_a:.4f}  DO-rnd {dr_a:.4f}")
        print(f"DO-rnd 가 FC 보다 나은 seed: {wins}/{len(ok_seeds)} "
              f"(평균 {100 * (dr_a - fc_a) / fc_a:+.2f}%)")
    for arm in ("FC", "DO-can", "DO-rnd"):
        sc_ok = sum(
            1 for s in seeds
            if not by[(s, arm)]["failed"] and by[(s, arm)]["strong_connect_rate"] >= 1.0
        )
        print(f"  {arm:<6} SC 성공 {sc_ok}/{len(seeds)}")


if __name__ == "__main__":
    main()
