"""door-only vs face-cycle 솔버 성능 벤치 (V=200, DnC+SA, 100 seed 페어드).

두 팔은 face_cycle 파티션 클래스만 다르다:
    - FC (face-cycle) : FaceClusterPartition — door + 외곽선 전부 directed 고정
    - DO (door-only)  : DoorOnlyFaceClusterPartition — 진짜 door(인접 macro
      공유 경계)만 face 순회 방향 카피, 외곽선은 free 변수

측정: strong_connect_rate(실측), SC 성공(==1.0) 횟수, apsp_sum=평균 stretch(양쪽 SC 성공
seed 만 비교), flow_score, 벽시계 시간.

    PYTHONHASHSEED=0 python tests/util/door_vs_facecycle_solver_bench.py [start] [end]
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np

from mr2s_module.domain import Edge, Graph
from mr2s_module.evaluator import Evaluator
from mr2s_module.reduction import ReductionMr2sSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected
from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer

N_VERTICES = 200
REMOVE_RATIO = 0.4
MAX_VERTICES = 40
SEED_START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEED_END = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # exclusive


def build_solver(face_cycle_cls, seed: int, cap: int) -> ReductionMr2sSolver:
    sa = SAMR2SSolver(
        sweeps_per_temperature=2,
        num_restarts=4,
        random_seed=seed,
    )
    face_cycle = face_cycle_cls(target_k=2, clusterer=KMeansFaceClusterer())
    solver = DnCMr2sSolver(
        mr2s_solver=sa,
        face_cycle=face_cycle,
        graph_partition_strategy=VertexCountPartitionStrategy(
            face_cycle=face_cycle,
            max_vertices=cap,
        ),
    )
    return ReductionMr2sSolver(mr2s_solver=solver, evaluator=Evaluator())


def run_arm(face_cycle_cls, graph: Graph, seed: int, cap: int):
    np.random.seed(seed)
    random.seed(seed)
    solver = build_solver(face_cycle_cls, seed, cap)
    t0 = time.perf_counter()
    sol = solver.run(graph)
    dt = time.perf_counter() - t0
    s = sol.score
    return s.strong_connect_rate, s.apsp_sum, s.flow_score, dt


def main() -> None:
    rows = []
    for seed in range(SEED_START, SEED_END):
        raw, _pts = build_delaunay(N_VERTICES, seed)
        pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
        graph = Graph(
            edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()]
        )
        # cap 40 이 안 쪼개지는 fixture 는 60, 80 으로 승급 (양팔 동일 cap 유지)
        for cap in (MAX_VERTICES, 60, 80):
            try:
                fc = run_arm(FaceClusterPartition, graph, seed, cap)
                do = run_arm(DoorOnlyFaceClusterPartition, graph, seed, cap)
                break
            except RuntimeError:
                continue
        else:
            print(f"seed {seed:>3}  SKIP: partition failed at all caps", flush=True)
            continue
        rows.append((seed, fc, do))
        print(
            f"seed {seed:>3}  FC sc={fc[0]:.3f} apsp={fc[1]:>7.4f} "
            f"flow={fc[2]:>7.1f} t={fc[3]:>5.1f}s | "
            f"DO sc={do[0]:.3f} apsp={do[1]:>7.4f} flow={do[2]:>7.1f} "
            f"t={do[3]:>5.1f}s",
            flush=True,
        )

    def agg(idx: int):
        fc_v = [r[1][idx] for r in rows]
        do_v = [r[2][idx] for r in rows]
        return sum(fc_v) / len(fc_v), sum(do_v) / len(do_v)

    n = len(rows)
    fc_sc_ok = sum(1 for r in rows if r[1][0] >= 1.0)
    do_sc_ok = sum(1 for r in rows if r[2][0] >= 1.0)
    sc_fc, sc_do = agg(0)
    fl_fc, fl_do = agg(2)
    t_fc, t_do = agg(3)

    both = [r for r in rows if r[1][0] >= 1.0 and r[2][0] >= 1.0]
    print(f"\n=== 요약 (seed {n}개, V={N_VERTICES}, max_vertices={MAX_VERTICES}) ===")
    print(f"SC 성공     : FC {fc_sc_ok}/{n}  DO {do_sc_ok}/{n}")
    print(f"SC rate 평균: FC {sc_fc:.4f}  DO {sc_do:.4f}")
    print(f"flow 평균   : FC {fl_fc:.2f}  DO {fl_do:.2f}")
    print(f"시간 평균   : FC {t_fc:.2f}s  DO {t_do:.2f}s")
    if both:
        a_fc = sum(r[1][1] for r in both) / len(both)
        a_do = sum(r[2][1] for r in both) / len(both)
        wins_do = sum(1 for r in both if r[2][1] < r[1][1])
        ties = sum(1 for r in both if r[2][1] == r[1][1])
        print(
            f"APSP (양쪽 SC 성공 {len(both)}개): FC {a_fc:.4f}  DO {a_do:.4f} "
            f"({100 * (a_do - a_fc) / a_fc:+.2f}%)  DO 승 {wins_do} "
            f"무 {ties} 패 {len(both) - wins_do - ties}"
        )


if __name__ == "__main__":
    main()
