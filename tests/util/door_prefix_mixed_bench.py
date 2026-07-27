"""Door 선고정 시 macro 별 mixed-graph 강방향화 존재 벤치.

정리(Boesch-Tindell): bridgeless mixed graph 가 강방향화를 가지려면
mixed 상태(고정 간선 단방향 + 자유 간선 양방향)로 강연결이면 충분.

각 macro 부분그래프에 대해
  - pipeline door 방향(FaceClusterPartition 출력) 고정 시 강연결 여부
  - 무작위 door 방향(전역 일관 동전던지기) 고정 시 강연결 여부
를 여러 seed 에 걸쳐 센다.

PYTHONHASHSEED=0 python tests/util/door_prefix_mixed_bench.py
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np

from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

N_POINTS = 200
TARGET_K = 10
SEEDS = range(30)
N_RANDOM_TRIALS = 20


def macro_mixed_sc(sub_graph, door_dir: dict[int, tuple[int, int]]) -> bool:
    """door 는 door_dir 방향 단방향, 자유 간선은 양방향으로 강연결 검사."""
    D = nx.DiGraph()
    for e in sub_graph.edges.values():
        u, v = e.endpoints()
        D.add_node(u)
        D.add_node(v)
        if e.id in door_dir:
            t, h = door_dir[e.id]
            D.add_edge(t, h)
        else:
            D.add_edge(u, v)
            D.add_edge(v, u)
    return nx.is_strongly_connected(D)


def main() -> None:
    total_macros = 0
    pipe_fail = 0
    rand_fail_frac_sum = 0.0
    rand_all_ok_seeds = 0
    per_seed_rows = []

    for seed in SEEDS:
        raw, _pts = build_delaunay(N_POINTS, seed)
        pruned = prune_keep_biconnected(raw, 0.4, seed)
        graph = Graph(
            edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()]
        )
        np.random.seed(seed)
        random.seed(seed)
        result = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)
        subs = [sg for sg in result.sub_graphs if sg.edges]

        # DoorOnly: directed 간선 = 진짜 door (인접 macro 공유 경계) 뿐.
        # 외평면과 맞닿는 외곽선/intra boundary 는 이미 free.
        pipe_dir: dict[int, tuple[int, int]] = {}
        for sg in subs:
            for e in sg.edges.values():
                if e.directed:
                    pipe_dir[e.id] = e.vertices

        n_macros = len(subs)
        n_pipe_fail = sum(not macro_mixed_sc(sg, pipe_dir) for sg in subs)

        # 무작위 door 방향 (edge id 별 전역 일관 flip)
        rng = random.Random(seed * 7919 + 1)
        n_rand_fail_total = 0
        for _ in range(N_RANDOM_TRIALS):
            rand_dir = {
                eid: (th if rng.random() < 0.5 else (th[1], th[0]))
                for eid, th in pipe_dir.items()
            }
            n_rand_fail_total += sum(
                not macro_mixed_sc(sg, rand_dir) for sg in subs
            )
        rand_frac = n_rand_fail_total / (N_RANDOM_TRIALS * n_macros)

        total_macros += n_macros
        pipe_fail += n_pipe_fail
        rand_fail_frac_sum += rand_frac
        if n_rand_fail_total == 0:
            rand_all_ok_seeds += 1
        per_seed_rows.append(
            (seed, n_macros, len(pipe_dir), n_pipe_fail, rand_frac)
        )

    print("seed  macros  doors  pipe_fail  rand_fail_frac")
    for row in per_seed_rows:
        print(f"{row[0]:>4}  {row[1]:>6}  {row[2]:>5}  {row[3]:>9}  {row[4]:>13.3f}")
    n_seeds = len(per_seed_rows)
    print(
        f"\n총 macro {total_macros} (seed {n_seeds}개): "
        f"pipeline door 고정 실패 {pipe_fail} "
        f"({100.0 * pipe_fail / total_macros:.1f}%)"
    )
    print(
        f"무작위 door 고정 실패율 평균 {100.0 * rand_fail_frac_sum / n_seeds:.1f}% "
        f"(seed 당 {N_RANDOM_TRIALS}회, 전 시도 무실패 seed {rand_all_ok_seeds}/{n_seeds})"
    )


if __name__ == "__main__":
    main()
