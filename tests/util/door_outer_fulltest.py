"""door+외곽 방향 전수: FC(2색칠 경계 방향) 가 최적인가.

FaceClusterPartition 이 방향 고정하는 경계 간선(door + outer) 전부를 조합 변수로
삼아, 강연결 유지 조합(사이클공간 feasible)을 전수하고 각각 내부만 SA 로 푼다.
bits=0 = FC 재현(하한). best < FC 면 2색칠 경계 방향이 최적이 아니라는 증거.

door_combo_scan 의 Feasibility / cycle_space_basis / solve_combo_task 를 그대로
재사용한다. 차이는 build_case 가 door 만이 아니라 door+outer 전부를 고정하는 것뿐.

    PYTHONHASHSEED=0 python tests/util/door_outer_fulltest.py [seeds] [N]

출력: tests/util/door_outer_fulltest.json
실험 전용 — 프로덕션(mr2s_module/)에 두지 않는다.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util import door_combo_scan as scan
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

REMOVE_RATIO = 0.4
TARGET_K = 10
N_PROCS = 12
MAX_FEASIBLE = 400   # seed 당 솔버 실행 상한 (초과 시 canonical + 무작위 샘플)
OUT = "tests/util/door_outer_fulltest.json"


def build_case_fc(seed: int, n: int) -> dict:
    """door+외곽 전부 directed. scan.build_case 와 같은 dict 구조 (bits=0=FC).

    Edge id 카운터를 0 으로 리셋하고 그래프를 만든다. 파티션(FaceClusterPartition)이
    내부에서 간선/면을 id 순서로 순회하므로, 리셋하지 않으면 build 호출마다 id 가
    전진해 클러스터링이 달라진다(같은 시드인데 파티션 비결정). 리셋하면 4/4 동일 파티션.
    """
    Edge._id_counter = itertools.count(0)
    raw, pts = build_delaunay(n, seed)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])
    np.random.seed(seed)
    random.seed(seed)
    partition = FaceClusterPartition(target_k=TARGET_K).run(graph)

    macros = []
    canon = {}
    owner = defaultdict(set)
    for mi, sg in enumerate(partition.sub_graphs):
        doors, free, verts = [], [], set()
        for e in sg.edges.values():
            verts.update(e.endpoints())
            if e.directed:
                doors.append(e.id)
                canon[e.id] = e.vertices
                owner[e.id].add(mi)
            else:
                free.append((e.id, *e.endpoints()))
        macros.append({"verts": sorted(verts), "doors": sorted(doors), "free": free})
    dids = sorted(canon)
    return {
        "seed": seed,
        "edges": [(e.id, *e.endpoints()) for e in graph.edges.values()],
        "pos": {int(i): [float(p[0]), float(p[1])] for i, p in enumerate(pts)},
        "macros": macros, "canon": canon,
        "owner": {e: sorted(owner[e]) for e in dids},
        "door_ids": dids, "bit_of": {e: i for i, e in enumerate(dids)},
    }


def enumerate_feasible(case: dict) -> tuple[list[int], dict]:
    basis = scan.cycle_space_basis(case)
    feas = scan.Feasibility(case)
    feasible = []
    for mask in range(1 << len(basis)):
        bits = 0
        for i, cyc in enumerate(basis):
            if (mask >> i) & 1:
                bits ^= cyc
        if feas.first_failure(bits) < 0:
            feasible.append(bits)
    stats = {
        "directed": len(case["door_ids"]),
        "cycle_dim": len(basis),
        "cycle_total": 1 << len(basis),
        "feasible": len(feasible),
        "macros": len(case["macros"]),
    }
    if 0 not in feasible:
        feasible = [0] + feasible  # FC(하한) 항상 포함
    if len(feasible) > MAX_FEASIBLE:
        rng = random.Random(case["seed"])
        rest = [b for b in feasible if b != 0]
        rng.shuffle(rest)
        feasible = [0] + rest[: MAX_FEASIBLE - 1]
        stats["sampled"] = True
    return feasible, stats


def main() -> None:
    seeds = list(range(6))
    n = 200
    if len(sys.argv) > 1:
        seeds = [int(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 2:
        n = int(sys.argv[2])

    cases, plans, stats = {}, {}, {}
    tasks = []
    for seed in seeds:
        case = build_case_fc(seed, n)
        feasible, stat = enumerate_feasible(case)
        cases[seed] = case
        plans[seed] = feasible
        stats[seed] = stat
        tasks += [(seed, b) for b in feasible]
        print(f"seed {seed}: directed={stat['directed']} cycle_dim={stat['cycle_dim']} "
              f"조합={stat['cycle_total']} feasible={stat['feasible']} "
              f"solver={len(feasible)}", flush=True)

    print(f"\n실행 {len(tasks)}회 (procs {N_PROCS}, N={n})", flush=True)
    results = {}
    started = time.perf_counter()
    with Pool(N_PROCS, initializer=scan._worker_init, initargs=(cases,)) as pool:
        for row in pool.imap_unordered(scan.solve_combo_task, tasks, 1):
            results[(row["seed"], row["bits"])] = row
            done = len(results)
            if done % 10 == 0 or done == len(tasks):
                rate = done / max(1e-9, time.perf_counter() - started)
                print(f"  {done}/{len(tasks)} "
                      f"(eta {(len(tasks)-done)/max(rate,1e-9)/60:.1f}m)", flush=True)

    payload = {"config": {"n_vertices": n, "remove_ratio": REMOVE_RATIO,
                          "target_k": TARGET_K, "seeds": seeds},
               "stats": {str(s): stats[s] for s in seeds}, "cases": {}}
    for seed in seeds:
        rows = []
        for bits in plans[seed]:
            r = dict(results[(seed, bits)])
            r["bits"] = str(bits)
            rows.append(r)
        payload["cases"][str(seed)] = {"stats": stats[seed], "rows": rows}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"\nwrote {OUT}", flush=True)

    print(f"\n=== 요약: FC(bits=0) vs 전수 best ===")
    print(f"{'seed':>4} {'FC':>9} {'best':>9} {'개선':>8} {'best flip':>9} {'feasible':>8} {'SC실패':>7}")
    wins = 0
    for seed in seeds:
        rows = payload["cases"][str(seed)]["rows"]
        fc = next((r for r in rows if r["bits"] == "0"), None)
        ok = [r for r in rows if not r.get("failed") and r["strong_connect_rate"] >= 1.0]
        nfail = len(rows) - len(ok)
        if not fc or fc.get("failed") or fc["strong_connect_rate"] < 1.0:
            print(f"{seed:>4}  FC 자체가 SC 실패?!", flush=True)
            continue
        best = min(ok, key=lambda r: r["apsp_sum"])
        imp = 100 * (best["apsp_sum"] - fc["apsp_sum"]) / fc["apsp_sum"]
        bf = bin(int(best["bits"])).count("1")
        if best["apsp_sum"] < fc["apsp_sum"] - 1e-9:
            wins += 1
        print(f"{seed:>4} {fc['apsp_sum']:>9.4f} {best['apsp_sum']:>9.4f} "
              f"{imp:>+7.2f}% {bf:>9} {len(ok):>8} {nfail:>7}", flush=True)
    print(f"\n전수 best 가 FC 보다 나은 seed: {wins}/{len(seeds)}", flush=True)


if __name__ == "__main__":
    main()
