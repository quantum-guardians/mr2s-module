"""door flip 후처리 시각화용 데이터 생성 (10 seed, V=200, remove 0.4).

seed 마다:
  1) face-cycle 기본 솔버(DnC+SA) 해 → 기본 우회율(stretch)
  2) door(두 구역 공유 경계 간선)만 첫개선 탐욕 flip 후처리 → 개선 우회율
  3) 어떤 door 를 뒤집었는지 + 좌표/면(구역 색칠용) 기록

    PYTHONHASHSEED=0 python tests/util/door_flip_viz_bench.py

출력: tests/util/door_flip_viz_data.json (export_door_flip_viz.py 가 HTML 생성)
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import ApspSumRanker
from mr2s_module.util.sample_set import empty_binary_sample_set
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.door_vs_facecycle_solver_bench import build_solver
from tests.util.robbins_door_bruteforce import (
    build_delaunay,
    extract_cells_and_shared,
    prune_keep_biconnected,
)

N_VERTICES = 200
REMOVE_RATIO = 0.4
TARGET_K = 10
SEEDS = list(range(10))
N_PROCS = 8
MAX_PASSES = 4
MAX_EVALS = 400
OUT = "tests/util/door_flip_viz_data.json"


def face_macros(cells, key2macros, macro_size):
    """각 면 → 소속 구역 (export_macro_viz 와 동일한 투표 방식)."""
    def best(votes: Counter) -> int:
        return max(votes.items(), key=lambda kv: (kv[1], -macro_size[kv[0]]))[0]

    out = []
    for cell in cells:
        votes: Counter = Counter()
        for i in range(len(cell)):
            k = tuple(sorted((cell[i], cell[(i + 1) % len(cell)])))
            ms = key2macros.get(k, set())
            if len(ms) == 1:
                votes[next(iter(ms))] += 1
        if not votes:
            for i in range(len(cell)):
                k = tuple(sorted((cell[i], cell[(i + 1) % len(cell)])))
                for m in key2macros.get(k, ()):
                    votes[m] += 1
        out.append(best(votes) if votes else -1)
    return out


def run_seed(seed: int) -> dict:
    raw, pts = build_delaunay(N_VERTICES, seed)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
    cells, _ = extract_cells_and_shared(pruned, pts)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])

    # 기본 해 (face-cycle 솔버, 100-seed 벤치와 동일 구성)
    np.random.seed(seed)
    random.seed(seed)
    solver = build_solver(FaceClusterPartition, seed, 40)
    try:
        sol = solver.run(graph)
    except RuntimeError:
        np.random.seed(seed)
        random.seed(seed)
        sol = build_solver(FaceClusterPartition, seed, 60).run(graph)

    ranker = ApspSumRanker()
    sset = empty_binary_sample_set()
    base_stretch = ranker.run(
        Solution(edges=dict(sol.edges), graph=graph, sample_set=sset))

    # 구역/door 분류 (target_k=10 파티션)
    np.random.seed(seed)
    random.seed(seed)
    fc = FaceClusterPartition(target_k=TARGET_K).run(graph)
    np.random.seed(seed)
    random.seed(seed)
    do = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)
    door_ids = {e.id for sg in do.sub_graphs for e in sg.edges.values() if e.directed}

    key2macros: dict[tuple[int, int], set[int]] = {}
    macro_size: dict[int, int] = {}
    for mi, sg in enumerate(fc.sub_graphs):
        vs: set[int] = set()
        for e in sg.edges.values():
            k = tuple(sorted(e.endpoints()))
            key2macros.setdefault(k, set()).add(mi)
            vs.update(e.endpoints())
        macro_size[mi] = len(vs)

    # door 만 첫개선 탐욕 flip
    edges = dict(sol.edges)
    cur = base_stretch
    flipped: list[int] = []
    n_eval = 0
    t0 = time.perf_counter()
    for _ in range(MAX_PASSES):
        improved = False
        for eid in sorted(door_ids):
            if eid not in edges or n_eval >= MAX_EVALS:
                continue
            u, v = edges[eid]
            edges[eid] = (v, u)
            s = ranker.run(Solution(edges=edges, graph=graph, sample_set=sset))
            n_eval += 1
            if s < cur:
                cur = s
                improved = True
                if eid in flipped:
                    flipped.remove(eid)  # 재뒤집기 = 원복
                else:
                    flipped.append(eid)
            else:
                edges[eid] = (u, v)
        if not improved or n_eval >= MAX_EVALS:
            break
    t_pp = time.perf_counter() - t0

    edge_rows = []
    for e in graph.edges.values():
        u, v = e.endpoints()
        edge_rows.append({
            "eid": e.id, "u": int(u), "v": int(v),
            "door": e.id in door_ids,
            "base": [int(x) for x in sol.edges[e.id]],
            "after": [int(x) for x in edges[e.id]],
            "flipped": e.id in flipped,
        })

    fm = face_macros(cells, key2macros, macro_size)
    print(f"seed {seed}: base={base_stretch:.4f} after={cur:.4f} "
          f"({100*(base_stretch-cur)/base_stretch:+.2f}%) flips={len(flipped)} "
          f"pp={t_pp:.0f}s", flush=True)
    return {
        "seed": seed,
        "verts": {int(i): [float(pts[i, 0]), float(pts[i, 1])]
                  for i in range(len(pts))},
        "faces": [{"verts": [int(x) for x in c], "macro": m}
                  for c, m in zip(cells, fm)],
        "edges": edge_rows,
        "n_door": len(door_ids),
        "stretch_base": base_stretch,
        "stretch_after": cur,
        "n_flip": len(flipped),
    }


def main() -> None:
    with Pool(N_PROCS) as pool:
        rows = list(pool.imap_unordered(run_seed, SEEDS))
    rows.sort(key=lambda r: r["seed"])
    with open(OUT, "w") as f:
        json.dump({"n_vertices": N_VERTICES, "remove_ratio": REMOVE_RATIO,
                   "seeds": SEEDS, "graphs": rows}, f)
    imp = [100 * (r["stretch_base"] - r["stretch_after"]) / r["stretch_base"]
           for r in rows]
    print(f"평균 개선 {sum(imp)/len(imp):+.2f}%  wrote {OUT}")


if __name__ == "__main__":
    main()
