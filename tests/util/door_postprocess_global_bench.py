"""전역 지표 후처리 벤치: FC 솔버 해 + (외곽 / 외곽+door) flip 후처리.

질문 대응:
  Q2. 외곽 flip 후처리의 '실제(전역 stretch)' 개선 폭.
  Q3. 방법 T = "전역 SC 유지 + 전역 stretch 개선 시 door flip 수용하는
      탐욕 후처리" — T(외곽+door) 가 face-cycle 원본보다 좋은가.

절차 (seed 페어드, 멀티프로세싱):
  1) FC 솔버(door_vs_facecycle_solver_bench.build_solver, FaceClusterPartition)
     로 해를 구함 → base stretch.
  2) 간선 분류: 원 그래프에 target_k=10 파티션 재실행 → door(2-macro 공유),
     외곽(directed 인데 door 아님). 솔버 내부 파티션(축약 그래프, k 자동)과
     다를 수 있으나 후보 집합 정의용 — flip 자체는 전역 SC/stretch 만 본다.
  3) 후처리 A: 외곽 후보만 첫개선 탐욕 flip (stretch 낮아지면 수용, inf=SC
     깨짐이면 거부). 개선 없는 패스가 나올 때까지, 최대 4패스.
  4) 후처리 T: 같은 방식, 후보 = 외곽 + door. (FC 원본에서 시작 — A 와 독립.)

    PYTHONHASHSEED=0 python tests/util/door_postprocess_global_bench.py [n_seeds]

출력: tests/util/door_postprocess_global.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle.face_cluster_partition import FaceClusterPartition
from mr2s_module.domain import Edge, Graph, Solution
from mr2s_module.evaluator import ApspSumRanker
from mr2s_module.util.sample_set import empty_binary_sample_set
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.door_vs_facecycle_solver_bench import build_solver
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

N_VERTICES = 200
REMOVE_RATIO = 0.4
TARGET_K = 10
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_PROCS = 8
MAX_PASSES = 4
MAX_EVALS = 400
OUT = "tests/util/door_postprocess_global.json"


def classify(graph: Graph, seed: int) -> tuple[set[int], set[int]]:
    """(door_ids, outer_ids) — target_k=10 파티션 기준."""
    np.random.seed(seed)
    random.seed(seed)
    fc = FaceClusterPartition(target_k=TARGET_K).run(graph)
    np.random.seed(seed)
    random.seed(seed)
    do = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)
    fc_dir = {e.id for sg in fc.sub_graphs for e in sg.edges.values() if e.directed}
    door_ids = {e.id for sg in do.sub_graphs for e in sg.edges.values() if e.directed}
    return door_ids, fc_dir - door_ids


def greedy_flip(
    ranker: ApspSumRanker,
    graph: Graph,
    edges: dict[int, tuple[int, int]],
    candidates: list[int],
    base: float,
) -> tuple[float, int, int]:
    """첫개선 탐욕 flip. 반환 (최종 stretch, 수용 flip 수, 평가 횟수)."""
    cur = base
    n_flip = 0
    n_eval = 0
    sset = empty_binary_sample_set()
    for _ in range(MAX_PASSES):
        improved = False
        for eid in candidates:
            if n_eval >= MAX_EVALS:
                break
            u, v = edges[eid]
            edges[eid] = (v, u)
            s = ranker.run(Solution(edges=edges, graph=graph, sample_set=sset))
            n_eval += 1
            if s < cur:
                cur = s
                n_flip += 1
                improved = True
            else:
                edges[eid] = (u, v)
        if not improved or n_eval >= MAX_EVALS:
            break
    return cur, n_flip, n_eval


def run_seed(seed: int) -> dict:
    raw, _pts = build_delaunay(N_VERTICES, seed)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])

    np.random.seed(seed)
    random.seed(seed)
    solver = build_solver(FaceClusterPartition, seed, 40)
    t0 = time.perf_counter()
    try:
        sol = solver.run(graph)
    except RuntimeError:  # cap 40 분할 실패 seed → 60 승급 (벤치와 동일 규칙)
        np.random.seed(seed)
        random.seed(seed)
        sol = build_solver(FaceClusterPartition, seed, 60).run(graph)
    t_solve = time.perf_counter() - t0

    ranker = ApspSumRanker()
    sset = empty_binary_sample_set()
    base = ranker.run(Solution(edges=dict(sol.edges), graph=graph, sample_set=sset))

    door_ids, outer_ids = classify(graph, seed)
    cand_outer = sorted(i for i in outer_ids if i in sol.edges)
    cand_all = sorted(i for i in (outer_ids | door_ids) if i in sol.edges)

    t0 = time.perf_counter()
    e_a = dict(sol.edges)
    s_a, f_a, ev_a = greedy_flip(ranker, graph, e_a, cand_outer, base)
    e_t = dict(sol.edges)
    s_t, f_t, ev_t = greedy_flip(ranker, graph, e_t, cand_all, base)
    t_pp = time.perf_counter() - t0

    return {
        "seed": seed, "base": base,
        "outer_pp": s_a, "outer_flips": f_a, "outer_evals": ev_a,
        "t_pp": s_t, "t_flips": f_t, "t_evals": ev_t,
        "n_outer": len(cand_outer), "n_door": len(cand_all) - len(cand_outer),
        "t_solve_s": round(t_solve, 1), "t_pp_s": round(t_pp, 1),
    }


def main() -> None:
    with Pool(N_PROCS) as pool:
        rows = []
        for r in pool.imap_unordered(run_seed, range(N_SEEDS)):
            rows.append(r)
            print(f"seed {r['seed']:>3}: base={r['base']:.4f} "
                  f"outer→{r['outer_pp']:.4f} ({r['outer_flips']}flip) "
                  f"T→{r['t_pp']:.4f} ({r['t_flips']}flip) "
                  f"[solve {r['t_solve_s']}s pp {r['t_pp_s']}s]", flush=True)
    rows.sort(key=lambda r: r["seed"])

    import statistics as st
    base = [r["base"] for r in rows]
    o = [r["outer_pp"] for r in rows]
    t = [r["t_pp"] for r in rows]
    d_o = [100 * (r["base"] - r["outer_pp"]) / r["base"] for r in rows]
    d_t = [100 * (r["base"] - r["t_pp"]) / r["base"] for r in rows]
    print(f"\n=== 요약 (seed {len(rows)}개, V={N_VERTICES}) ===")
    print(f"base stretch 평균     : {st.mean(base):.4f}")
    print(f"외곽 후처리(A) 평균   : {st.mean(o):.4f}  개선 {st.mean(d_o):+.3f}% "
          f"(개선 seed {sum(1 for x in d_o if x > 1e-9)}/{len(rows)}, "
          f"flip 평균 {st.mean([r['outer_flips'] for r in rows]):.1f})")
    print(f"T=외곽+door(T) 평균   : {st.mean(t):.4f}  개선 {st.mean(d_t):+.3f}% "
          f"(개선 seed {sum(1 for x in d_t if x > 1e-9)}/{len(rows)}, "
          f"flip 평균 {st.mean([r['t_flips'] for r in rows]):.1f})")

    with open(OUT, "w") as f:
        json.dump({"n_vertices": N_VERTICES, "target_k": TARGET_K,
                   "max_passes": MAX_PASSES, "max_evals": MAX_EVALS,
                   "rows": rows}, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
