"""Q1 2단계: P=TRUE macro 에서 best(S_do) vs best(S_fc) 비교.

평가 기준:
  - 총 자유 비트(내부 + 외곽) ≤ EXACT_CAP → 전수열거 = 참값 (exact).
  - 초과(big macro) → 동일 예산 페어드 탐색 (budgeted):
      두 팔 모두 '그리디 mixed-SC 구성 초기해 + 단일 비트 flip 언덕오르기',
      restart R회 × 평가 EVAL_BUDGET 회로 예산 동일. 차이는 탐색 변수뿐
      (FC=내부만, DO=내부+외곽). 추가로 FC 최적해에 내부 고정한 채 외곽
      flip 조합만 바꿔 개선 존재 여부 교차검사(outer sweep).

목적 함수: macro-국소 APSP 합(hop, 낮을수록 좋음), 강연결 아니면 무효.

    PYTHONHASHSEED=0 python tests/util/door_q1_stage2.py

출력: tests/util/door_q1_stage2.json
"""

from __future__ import annotations

import json
import random as pyrandom
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.door_outer_flip_scan import build_macro_model
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

SEEDS = [42, 0, 1, 2, 3]
N_VERTICES = 200
TARGET_K = 10
REMOVE_RATIO = 0.4
N_PROCS = 14
EXACT_CAP = 22          # 2^22 이하면 전수
EVAL_BUDGET = 40000     # budgeted 팔: restart 당 목적함수 평가 횟수
N_RESTARTS = 8
OUTER_SWEEP_CAP = 1 << 18
OUT = "tests/util/door_q1_stage2.json"


def apsp(n: int, arcs) -> float | None:
    fwd: list[list[int]] = [[] for _ in range(n)]
    for u, v in arcs:
        fwd[u].append(v)
    total = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        seen = 1
        while q:
            nxt = []
            for x in q:
                dx = dist[x] + 1
                for y in fwd[x]:
                    if dist[y] < 0:
                        dist[y] = dx
                        nxt.append(y)
                        seen += 1
            q = nxt
        if seen < n:
            return None
        total += sum(dist)
    return float(total)


def arcs_of(fixed, edges, bits):
    arcs = list(fixed)
    for i, (u, v) in enumerate(edges):
        arcs.append((v, u) if (bits >> i) & 1 else (u, v))
    return arcs


# ---------- exact ----------

def exact_chunk(args):
    n, fixed, edges, start, end, mask_filter, mask_bits = args
    """mask_filter 가 None 이 아니면 (bits & mask_bits)==mask_filter 만 평가."""
    best = None
    best_bits = -1
    for bits in range(start, end):
        if mask_filter is not None and (bits & mask_bits) != mask_filter:
            continue
        s = apsp(n, arcs_of(fixed, edges, bits))
        if s is not None and (best is None or s < best):
            best = s
            best_bits = bits
    return best, best_bits


def run_exact(pool, n, fixed, edges, outer_mask):
    """반환 (best_do, best_fc). outer_mask: 외곽 비트 위치 (fc 는 그 비트 0 고정)."""
    m = len(edges)
    total = 1 << m
    step = 1 << 20
    jobs_do = [(n, fixed, edges, s, min(s + step, total), None, 0)
               for s in range(0, total, step)]
    jobs_fc = [(n, fixed, edges, s, min(s + step, total), 0, outer_mask)
               for s in range(0, total, step)]
    best_do = best_fc = None
    for b, _ in pool.imap_unordered(exact_chunk, jobs_do):
        if b is not None and (best_do is None or b < best_do):
            best_do = b
    for b, _ in pool.imap_unordered(exact_chunk, jobs_fc):
        if b is not None and (best_fc is None or b < best_fc):
            best_fc = b
    return best_do, best_fc


# ---------- budgeted ----------

def greedy_init(n, fixed, edges, locked_bits, lock_mask, rng):
    """mixed-SC 유지하며 간선 하나씩 방향 확정. locked(외곽 canonical 등)는
    lock_mask 비트가 1 인 위치로, locked_bits 값 그대로 둔다."""
    m = len(edges)
    bits = locked_bits
    undecided = [i for i in range(m) if not (lock_mask >> i) & 1]
    rng.shuffle(undecided)
    decided_mask = lock_mask

    def mixed_sc(cur_bits, cur_mask):
        arcs = list(fixed)
        for i, (u, v) in enumerate(edges):
            if (cur_mask >> i) & 1:
                arcs.append((v, u) if (cur_bits >> i) & 1 else (u, v))
            else:
                arcs.append((u, v))
                arcs.append((v, u))
        fwd = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        for u, v in arcs:
            fwd[u].append(v)
            rev[v].append(u)
        for a in (fwd, rev):
            seen = bytearray(n)
            seen[0] = 1
            st = [0]
            c = 1
            while st:
                x = st.pop()
                for y in a[x]:
                    if not seen[y]:
                        seen[y] = 1
                        st.append(y)
                        c += 1
            if c < n:
                return False
        return True

    for i in undecided:
        pref = rng.getrandbits(1)
        for b in (pref, 1 - pref):
            trial = (bits & ~(1 << i)) | (b << i)
            if mixed_sc(trial, decided_mask | (1 << i)):
                bits = trial
                decided_mask |= 1 << i
                break
        else:
            return None  # bridgeless 라 이론상 안 옴
    return bits


def climb_task(args):
    (n, fixed, edges, lock_mask, locked_bits, seed, budget) = args
    rng = pyrandom.Random(seed)
    bits = greedy_init(n, fixed, edges, locked_bits, lock_mask, rng)
    if bits is None:
        return None, -1
    free_idx = [i for i in range(len(edges)) if not (lock_mask >> i) & 1]
    cur = apsp(n, arcs_of(fixed, edges, bits))
    evals = 1
    best, best_bits = cur, bits
    while evals < budget:
        i = free_idx[rng.randrange(len(free_idx))]
        cand = bits ^ (1 << i)
        s = apsp(n, arcs_of(fixed, edges, cand))
        evals += 1
        if s is not None and (s < cur or (s == cur and rng.random() < 0.2)):
            bits, cur = cand, s
            if cur < best:
                best, best_bits = cur, bits
    return best, best_bits


def run_budgeted(pool, n, fixed, edges, outer_mask, macro_tag):
    """FC 팔: 외곽 비트 lock(=canonical 0). DO 팔: lock 없음."""
    jobs = []
    for arm, lock in (("fc", outer_mask), ("do", 0)):
        for r in range(N_RESTARTS):
            jobs.append((arm, (n, fixed, edges, lock, 0,
                               hash((macro_tag, arm, r)) & 0x7FFFFFFF,
                               EVAL_BUDGET)))
    results = {"fc": [], "do": []}
    outs = pool.map(climb_task, [j for _, j in jobs])
    fc_best_bits = None
    for (arm, _), (best, bits) in zip(jobs, outs):
        if best is not None:
            results[arm].append(best)
            if arm == "fc" and best == min(results["fc"]):
                fc_best_bits = bits
    best_fc = min(results["fc"]) if results["fc"] else None
    best_do = min(results["do"]) if results["do"] else None

    # outer sweep: FC 최적해 내부 고정, 외곽 조합만 전수/부분 전수
    sweep_best = None
    if fc_best_bits is not None:
        outer_idx = [i for i in range(len(edges)) if (outer_mask >> i) & 1]
        n_outer = len(outer_idx)
        n_pat = 1 << n_outer
        pats = range(n_pat) if n_pat <= OUTER_SWEEP_CAP else None
        if pats is not None:
            for pat in pats:
                bits = fc_best_bits
                for j, i in enumerate(outer_idx):
                    if (pat >> j) & 1:
                        bits ^= 1 << i
                s = apsp(n, arcs_of(fixed, edges, bits))
                if s is not None and (sweep_best is None or s < sweep_best):
                    sweep_best = s
    return best_fc, best_do, sweep_best


def main() -> None:
    scan = json.load(open("tests/util/door_outer_flip_scan.json"))
    ptrue = {(g["seed"], m["macro"]): m
             for g in scan["graphs"] for m in g["macros"] if m["p_flip_exists"]}
    print(f"P=TRUE macros: {sorted(ptrue)}")

    rows = []
    with Pool(N_PROCS) as pool:
        for seed in SEEDS:
            if not any(s == seed for s, _ in ptrue):
                continue
            raw, _pts = build_delaunay(N_VERTICES, seed)
            pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
            graph = Graph(
                edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()]
            )
            np.random.seed(seed)
            pyrandom.seed(seed)
            fc = FaceClusterPartition(target_k=TARGET_K).run(graph)
            np.random.seed(seed)
            pyrandom.seed(seed)
            do = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)
            fc_dir = {e.id: e.vertices for sg in fc.sub_graphs
                      for e in sg.edges.values() if e.directed}
            door_ids = {e.id for sg in do.sub_graphs
                        for e in sg.edges.values() if e.directed}

            for mi, sg in enumerate(fc.sub_graphs):
                if (seed, mi) not in ptrue or not sg.edges:
                    continue
                nv, doors, outers, interns = build_macro_model(sg, fc_dir, door_ids)
                # 변수 간선 = 외곽 + 내부. 외곽 canonical 은 bit0 방향으로 인코딩.
                edges = outers + interns
                outer_mask = (1 << len(outers)) - 1
                m = len(edges)
                t0 = time.perf_counter()
                if m <= EXACT_CAP:
                    b_do, b_fc = run_exact(pool, nv, doors, edges, outer_mask)
                    mode = "exact"
                    sweep = None
                else:
                    b_fc, b_do, sweep = run_budgeted(
                        pool, nv, doors, edges, outer_mask, f"{seed}:{mi}")
                    mode = "budgeted"
                dt = round(time.perf_counter() - t0, 1)
                gain = (None if b_fc is None or b_do is None
                        else 100.0 * (b_fc - b_do) / b_fc)
                rows.append({
                    "seed": seed, "macro": mi, "nv": nv,
                    "n_outer": len(outers), "n_intern": len(interns),
                    "mode": mode, "best_fc": b_fc, "best_do": b_do,
                    "outer_sweep_best": sweep, "gain_pct": gain,
                    "elapsed_s": dt,
                })
                print(f"seed {seed} macro {mi} [{mode}] nv={nv} "
                      f"bits={m}: best_fc={b_fc} best_do={b_do} "
                      f"sweep={sweep} gain={gain if gain is None else round(gain,3)}% "
                      f"({dt}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump({"exact_cap": EXACT_CAP, "eval_budget": EVAL_BUDGET,
                   "n_restarts": N_RESTARTS, "rows": rows}, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
