"""Q1 완전 판정: 큰 macro 포함 전 macro 에서 S_do = S_fc 인지 전수 확인.

평가 기준 (정확한 명제):
  S_fc = { 강연결 배향 : door·외곽 canonical(face 순회 방향), 내부 임의 }
  S_do = { 강연결 배향 : door canonical, 외곽·내부 임의 }
  판정 명제 P(macro): "canonical 아닌 외곽 방향 조합 중, 어떤 내부 배향으로든
  강연결이 되는 것이 존재하는가"
  P 거짓  ⇔  S_do = S_fc  ⇒  best(S_do) = best(S_fc)  (Q1 종결: SA 탓 아님)
  P 참    ⇒  S_do ⊋ S_fc  (그 조합 수를 세고 2단계 필요)

내부 자유도 처리 (전수 대신 정리 사용, 손실 없음):
  macro 는 bridgeless(면 사이클 논증 + 실측) → Boesch–Tindell 정리에 의해
  "door+외곽 고정, 내부 자유" mixed graph 가 강연결(내부=양방향 취급)이면
  내부 배향 존재. 즉 외곽 2^o 만 돌고 내부는 mixed-SC 검사로 정확 판정.

계산 축약 (동치 변환):
  1) 내부 간선 연결성분을 supernode 로 축약 — 내부 양방향이므로 mixed-SC 는
     축약 그래프의 SC 와 동치. 외곽 간선의 양끝이 같은 supernode 면 self-loop:
     방향 무관하게 항상 유효 → 그 간선 하나만 뒤집어도 강연결 유지 ⇒ P 즉시 참.
  2) 체인 병합 — 고정 arc 없이 변수 간선 정확히 2개만 달린 노드는 직렬 통과만
     가능(아니면 그 노드가 source/sink) → 간선 2개를 변수 1개로 병합.

추가 검증 1번 항목: canonical outline(door+외곽)이 방향 폐곡선(원형)인지 —
outline 정점 전부 in-deg == out-deg 인지, 단순 사이클(전부 1)인지 검사.

    PYTHONHASHSEED=0 python tests/util/door_outer_flip_scan.py

출력: tests/util/door_outer_flip_scan.json
"""

from __future__ import annotations

import json
import random
import time
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

SEEDS = [42, 0, 1, 2, 3]
N_VERTICES = 200
TARGET_K = 10
REMOVE_RATIO = 0.4
N_PROCS = 14
CHUNK_BITS = 20  # 2^20 단위로 분할
OUT = "tests/util/door_outer_flip_scan.json"


# ---------- 축약 그래프 SC 판정 ----------

def _sc(n: int, adj: list[list[int]], radj: list[list[int]]) -> bool:
    """0..n-1, 정방향/역방향 인접 — 정점 0 기준 양방향 전도달이면 SC."""
    for a in (adj, radj):
        seen = bytearray(n)
        seen[0] = 1
        stack = [0]
        cnt = 1
        while stack:
            x = stack.pop()
            for y in a[x]:
                if not seen[y]:
                    seen[y] = 1
                    stack.append(y)
                    cnt += 1
        if cnt < n:
            return False
    return True


def scan_chunk(args) -> tuple[int, int | None]:
    """(n, fixed_arcs, var_edges, start, end) 범위 전수.

    bit=0 이 canonical. 반환: (유효 조합 수, canonical 아닌 유효 예시 mask 1개).
    """
    n, fixed_arcs, var_edges, start, end = args
    m = len(var_edges)
    n_ok = 0
    example = None
    for mask in range(start, end):
        adj: list[list[int]] = [[] for _ in range(n)]
        radj: list[list[int]] = [[] for _ in range(n)]
        for u, v in fixed_arcs:
            adj[u].append(v)
            radj[v].append(u)
        for i in range(m):
            u, v = var_edges[i]
            if (mask >> i) & 1:
                u, v = v, u
            adj[u].append(v)
            radj[v].append(u)
        if _sc(n, adj, radj):
            n_ok += 1
            if mask != 0 and example is None:
                example = mask
    return n_ok, example


# ---------- macro 추출/축약 ----------

def build_macro_model(sg, fc_dir, door_ids):
    verts = sorted({v for e in sg.edges.values() for v in e.endpoints()})
    vid = {v: i for i, v in enumerate(verts)}
    doors, outers, interns = [], [], []
    for e in sg.edges.values():
        u, v = e.endpoints()
        if e.id in door_ids:
            t, h = fc_dir[e.id]
            doors.append((vid[t], vid[h]))
        elif e.id in fc_dir:
            t, h = fc_dir[e.id]
            outers.append((vid[t], vid[h]))
        else:
            interns.append((vid[u], vid[v]))
    return len(verts), doors, outers, interns


def outline_circularity(n, doors, outers):
    """canonical outline 이 방향 폐곡선인지."""
    indeg = [0] * n
    outdeg = [0] * n
    for t, h in doors + outers:
        outdeg[t] += 1
        indeg[h] += 1
    on = [i for i in range(n) if indeg[i] + outdeg[i] > 0]
    balanced = all(indeg[i] == outdeg[i] for i in on)
    simple = balanced and all(indeg[i] == 1 for i in on)
    return {
        "n_outline_verts": len(on),
        "balanced": balanced,        # 전부 in==out → 폐곡선(들)로 분해 가능
        "simple_cycles": simple,     # 전부 in==out==1 → 단순 사이클(들)
    }


def contract(n, doors, outers, interns):
    """내부 성분 supernode 축약 + 체인 병합. 반환 (n', fixed, var, n_selfloop)."""
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for u, v in interns:
        par[find(u)] = find(v)
    roots = sorted({find(i) for i in range(n)})
    rid = {r: i for i, r in enumerate(roots)}
    ns = len(roots)

    fixed = []
    for t, h in doors:
        a, b = rid[find(t)], rid[find(h)]
        if a != b:
            fixed.append((a, b))
    var = []
    n_selfloop = 0
    for t, h in outers:
        a, b = rid[find(t)], rid[find(h)]
        if a == b:
            n_selfloop += 1  # 방향 무관 → 항상 유효
        else:
            var.append((a, b))

    # 체인 병합: 고정 arc 0개 + 변수 간선 딱 2개인 노드는 직렬.
    changed = True
    while changed:
        changed = False
        deg_fix = [0] * ns
        for a, b in fixed:
            deg_fix[a] += 1
            deg_fix[b] += 1
        inc: dict[int, list[int]] = {}
        for i, (a, b) in enumerate(var):
            inc.setdefault(a, []).append(i)
            inc.setdefault(b, []).append(i)
        for node, eis in inc.items():
            if deg_fix[node] or len(eis) != 2:
                continue
            i1, i2 = eis
            if i1 == i2:
                continue  # 같은 간선 양끝(자기 loop) — 위에서 제거됨
            a1, b1 = var[i1]
            a2, b2 = var[i2]
            e1_in = (b1 == node)   # e1 canonical 이 node 로 들어옴
            e2_in = (b2 == node)
            t1 = a1 if b1 == node else b1   # e1 의 node 아닌 끝
            t2 = a2 if b2 == node else b2   # e2 의 node 아닌 끝
            if t1 == t2:
                continue  # 평행 2간선 사이클 — 병합하면 self-loop, 그대로 둠
            if e1_in == e2_in:
                continue  # canonical 이 이 노드에서 직렬 불일치 — 병합 불가
            # 직렬 유효 상태는 t1→node→t2 또는 t2→node→t1 뿐.
            # canonical(mask=0) 이 그 중 어느 쪽인지에 맞춰 병합 간선 방향 부여:
            # e1 이 들어오는 쪽이면 흐름은 t1→node→t2.
            merged = (t1, t2) if e1_in else (t2, t1)
            keep = [var[i] for i in range(len(var)) if i not in (i1, i2)]
            keep.append(merged)
            var = keep
            changed = True
            break

    # 고립 노드 제거(재번호)
    used = sorted({x for a, b in fixed + var for x in (a, b)})
    remap = {o: i for i, o in enumerate(used)}
    fixed = [(remap[a], remap[b]) for a, b in fixed]
    var = [(remap[a], remap[b]) for a, b in var]
    return len(used), fixed, var, n_selfloop


# ---------- 실행 ----------

def scan_macro(pool, n, fixed, var, n_selfloop):
    m = len(var)
    total = 1 << m
    if m <= CHUNK_BITS:
        n_ok, example = scan_chunk((n, fixed, var, 0, total))
    else:
        step = 1 << CHUNK_BITS
        jobs = [(n, fixed, var, s, min(s + step, total)) for s in range(0, total, step)]
        n_ok = 0
        example = None
        for c_ok, c_ex in pool.imap_unordered(scan_chunk, jobs):
            n_ok += c_ok
            if example is None and c_ex is not None:
                example = c_ex
    return n_ok, example, total


def main() -> None:
    results = []
    with Pool(N_PROCS) as pool:
        for seed in SEEDS:
            raw, _pts = build_delaunay(N_VERTICES, seed)
            pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
            graph = Graph(
                edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()]
            )
            np.random.seed(seed)
            random.seed(seed)
            fc = FaceClusterPartition(target_k=TARGET_K).run(graph)
            np.random.seed(seed)
            random.seed(seed)
            do = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)
            fc_dir = {
                e.id: e.vertices
                for sg in fc.sub_graphs
                for e in sg.edges.values()
                if e.directed
            }
            door_ids = {
                e.id
                for sg in do.sub_graphs
                for e in sg.edges.values()
                if e.directed
            }

            g_row = {"seed": seed, "n_vertices": pruned.number_of_nodes(),
                     "n_edges": pruned.number_of_edges(), "macros": []}
            for mi, sg in enumerate(fc.sub_graphs):
                if not sg.edges:
                    continue
                t0 = time.perf_counter()
                nv, doors, outers, interns = build_macro_model(sg, fc_dir, door_ids)
                circ = outline_circularity(nv, doors, outers)
                nc, fixed, var, n_selfloop = contract(nv, doors, outers, interns)
                n_ok, example, total = scan_macro(pool, nc, fixed, var, n_selfloop)
                # canonical(=mask 0) 유효 여부 sanity
                canon_ok, _ = scan_chunk((nc, fixed, var, 0, 1))
                p_true = (n_ok > 1) or (n_selfloop > 0 and n_ok >= 1)
                feasible_total = n_ok * (2 ** n_selfloop)
                row = {
                    "macro": mi, "nv": nv,
                    "n_door": len(doors), "n_outer": len(outers),
                    "n_intern": len(interns),
                    "outline": circ,
                    "n_super": nc, "n_var_after": len(var),
                    "n_selfloop": n_selfloop,
                    "canonical_feasible": bool(canon_ok),
                    "n_feasible_outer": feasible_total,
                    "p_flip_exists": bool(p_true),
                    "example_mask": example,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }
                g_row["macros"].append(row)
                print(f"seed {seed} macro {mi}: nv={nv} d={len(doors)} "
                      f"o={len(outers)} i={len(interns)} | super={nc} "
                      f"var={len(var)} loop={n_selfloop} | outline "
                      f"balanced={circ['balanced']} simple={circ['simple_cycles']} "
                      f"| feasible={feasible_total}/{total * 2**n_selfloop} "
                      f"P={'TRUE' if p_true else 'false'} "
                      f"({row['elapsed_s']}s)", flush=True)
            results.append(g_row)

    with open(OUT, "w") as f:
        json.dump({"n_vertices": N_VERTICES, "target_k": TARGET_K,
                   "remove_ratio": REMOVE_RATIO, "seeds": SEEDS,
                   "graphs": results}, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
