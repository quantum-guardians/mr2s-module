"""macro 별 배향 해공간 전수열거: S_fc ⊂ S_do ⊂ S_all 3팔 비교.

가정 검증 대상:
  Q1. 100-seed 솔버 벤치의 FC≈DO 동률이 "SA 가 못 찾아서"인가,
      "S_do 에 더 좋은 해가 애초에 없어서"인가.
  Q2. door 방향 조합까지 풀면(S_all) 개선 여지가 있는가. (macro 국소 열거라
      이웃 macro 정합성 무시 → S_all 이득은 개선 '상한'.)

팔 정의 (macro 부분그래프, 강연결 배향만 유효해):
  S_fc : door + 외곽 canonical 고정, 내부만 free       (face-cycle 상당)
  S_do : door canonical 고정, 외곽 + 내부 free          (door-only 상당)
  S_all: 전부 free (door 포함)                          (상한)

각 팔에서 유효 배향 수와 macro-국소 APSP 합(방향 최단경로 총합, hop) 최소를
구한다. 열거는 팔 별 free 간선 수가 ENUM_CAP 이하일 때만.

    PYTHONHASHSEED=0 python tests/util/door_combo_enum.py

출력: tests/util/door_combo_enum.json
"""

from __future__ import annotations

import json
import random
import time
from itertools import product

import numpy as np

from mr2s_module.cycle import FaceClusterPartition
from mr2s_module.domain import Edge, Graph
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

SEED = 42
REMOVE_RATIO = 0.4
ENUM_CAP = 18  # 2^18 = 262,144 조합까지
CONFIGS = [(200, 10), (100, 10)]
OUT = "tests/util/door_combo_enum.json"


def sc_and_apsp(n: int, arcs: list[tuple[int, int]]) -> float | None:
    """0..n-1 정점, arcs 방향 간선. 강연결이면 APSP 합(hop), 아니면 None."""
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
            nxt: list[int] = []
            for x in q:
                dx = dist[x]
                for y in fwd[x]:
                    if dist[y] < 0:
                        dist[y] = dx + 1
                        nxt.append(y)
                        seen += 1
            q = nxt
        if seen < n:
            return None  # src 에서 전체 미도달 → 강연결 아님
        total += sum(dist)
    return float(total)


def enumerate_arm(
    n: int,
    fixed: list[tuple[int, int]],
    free: list[tuple[int, int]],
) -> tuple[int, float | None]:
    """fixed 방향 고정 + free 2^k 열거. (유효 배향 수, 최소 APSP)."""
    n_valid = 0
    best: float | None = None
    for bits in product((0, 1), repeat=len(free)):
        arcs = list(fixed)
        for i, (u, v) in enumerate(free):
            arcs.append((u, v) if bits[i] == 0 else (v, u))
        s = sc_and_apsp(n, arcs)
        if s is None:
            continue
        n_valid += 1
        if best is None or s < best:
            best = s
    return n_valid, best


def run_config(n_vertices: int, target_k: int) -> dict:
    raw, _pts = build_delaunay(n_vertices, SEED)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, SEED)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])

    np.random.seed(SEED)
    random.seed(SEED)
    fc = FaceClusterPartition(target_k=target_k).run(graph)
    np.random.seed(SEED)
    random.seed(SEED)
    do = DoorOnlyFaceClusterPartition(target_k=target_k).run(graph)

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

    macros = []
    for mi, sg in enumerate(fc.sub_graphs):
        if not sg.edges:
            continue
        verts = sorted({v for e in sg.edges.values() for v in e.endpoints()})
        vid = {v: i for i, v in enumerate(verts)}
        n = len(verts)
        door_e, outer_e, intern_e = [], [], []
        for e in sg.edges.values():
            u, v = e.endpoints()
            if e.id in door_ids:
                t, h = fc_dir[e.id]
                door_e.append((vid[t], vid[h]))  # canonical 방향
            elif e.id in fc_dir:
                t, h = fc_dir[e.id]
                outer_e.append((vid[t], vid[h]))
            else:
                intern_e.append((vid[u], vid[v]))

        row: dict = {
            "macro": mi,
            "nv": n,
            "n_door": len(door_e),
            "n_outer": len(outer_e),
            "n_intern": len(intern_e),
        }
        t0 = time.perf_counter()
        arms = {
            "fc": (door_e + outer_e, intern_e),
            "do": (door_e, outer_e + intern_e),
            "all": ([], door_e + outer_e + intern_e),
        }
        for name, (fixed, free) in arms.items():
            if len(free) > ENUM_CAP:
                row[name] = {"skip": True, "bits": len(free)}
                continue
            n_valid, best = enumerate_arm(n, fixed, free)
            row[name] = {"skip": False, "bits": len(free), "n": n_valid, "best": best}
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        macros.append(row)
        print(f"V={n_vertices} macro {mi}: nv={n} d={len(door_e)} "
              f"o={len(outer_e)} i={len(intern_e)} -> "
              + " ".join(
                  f"{k}={'skip' if row[k].get('skip') else (row[k]['n'], row[k]['best'])}"
                  for k in ("fc", "do", "all")
              ),
              flush=True)

    return {
        "n_vertices": pruned.number_of_nodes(),
        "n_edges": pruned.number_of_edges(),
        "target_k": target_k,
        "macros": macros,
    }


def main() -> None:
    data = {
        "seed": SEED,
        "remove_ratio": REMOVE_RATIO,
        "enum_cap": ENUM_CAP,
        "configs": [run_config(nv, k) for nv, k in CONFIGS],
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
