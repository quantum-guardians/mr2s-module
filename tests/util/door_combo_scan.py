"""door 방향 조합 전수 스캔 (ISSUE-52 door 패턴 분석).

DoorOnlyFaceClusterPartition(target_k=10) 이 뽑은 door(인접 macro 공유 경계)의
방향 조합을 바꿔가며 프로덕션과 같은 DnC+SA 솔버를 돌리고, 전역 stretch
(apsp_sum)가 어떻게 달라지는지 조합마다 기록한다.

사전 실험에서 확인한 구조:
  - door 부분그래프는 전 정점 짝수 차수 = 닫힌 곡선들의 합집합.
  - 균등 랜덤 방향 조합은 per-macro 강연결(mixed-SC)이 사실상 항상 깨진다
    (0/2000). 살아남는 조합은 "닫힌 door 루프 통째 반전" = 사이클공간 원소.
  - 그래서 전수 대상은 사이클공간(2^dim, dim=7~9) 전체 + 해밍 1·2·3 스윕.

프리필터: 조합마다 macro 하나씩 mixed-SC 판정(door=고정 arc, 나머지=양방향).
door 많은 macro 부터 검사해 조기 탈락. 전 macro 통과한 조합만 솔버를 돌린다.

    PYTHONHASHSEED=0 python tests/util/door_combo_scan.py [seeds]

출력: tests/util/door_combo_scan.json (export_door_combo_report.py 가 HTML 생성)
실험 전용 — 프로덕션(mr2s_module/)에 두지 않는다.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from collections import Counter, defaultdict, deque
from multiprocessing import Pool

import numpy as np

from mr2s_module.cycle.face_clusterer import KMeansFaceClusterer
from mr2s_module.domain import Edge, Graph
from mr2s_module.evaluator import Evaluator
from mr2s_module.reduction import ReductionMr2sSolver
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver
from mr2s_module.solver.partition.vertex_count import VertexCountPartitionStrategy
from mr2s_module.solver.sa_mr2s_solver import SAMR2SSolver
from tests.util.door_only_partition import DoorOnlyFaceClusterPartition
from tests.util.fixed_door_partition import FixedDoorFaceClusterPartition
from tests.util.robbins_door_bruteforce import build_delaunay, prune_keep_biconnected

SEEDS = [0, 1, 2]
N_VERTICES = 200
REMOVE_RATIO = 0.4
TARGET_K = 10
MAX_VERTICES = 40
N_PROCS = 14
HAMMING_TRIPLE_BUDGET = 180  # 해밍3 스윕 상한(초/seed). 0 이면 생략
HAMMING_QUAD_BUDGET = 300    # 해밍4 스윕 상한(초/seed). 0 이면 생략
PERTURB_BUDGET = 240         # feasible 조합 ⊕ 해밍1·2 섭동 상한(초/seed). 0 이면 생략
MAX_SOLVER_COMBOS = 700      # seed 당 솔버 실행 상한 (초과 시 사이클공간 우선 + 균등 샘플)
OUT = "tests/util/door_combo_scan.json"


# ---------------------------------------------------------------------------
# [케이스 구성] 그래프 → macro/door 스냅샷 (파티션은 부모에서 1회만)
# ---------------------------------------------------------------------------
def build_case(seed: int) -> dict:
    raw, pts = build_delaunay(N_VERTICES, seed)
    pruned = prune_keep_biconnected(raw, REMOVE_RATIO, seed)
    graph = Graph(edges=[Edge(int(u), int(v), 1, False) for u, v in pruned.edges()])

    np.random.seed(seed)
    random.seed(seed)
    partition = DoorOnlyFaceClusterPartition(target_k=TARGET_K).run(graph)

    macros: list[dict] = []
    canon: dict[int, tuple[int, int]] = {}
    owner: dict[int, set[int]] = defaultdict(set)
    for macro_idx, sub_graph in enumerate(partition.sub_graphs):
        doors: list[int] = []
        free: list[tuple[int, int, int]] = []
        verts: set[int] = set()
        for edge in sub_graph.edges.values():
            verts.update(edge.endpoints())
            if edge.directed:
                doors.append(edge.id)
                canon[edge.id] = edge.vertices
                owner[edge.id].add(macro_idx)
            else:
                free.append((edge.id, *edge.endpoints()))
        macros.append({"verts": sorted(verts), "doors": sorted(doors), "free": free})

    door_ids = sorted(canon)
    return {
        "seed": seed,
        "edges": [(edge.id, *edge.endpoints()) for edge in graph.edges.values()],
        "pos": {int(i): [float(p[0]), float(p[1])] for i, p in enumerate(pts)},
        "macros": macros,
        "canon": canon,
        "owner": {eid: sorted(owner[eid]) for eid in door_ids},
        "door_ids": door_ids,
        "bit_of": {eid: i for i, eid in enumerate(door_ids)},
    }


def rebuild_graph(case: dict) -> Graph:
    """id 를 보존한 채 원본 무방향 그래프 재구성 (조합마다 새 인스턴스 필요)."""
    edges = []
    for eid, u, v in case["edges"]:
        edge = Edge(u, v, 1, False)
        edge.id = eid
        edges.append(edge)
    return Graph(edges=edges)


# ---------------------------------------------------------------------------
# [프리필터] macro 단위 mixed-SC 판정
# ---------------------------------------------------------------------------
def _mixed_sc(verts: list[int], arcs, free) -> bool:
    index = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    fwd: list[list[int]] = [[] for _ in range(n)]
    bwd: list[list[int]] = [[] for _ in range(n)]

    def add(a: int, b: int) -> None:
        fwd[index[a]].append(index[b])
        bwd[index[b]].append(index[a])

    for a, b in arcs:
        add(a, b)
    for _, u, v in free:
        add(u, v)
        add(v, u)

    def reach(adj: list[list[int]]) -> int:
        seen = {0}
        queue = deque([0])
        while queue:
            x = queue.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        return len(seen)

    return reach(fwd) == n and reach(bwd) == n


class Feasibility:
    """조합 비트 → per-macro mixed-SC 전원 통과 여부. door 많은 macro 부터."""

    def __init__(self, case: dict):
        self.macros = case["macros"]
        self.canon = case["canon"]
        self.bit_of = case["bit_of"]
        self.order = sorted(
            range(len(self.macros)),
            key=lambda i: -len(self.macros[i]["doors"]),
        )

    def first_failure(self, bits: int) -> int:
        """전부 통과면 -1, 아니면 처음 실패한 macro index."""
        for i in self.order:
            macro = self.macros[i]
            arcs = []
            for eid in macro["doors"]:
                tail, head = self.canon[eid]
                if (bits >> self.bit_of[eid]) & 1:
                    arcs.append((head, tail))
                else:
                    arcs.append((tail, head))
            if not _mixed_sc(macro["verts"], arcs, macro["free"]):
                return i
        return -1

    def ok(self, bits: int) -> bool:
        return self.first_failure(bits) < 0


# ---------------------------------------------------------------------------
# [조합 열거] door 그래프 사이클공간 + 해밍 스윕
# ---------------------------------------------------------------------------
def cycle_space_basis(case: dict) -> list[int]:
    """door 그래프의 기본 사이클(fundamental cycle) 비트마스크. dim = E - V + C."""
    canon = case["canon"]
    bit_of = case["bit_of"]
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for eid in case["door_ids"]:
        u, v = canon[eid]
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    parent: dict[int, tuple[int, int] | None] = {}
    depth: dict[int, int] = {}
    tree_edges: set[int] = set()
    for root in list(adj):
        if root in parent:
            continue
        parent[root] = None
        depth[root] = 0
        queue = deque([root])
        while queue:
            x = queue.popleft()
            for y, eid in adj[x]:
                if y not in parent:
                    parent[y] = (x, eid)
                    depth[y] = depth[x] + 1
                    tree_edges.add(eid)
                    queue.append(y)

    def path_bits(a: int, b: int) -> int:
        bits = 0
        while a != b:
            if depth[a] < depth[b]:
                a, b = b, a
            px, peid = parent[a]  # type: ignore[misc]
            bits |= 1 << bit_of[peid]
            a = px
        return bits

    basis: list[int] = []
    for eid in case["door_ids"]:
        if eid in tree_edges:
            continue
        u, v = canon[eid]
        basis.append((1 << bit_of[eid]) ^ path_bits(u, v))
    return basis


def enumerate_combos(case: dict, feas: Feasibility) -> tuple[list[dict], dict]:
    """조합 후보 열거 + 프리필터. (feasible 목록, 통계) 반환."""
    door_count = len(case["door_ids"])
    basis = cycle_space_basis(case)
    stats: dict = {"cycle_dim": len(basis), "doors": door_count}

    seen: set[int] = set()
    feasible: list[dict] = []
    tested = 0
    t_start = time.perf_counter()

    def consider(bits: int, source: str) -> None:
        nonlocal tested
        if bits in seen:
            return
        seen.add(bits)
        tested += 1
        if feas.first_failure(bits) < 0:
            feasible.append({"bits": bits, "source": source})

    # 1) 사이클공간 전수
    for mask in range(1 << len(basis)):
        bits = 0
        for i, cycle in enumerate(basis):
            if (mask >> i) & 1:
                bits ^= cycle
        consider(bits, "cycle_space")
    stats["cycle_space_total"] = 1 << len(basis)
    stats["cycle_space_feasible"] = len(feasible)

    # 2) 해밍 1
    before = len(feasible)
    for i in range(door_count):
        consider(1 << i, "hamming1")
    stats["hamming1_extra"] = len(feasible) - before

    # 3) 해밍 2
    before = len(feasible)
    for i, j in itertools.combinations(range(door_count), 2):
        consider((1 << i) | (1 << j), "hamming2")
    stats["hamming2_extra"] = len(feasible) - before

    # 4) 해밍 3 (시간 예산 안에서)
    before = len(feasible)
    checked3 = 0
    if HAMMING_TRIPLE_BUDGET > 0:
        deadline = time.perf_counter() + HAMMING_TRIPLE_BUDGET
        for i, j, k in itertools.combinations(range(door_count), 3):
            consider((1 << i) | (1 << j) | (1 << k), "hamming3")
            checked3 += 1
            if (checked3 & 0x3FF) == 0 and time.perf_counter() > deadline:
                break
    stats["hamming3_checked"] = checked3
    stats["hamming3_total"] = door_count * (door_count - 1) * (door_count - 2) // 6
    stats["hamming3_extra"] = len(feasible) - before

    # 5) 해밍 4 (시간 예산 안에서)
    before = len(feasible)
    checked4 = 0
    if HAMMING_QUAD_BUDGET > 0:
        deadline = time.perf_counter() + HAMMING_QUAD_BUDGET
        for quad in itertools.combinations(range(door_count), 4):
            bits = 0
            for i in quad:
                bits |= 1 << i
            consider(bits, "hamming4")
            checked4 += 1
            if (checked4 & 0x3FF) == 0 and time.perf_counter() > deadline:
                break
    stats["hamming4_checked"] = checked4
    stats["hamming4_extra"] = len(feasible) - before

    # 6) 지금까지 찾은 feasible 조합 ⊕ 해밍1·2 섭동 (feasible 근방 훑기)
    before = len(feasible)
    checked_p = 0
    if PERTURB_BUDGET > 0:
        deadline = time.perf_counter() + PERTURB_BUDGET
        seeds_for_perturb = [entry["bits"] for entry in list(feasible)]
        stop = False
        for anchor in seeds_for_perturb:
            for i in range(door_count):
                consider(anchor ^ (1 << i), "perturb1")
                checked_p += 1
            for i, j in itertools.combinations(range(door_count), 2):
                consider(anchor ^ (1 << i) ^ (1 << j), "perturb2")
                checked_p += 1
                if (checked_p & 0x3FF) == 0 and time.perf_counter() > deadline:
                    stop = True
                    break
            if stop:
                break
    stats["perturb_checked"] = checked_p
    stats["perturb_extra"] = len(feasible) - before

    stats["prefilter_tested"] = tested
    stats["prefilter_sec"] = time.perf_counter() - t_start
    stats["feasible_total"] = len(feasible)
    return feasible, stats


# ---------------------------------------------------------------------------
# [솔버] 조합마다 프로덕션과 동일한 DnC+SA 1회
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _worker_init(cases: dict) -> None:
    _WORKER["cases"] = cases


def build_solver(seed: int, cap: int) -> ReductionMr2sSolver:
    """door_vs_facecycle_solver_bench 와 같은 구성 + door 고정 보존 래퍼."""
    sa = SAMR2SSolver(
        sweeps_per_temperature=2,
        num_restarts=4,
        random_seed=seed,
    )
    face_cycle = FixedDoorFaceClusterPartition(
        target_k=2, clusterer=KMeansFaceClusterer()
    )
    dnc = DnCMr2sSolver(
        mr2s_solver=sa,
        face_cycle=face_cycle,
        graph_partition_strategy=VertexCountPartitionStrategy(
            face_cycle=face_cycle,
            max_vertices=cap,
        ),
        subgraph_processes=1,  # 바깥 Pool 과 중첩 방지
    )
    return ReductionMr2sSolver(mr2s_solver=dnc, evaluator=Evaluator())


def solve_combo_task(args: tuple[int, int]) -> dict:
    seed, bits = args
    case = _WORKER["cases"][seed]
    canon = case["canon"]
    bit_of = case["bit_of"]

    # 전역 Edge id 카운터를 조합마다 같은 값으로 되돌린다. 솔버 내부에서 새로
    # 만들어지는 Edge(방향 사본·축약 super edge)의 id 가 고정 id 들과 섞여
    # 정렬 순서를 바꾸므로, 리셋하지 않으면 같은 조합이 호출 순서에 따라 다른
    # 해를 낸다(실측 stretch 편차 5%, SC 성패까지 뒤집힘).
    Edge._id_counter = itertools.count(
        max(eid for eid, _u, _v in case["edges"]) + 1000
    )

    graph = rebuild_graph(case)
    fixed: dict[int, tuple[int, int]] = {}
    for eid in case["door_ids"]:
        tail, head = canon[eid]
        if (bits >> bit_of[eid]) & 1:
            tail, head = head, tail
        graph.edges[eid].set_direction(tail, head)
        fixed[eid] = (tail, head)

    np.random.seed(seed)
    random.seed(seed)
    started = time.perf_counter()
    error = ""
    score = None
    for cap in (MAX_VERTICES, 60, 80):
        try:
            solution = build_solver(seed, cap).run(graph)
            score = solution.score
            kept = sum(
                1 for eid, direction in fixed.items()
                if solution.edges.get(eid) == direction
            )
            break
        except RuntimeError as exc:
            error = str(exc)
            graph = rebuild_graph(case)
            for eid, direction in fixed.items():
                graph.edges[eid].set_direction(*direction)
    else:
        return {
            "seed": seed,
            "bits": bits,
            "failed": True,
            "error": error,
            "sec": time.perf_counter() - started,
        }

    return {
        "seed": seed,
        "bits": bits,
        "failed": False,
        "apsp_sum": float(score.apsp_sum),
        "flow_score": float(score.flow_score),
        "strong_connect_rate": float(score.strong_connect_rate),
        "door_kept": kept,
        "door_total": len(fixed),
        "sec": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# [main]
# ---------------------------------------------------------------------------
def main() -> None:
    seeds = SEEDS
    if len(sys.argv) > 1:
        seeds = [int(x) for x in sys.argv[1].split(",")]

    cases: dict[int, dict] = {}
    plans: dict[int, list[dict]] = {}
    stats: dict[int, dict] = {}
    tasks: list[tuple[int, int]] = []
    for seed in seeds:
        case = build_case(seed)
        feas = Feasibility(case)
        feasible, stat = enumerate_combos(case, feas)
        if len(feasible) > MAX_SOLVER_COMBOS:
            rng = random.Random(seed)
            cycle_rows = [f for f in feasible if f["source"] == "cycle_space"]
            rest = [f for f in feasible if f["source"] != "cycle_space"]
            rng.shuffle(rest)
            feasible = cycle_rows + rest[: max(0, MAX_SOLVER_COMBOS - len(cycle_rows))]
            stat["sampled"] = True
        cases[seed] = case
        plans[seed] = feasible
        stats[seed] = stat
        tasks.extend((seed, entry["bits"]) for entry in feasible)
        print(
            f"seed {seed}: macros={len(case['macros'])} doors={stat['doors']} "
            f"cycle_dim={stat['cycle_dim']} feasible={stat['feasible_total']} "
            f"(cycle {stat['cycle_space_feasible']}/{stat['cycle_space_total']}, "
            f"h1 +{stat['hamming1_extra']}, h2 +{stat['hamming2_extra']}, "
            f"h3 +{stat['hamming3_extra']}/{stat['hamming3_checked']}, "
            f"h4 +{stat.get('hamming4_extra', 0)}/{stat.get('hamming4_checked', 0)}, "
            f"pert +{stat.get('perturb_extra', 0)}/{stat.get('perturb_checked', 0)}) "
            f"solver={len(feasible)} prefilter={stat['prefilter_sec']:.0f}s",
            flush=True,
        )

    print(f"\nsolver runs = {len(tasks)} (procs {N_PROCS})", flush=True)
    results: dict[tuple[int, int], dict] = {}
    started = time.perf_counter()
    with Pool(N_PROCS, initializer=_worker_init, initargs=(cases,)) as pool:
        done = 0
        for row in pool.imap_unordered(solve_combo_task, tasks, 1):
            results[(row["seed"], row["bits"])] = row
            done += 1
            if done % 10 == 0 or done == len(tasks):
                rate = done / max(1e-9, time.perf_counter() - started)
                print(
                    f"  {done}/{len(tasks)} ({rate:.2f}/s, "
                    f"eta {(len(tasks) - done) / max(rate, 1e-9) / 60:.1f}m)",
                    flush=True,
                )

    payload: dict = {
        "config": {
            "n_vertices": N_VERTICES,
            "remove_ratio": REMOVE_RATIO,
            "target_k": TARGET_K,
            "max_vertices": MAX_VERTICES,
            "seeds": seeds,
        },
        "cases": {},
    }
    for seed in seeds:
        case = cases[seed]
        rows = []
        for entry in plans[seed]:
            bits = entry["bits"]
            row = dict(results[(seed, bits)])
            flipped = [
                eid for eid in case["door_ids"] if (bits >> case["bit_of"][eid]) & 1
            ]
            per_macro: Counter[int] = Counter()
            for eid in flipped:
                for macro_idx in case["owner"][eid]:
                    per_macro[macro_idx] += 1
            row["bits"] = str(bits)
            row["source"] = entry["source"]
            row["flipped"] = flipped
            row["n_flipped"] = len(flipped)
            row["per_macro_flips"] = {str(k): v for k, v in sorted(per_macro.items())}
            rows.append(row)
        payload["cases"][str(seed)] = {
            "stats": stats[seed],
            "pos": case["pos"],
            "edges": case["edges"],
            "door_ids": case["door_ids"],
            "canon": {str(k): list(v) for k, v in case["canon"].items()},
            "owner": {str(k): v for k, v in case["owner"].items()},
            "macros": case["macros"],
            "basis": [str(b) for b in cycle_space_basis(case)],
            "rows": rows,
        }

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"\nwrote {OUT}", flush=True)

    for seed in seeds:
        rows = payload["cases"][str(seed)]["rows"]
        ok = [
            r for r in rows
            if not r.get("failed") and r["strong_connect_rate"] >= 1.0
        ]
        canonical = next((r for r in rows if int(r["bits"]) == 0), None)
        if not ok:
            print(f"seed {seed}: SC 성공 조합 없음 ({len(rows)} 조합)")
            continue
        best = min(ok, key=lambda r: r["apsp_sum"])
        worst = max(ok, key=lambda r: r["apsp_sum"])
        base = canonical["apsp_sum"] if canonical and not canonical.get("failed") else float("nan")
        print(
            f"seed {seed}: SC ok {len(ok)}/{len(rows)}  canonical={base:.4f}  "
            f"best={best['apsp_sum']:.4f}(flip {best['n_flipped']})  "
            f"worst={worst['apsp_sum']:.4f}(flip {worst['n_flipped']})"
        )


if __name__ == "__main__":
    main()
