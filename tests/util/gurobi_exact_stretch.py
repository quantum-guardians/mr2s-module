"""exact stretch MIP — Gurobi 백엔드 (학술 라이선스, 변수 상한 없음).

단독 실행 파일 — 저장소 밖에서도 이 파일 하나로 돈다.
필요 패키지: pip install gurobipy networkx numpy scipy

Gurobi 라이선스 — pip 에 딸려오는 trial 은 변수 2,000개 상한이라 부족하다
(V=200 이면 flow 변수만 약 12만 개). 실제 라이선스 발급 절차:
  1. 학술 named-user (개인 PC 권장):
     https://portal.gurobi.com 에 대학 메일로 가입 → Licenses →
     "Named-User Academic" 발급 → 대학 네트워크(또는 VPN)에 물린 상태로
     `grbgetkey <발급된 키>` 실행 → ~/gurobi.lic 생성. 환경변수 불필요.
     비표준 경로에 두는 경우만 GRB_LICENSE_FILE=<경로> 설정.
  2. 학술 WLS (서버·컨테이너 권장, 어디서든 동작):
     포털에서 "WLS Academic" 발급 → WLSACCESSID/WLSSECRET/LICENSEID
     3줄짜리 gurobi.lic 다운로드 → ~/gurobi.lic 에 두거나 같은 이름의
     환경변수(GRB_WLSACCESSID 등)로 설정.
  gurobi.lic 과 WLS 키는 절대 저장소에 커밋하지 않는다.

정식화 (stretch_exact_mip.py, 커밋 0fb6368 과 동일):
  x[e] ∈ {0,1}  — 무방향 간선 e 의 방향 비트 (0: u→v, 1: v→u)
  f[s,a] ≥ 0    — 출발점 s 의 호 a 위 유량
  목적          — Σ cost(a)·f[s,a],  cost = 1/w
  게이팅        — f[s,a] ≤ TD(s)·(1−x[e]) 또는 TD(s)·x[e]
  보존          — 출발점 s 는 TD(s) 를 뿜고 각 t 는 alpha(s,t)=1/D(s,t) 를 흡수
고정된 x 에서 최소비용유량은 각 수요를 최단경로로 보내므로 목적값이 정확히
Σ d→(s,t)/D(s,t) 가 된다. 강연결은 infeasibility 로 공짜 하드 제약.

MIP start 로 Robbins(DFS 트리 전방 + 백에지 후방) 방향화를 넣는다. bridgeless
그래프에서 항상 강연결이므로 즉시 유효한 incumbent 가 생긴다.

실행 (가만히 둬도 아래 기록이 전부 자동으로 남는다):
    PYTHONHASHSEED=0 python gurobi_exact_stretch.py --v 200 --seeds 0,1,2

기록 (--out-dir, 기본 gurobi_runs/):
  results_v{V}.json           — 시드 하나 끝날 때마다 갱신되는 전체 결과
  gurobi_v{V}_s{seed}.log     — Gurobi 진행 로그 (실시간 tail 가능)
  incumbent_v{V}_s{seed}.json — 새 최선해가 나올 때마다 덮어쓰는 방향화.
                                프로세스가 죽어도 마지막 최선해가 남고,
                                --warm-start 로 넣어 이어서 돌릴 수 있다.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

REMOVE_RATIO = 0.4  # door_combo_scan.REMOVE_RATIO 와 반드시 동일


def build_delaunay(n_points: int, seed: int) -> nx.Graph:
    """무작위 2D 점의 Delaunay 삼각분할 → 무방향 평면 그래프."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))
    triangulation = Delaunay(points)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_points))
    for simplex in triangulation.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            graph.add_edge(a, b)
    return graph


def prune_keep_biconnected(
    graph: nx.Graph, removal_ratio: float, seed: int
) -> nx.Graph:
    """biconnected(=bridgeless) 를 유지하며 간선을 무작위 제거.

    목표 비율 도달, 또는 더 제거하면 biconnected 가 깨질 때 중단.
    """
    graph = graph.copy()
    rng = np.random.default_rng(seed + 1)
    edges = list(graph.edges())
    rng.shuffle(edges)
    target_removed = int(graph.number_of_edges() * removal_ratio)
    removed = 0
    for u, v in edges:
        if removed >= target_removed:
            break
        graph.remove_edge(u, v)
        if nx.is_biconnected(graph):
            removed += 1
        else:
            graph.add_edge(u, v)  # bridge/절단점 생김 → 되돌림
    return graph


def build_graph(n: int, seed: int) -> nx.Graph:
    """벤치와 동일한 무방향 그래프. 위상은 (n, seed, REMOVE_RATIO) 로 완전 결정."""
    return prune_keep_biconnected(build_delaunay(n, seed), REMOVE_RATIO, seed)


def robbins_orientation(graph: nx.Graph) -> dict[tuple[int, int], tuple[int, int]]:
    """DFS 트리 전방 + 백에지 후방 → bridgeless 그래프에서 강연결 (Robbins).

    반환: {정렬된 간선 키 (u,v) with u<v: (tail, head)}
    """
    root = next(iter(graph.nodes()))
    visit_order: dict[int, int] = {}
    parent: dict[int, int] = {}
    # 방문 표시는 pop 시점에 해야 진짜 DFS 트리가 된다. push 시점에 표시하면
    # cross edge 가 생겨 "비트리 간선 = 조상-자손" 전제가 깨지고 강연결이 무너진다.
    stack: list[tuple[int | None, int]] = [(None, root)]
    while stack:
        par, node = stack.pop()
        if node in visit_order:
            continue
        visit_order[node] = len(visit_order)
        if par is not None:
            parent[node] = par
        for neighbor in graph.neighbors(node):
            if neighbor not in visit_order:
                stack.append((node, neighbor))

    directions: dict[tuple[int, int], tuple[int, int]] = {}
    for u, v in graph.edges():
        key = (u, v) if u < v else (v, u)
        if parent.get(v) == u:
            directions[key] = (u, v)
        elif parent.get(u) == v:
            directions[key] = (v, u)
        else:  # 백에지 — 깊은 쪽에서 얕은 쪽으로
            directions[key] = (u, v) if visit_order[u] > visit_order[v] else (v, u)
    return directions


def solve(
    graph: nx.Graph,
    time_limit: float,
    threads: int,
    log_path: str,
    incumbent_path: str,
    buckets: int = 1,
    mip_focus: int = 0,
    cuts: int = -1,
    presolve: int = -1,
    nodefile_gb: float = 8.0,
    nodefile_dir: str = "",
    warm_start: str = "",
) -> dict:
    """buckets>1 이면 목적지를 거리순 G개 버킷으로 쪼개 big-M 을 타이트하게 만든다.

    aggregated 정식화의 M = TD(s) 는 헐거워서 LP 완화가 x=0.5 로 양방향 반씩
    흘리는 걸 싸게 허용한다 (V=200 루트 갭 24%의 원인). 커모디티를 (s, 버킷)으로
    두면 M 이 버킷 수요합으로 준다. 완전 disaggregation 은 V=200 에서 변수가
    2,800만이라 불가능. 모든 G 에 대해 유효한 하한이다.
    """
    import gurobipy as gp  # pyright: ignore[reportMissingImports]
    from gurobipy import GRB  # pyright: ignore[reportMissingImports]

    vertices = sorted(graph.nodes())
    num_vertices = len(vertices)
    pair_count = num_vertices * (num_vertices - 1)

    # 거리 = 1/w, w=1 이므로 홉 거리. 수요 alpha(s,t) = 1/D(s,t).
    hops = dict(nx.all_pairs_shortest_path_length(graph))

    edges = [(u, v) if u < v else (v, u) for u, v in graph.edges()]
    edge_index = {edge: i for i, edge in enumerate(edges)}
    # 호: (edge_idx, tail, head, active_bit). active_bit == x 값일 때 열린다.
    arcs: list[tuple[int, int, int, int]] = []
    for edge_idx, (u, v) in enumerate(edges):
        arcs.append((edge_idx, u, v, 0))
        arcs.append((edge_idx, v, u, 1))
    arc_count = len(arcs)

    out_arcs: dict[int, list[int]] = {v: [] for v in vertices}
    in_arcs: dict[int, list[int]] = {v: [] for v in vertices}
    for arc_idx, (_edge_idx, tail, head, _bit) in enumerate(arcs):
        out_arcs[tail].append(arc_idx)
        in_arcs[head].append(arc_idx)

    model = gp.Model("exact_stretch")
    model.setParam("LogFile", log_path)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 0.0)
    if threads:
        model.setParam("Threads", threads)
    if mip_focus:
        model.setParam("MIPFocus", mip_focus)
    if cuts >= 0:
        model.setParam("Cuts", cuts)
    if presolve >= 0:
        model.setParam("Presolve", presolve)
    if nodefile_gb > 0:
        # 탐색 트리가 임계치를 넘으면 디스크로 스필 — 장시간 무인 실행에서
        # OOM 으로 죽는 대신 느려지기만 한다.
        model.setParam("NodefileStart", nodefile_gb)
        if nodefile_dir:
            model.setParam("NodefileDir", nodefile_dir)

    # 커모디티 = (출발점, 목적지 버킷). buckets=1 이면 기존 aggregated 정식화.
    commodities: list[tuple[int, dict[int, float]]] = []
    for source in vertices:
        hops_from_source = hops[source]
        targets = sorted(
            (t for t in vertices if t != source), key=lambda t: hops_from_source[t]
        )
        bucket_size = -(-len(targets) // buckets)
        for chunk_start in range(0, len(targets), bucket_size):
            bucket_targets = targets[chunk_start : chunk_start + bucket_size]
            commodities.append(
                (source, {t: 1.0 / hops_from_source[t] for t in bucket_targets})
            )
    num_commodities = len(commodities)

    x = model.addVars(len(edges), vtype=GRB.BINARY, name="x")
    flow = model.addVars(num_commodities, arc_count, lb=0.0, name="f")

    build_start = time.perf_counter()
    model.setObjective(
        gp.quicksum(
            flow[c, a] for c in range(num_commodities) for a in range(arc_count)
        ),
        GRB.MINIMIZE,
    )

    for com_idx, (source, demand) in enumerate(commodities):
        total_demand = sum(demand.values())
        for arc_idx, (edge_idx, _tail, _head, active_bit) in enumerate(arcs):
            gate = x[edge_idx] if active_bit == 1 else (1 - x[edge_idx])
            model.addConstr(flow[com_idx, arc_idx] <= total_demand * gate)
        for vertex in vertices:
            balance = total_demand if vertex == source else -demand.get(vertex, 0.0)
            model.addConstr(
                gp.quicksum(flow[com_idx, a] for a in out_arcs[vertex])
                - gp.quicksum(flow[com_idx, a] for a in in_arcs[vertex])
                == balance
            )
    model.update()
    build_s = time.perf_counter() - build_start

    # MIP start: Robbins 방향화 (bridgeless → 항상 강연결이므로 반드시 feasible)
    robbins_dirs = robbins_orientation(graph)
    for edge, (tail, _head) in robbins_dirs.items():
        x[edge_index[edge]].Start = 0 if tail == edge[0] else 1

    # 이전 실행의 incumbent 를 시작해로 — Robbins 에서 재출발하면 좋은 해를
    # 다시 찾는 데 수천 초를 쓴다. 그 시간을 전부 하한 조이기로 돌린다.
    if warm_start:
        with open(warm_start, encoding="utf-8") as fh:
            loaded = json.load(fh)
        arc_by_pair = {frozenset(arc): tuple(arc) for arc in loaded["orientation"]}
        for i, edge in enumerate(edges):
            arc = arc_by_pair.get(frozenset(edge))
            if arc is None:
                raise KeyError(f"warm start 에 간선 {edge} 가 없다 — 다른 그래프다")
            x[i].Start = 0 if arc == edge else 1
        print(
            f"  warm start: {warm_start} "
            f"(obj_avg={loaded.get('obj_avg')}, 간선 {len(loaded['orientation'])})",
            flush=True,
        )

    # 프로세스가 죽으면 그때까지 찾은 방향화를 잃으므로 새 incumbent 마다 덤프.
    # incumbent 는 단조 개선이라 덮어쓰기만 해도 항상 최선이 남는다.
    def _dump_incumbent(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        values = cb_model.cbGetSolution(x_vars)
        payload = {
            "vertices": num_vertices,
            "pair_count": pair_count,
            "obj_total": float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)),
            "bound_total": float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJBND)),
            "runtime_s": float(cb_model.cbGet(GRB.Callback.RUNTIME)),
            "orientation": [
                (edge[1], edge[0]) if values[i] > 0.5 else edge
                for i, edge in enumerate(edges)
            ],
        }
        payload["obj_avg"] = payload["obj_total"] / pair_count
        payload["bound_avg"] = payload["bound_total"] / pair_count
        with open(incumbent_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    x_vars = [x[i] for i in range(len(edges))]
    solve_start = time.perf_counter()
    model.optimize(_dump_incumbent)
    runtime = time.perf_counter() - solve_start

    has_sol = model.SolCount > 0
    ub = float(model.ObjVal) if has_sol else None
    lb = float(model.ObjBound)
    if model.Status == GRB.OPTIMAL:
        status = "optimal"
    elif has_sol:
        status = "bound-only"
    else:
        status = "no-incumbent"

    return {
        "vertices": num_vertices,
        "edges": len(edges),
        "arcs": arc_count,
        "buckets": buckets,
        "commodities": num_commodities,
        "orientation": (
            [
                (edge[1], edge[0]) if x[i].X > 0.5 else edge
                for i, edge in enumerate(edges)
            ]
            if has_sol
            else None
        ),
        "num_vars": model.NumVars,
        "num_constraints": model.NumConstrs,
        "pair_count": pair_count,
        "status": status,
        "gurobi_status": int(model.Status),
        "lb_total": lb,
        "ub_total": ub,
        "lb_avg": lb / pair_count,
        "ub_avg": ub / pair_count if ub is not None else None,
        "gap_pct": (100.0 * (ub - lb) / abs(ub)) if ub else None,
        "node_count": float(model.NodeCount),
        "build_s": build_s,
        "runtime_s": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=int, default=200)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default="gurobi_runs",
        help="결과·로그·incumbent 가 전부 여기 쌓인다",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=1,
        help="목적지 버킷 수 G. 클수록 big-M 이 타이트하고 변수는 G배",
    )
    parser.add_argument("--mipfocus", type=int, default=0, help="3 = dual bound 집중")
    parser.add_argument("--cuts", type=int, default=-1, help="3 = 공격적 컷")
    parser.add_argument("--presolve", type=int, default=-1, help="2 = 공격적 presolve")
    parser.add_argument(
        "--nodefile-gb",
        type=float,
        default=8.0,
        help="탐색 트리가 이 GB 를 넘으면 디스크로 스필. 0 = 끔",
    )
    parser.add_argument(
        "--warm-start",
        default="",
        help="이전 incumbent JSON 을 MIP start 로 사용 (시드 1개일 때만)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"results_v{args.v}.json")

    seeds = [int(s) for s in args.seeds.split(",")]
    rows: list[dict] = []
    for seed in seeds:
        graph = build_graph(args.v, seed)
        print(
            f"\n=== V={args.v} seed={seed}: 정점 {graph.number_of_nodes()} "
            f"간선 {graph.number_of_edges()} "
            f"bridgeless={nx.is_biconnected(graph)} ===",
            flush=True,
        )
        row: dict
        try:
            row = solve(
                graph,
                args.time_limit,
                args.threads,
                log_path=os.path.join(args.out_dir, f"gurobi_v{args.v}_s{seed}.log"),
                incumbent_path=os.path.join(
                    args.out_dir, f"incumbent_v{args.v}_s{seed}.json"
                ),
                buckets=args.buckets,
                mip_focus=args.mipfocus,
                cuts=args.cuts,
                presolve=args.presolve,
                nodefile_gb=args.nodefile_gb,
                nodefile_dir=args.out_dir,
                warm_start=args.warm_start,
            )
        except Exception as exc:
            row = {"status": "error", "message": str(exc)[:300]}
            print(f"  실패: {exc}", flush=True)
        row["target_n"] = args.v
        row["seed"] = seed
        row["time_limit_s"] = args.time_limit
        rows.append(row)
        if row["status"] != "error":
            print(
                f"  {row['status']} | 변수 {row['num_vars']:,} "
                f"제약 {row['num_constraints']:,} | "
                f"lb_avg={row['lb_avg']:.4f} "
                f"ub_avg={row['ub_avg'] if row['ub_avg'] is None else round(row['ub_avg'], 4)} "
                f"gap={row['gap_pct']} ({row['runtime_s']:.0f}s, "
                f"빌드 {row['build_s']:.0f}s)",
                flush=True,
            )
        # 시드 하나 끝날 때마다 저장 — 중간에 죽어도 끝난 시드는 남는다.
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"v": args.v, "rows": rows}, fh)

    print(f"\nwrote {out_path}")
    print(
        f"\n{'seed':>5} {'상태':>13} {'lb_avg':>9} {'ub_avg':>9} {'gap%':>8} {'초':>7}"
    )
    for row in rows:
        if row["status"] == "error":
            print(f"{row['seed']:>5} {'error':>13}")
            continue
        ub_str = f"{row['ub_avg']:.4f}" if row["ub_avg"] is not None else "-"
        gap_str = f"{row['gap_pct']:.2f}" if row["gap_pct"] is not None else "-"
        print(
            f"{row['seed']:>5} {row['status']:>13} {row['lb_avg']:>9.4f} "
            f"{ub_str:>9} {gap_str:>8} {row['runtime_s']:>7.0f}"
        )


if __name__ == "__main__":
    main()
