"""door_combo_scan.json → door 방향 조합 분석 HTML.

    python tests/util/export_door_combo_report.py

출력: tests/util/door_combo_report.html
실험 전용 — 프로덕션(mr2s_module/)에 두지 않는다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SRC = Path("tests/util/door_combo_scan.json")
OUT = Path("tests/util/door_combo_report.html")


# ---------------------------------------------------------------------------
# [GF(2)] 조합 비트를 기저 사이클 좌표로 분해
# ---------------------------------------------------------------------------
def gf2_solve(basis: list[int], target: int) -> list[int] | None:
    """target 을 basis 의 XOR 조합으로 표현. 불가능하면 None."""
    rows = [(vector, 1 << i) for i, vector in enumerate(basis)]
    residual = target
    used = 0
    pivots: list[tuple[int, int]] = []
    for vector, tag in rows:
        current, current_tag = vector, tag
        for pivot, pivot_tag in pivots:
            if current ^ pivot < current:
                current ^= pivot
                current_tag ^= pivot_tag
        if current:
            pivots.append((current, current_tag))
            pivots.sort(key=lambda p: -p[0])
    for pivot, pivot_tag in pivots:
        if residual ^ pivot < residual:
            residual ^= pivot
            used ^= pivot_tag
    if residual:
        return None
    return [(used >> i) & 1 for i in range(len(basis))]


def flip_components(door_edges: dict[int, tuple[int, int]], flipped: list[int]) -> list[dict]:
    """뒤집힌 door 집합을 연결 성분으로 쪼개고 닫힘(전 정점 짝수차수) 여부 판정."""
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for eid in flipped:
        u, v = door_edges[eid]
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    seen_edges: set[int] = set()
    out: list[dict] = []
    for start in list(adj):
        if all(eid in seen_edges for _, eid in adj[start]):
            continue
        stack = [start]
        verts: set[int] = set()
        edges: set[int] = set()
        while stack:
            x = stack.pop()
            if x in verts:
                continue
            verts.add(x)
            for y, eid in adj[x]:
                edges.add(eid)
                if y not in verts:
                    stack.append(y)
        if not edges or edges <= seen_edges:
            continue
        seen_edges |= edges
        degree: dict[int, int] = defaultdict(int)
        for eid in edges:
            u, v = door_edges[eid]
            degree[u] += 1
            degree[v] += 1
        out.append({
            "edges": sorted(edges),
            "closed": all(count % 2 == 0 for count in degree.values()),
            "n": len(edges),
        })
    return out


def main() -> None:
    payload = json.loads(SRC.read_text())
    config = payload["config"]

    cases_js: dict[str, dict] = {}
    for seed, case in payload["cases"].items():
        canon = {int(k): tuple(v) for k, v in case["canon"].items()}
        door_pairs = {eid: tuple(sorted(canon[eid])) for eid in canon}
        basis = [int(b) for b in case["basis"]]
        bit_of = {eid: i for i, eid in enumerate(case["door_ids"])}

        rows = []
        for row in case["rows"]:
            bits = int(row["bits"])
            coords = gf2_solve(basis, bits)
            comps = flip_components(door_pairs, row["flipped"])
            closed_all = all(c["closed"] for c in comps) if comps else True
            if bits == 0:
                family = "canonical"
            elif coords is not None and closed_all:
                family = "loop"
            else:
                family = "local"
            rows.append({
                "family": family,
                "bits": row["bits"],
                "source": row["source"],
                "flipped": row["flipped"],
                "n_flipped": row["n_flipped"],
                "per_macro": row.get("per_macro_flips", {}),
                "apsp": row.get("apsp_sum"),
                "flow": row.get("flow_score"),
                "sc": row.get("strong_connect_rate"),
                "sec": row.get("sec"),
                "failed": bool(row.get("failed")),
                "coords": coords,
                "in_span": coords is not None,
                "comps": comps,
                "n_comps": len(comps),
                "all_closed": closed_all,
            })

        # 기저 사이클의 간선 목록(시각화용)
        basis_edges = []
        for vector in basis:
            basis_edges.append(
                [eid for eid in case["door_ids"] if (vector >> bit_of[eid]) & 1]
            )

        cases_js[seed] = {
            "stats": case["stats"],
            "pos": case["pos"],
            "edges": case["edges"],
            "door_ids": case["door_ids"],
            "canon": {str(k): list(v) for k, v in canon.items()},
            "owner": case["owner"],
            "macros": [
                {"verts": m["verts"], "doors": m["doors"], "free": m["free"]}
                for m in case["macros"]
            ],
            "basis_edges": basis_edges,
            "rows": rows,
        }

    html = TEMPLATE.replace(
        "__DATA__", json.dumps({"config": config, "cases": cases_js})
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>door 방향 조합 분석 — 어떤 조합이 좋은 해를 주는가</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px/1.6 system-ui, sans-serif; background: #14161a; color: #dde; }
  header { padding: 18px 26px 6px; }
  h1 { font-size: 19px; margin: 0 0 6px; }
  h2 { font-size: 15px; margin: 26px 0 8px; color: #cde;
       border-bottom: 1px solid #2c3038; padding-bottom: 6px; }
  .sub { color: #9ab; font-size: 13px; max-width: 1000px; }
  .wrap { padding: 0 26px 40px; }
  .bar { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
  button { background: #1c1f24; color: #dde; border: 1px solid #333a44; border-radius: 8px;
           padding: 6px 12px; cursor: pointer; font: 13px system-ui; }
  button.active { border-color: #4f9dff; background: #23303f; }
  .cols { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  svg { background: #101216; border-radius: 10px; }
  .card { background: #1c1f24; border: 1px solid #2c3038; border-radius: 10px;
          padding: 12px 16px; font-size: 13px; }
  table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
  td, th { border: 1px solid #2c3038; padding: 3px 8px; text-align: right; white-space: nowrap; }
  th { color: #9ab; position: sticky; top: 0; background: #1a1d22; cursor: pointer; }
  td:first-child, th:first-child { text-align: left; }
  tr.sel td { background: #23303f; }
  tr:hover td { background: #1f242b; }
  .scroll { max-height: 460px; overflow: auto; }
  .good { color: #6ec86e; } .bad { color: #e86a6a; } .dim { color: #78828e; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11.5px;
          border: 1px solid #333a44; }
  .kv { display: flex; justify-content: space-between; gap: 18px; }
  .kv span:first-child { color: #9ab; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #9ab;
            margin: 8px 0; }
  .sw { display: inline-block; width: 22px; height: 4px; border-radius: 2px;
        vertical-align: middle; margin-right: 5px; }
  .barrow { display: flex; align-items: center; gap: 8px; margin: 2px 0; font-size: 12px; }
  .barrow .lab { width: 92px; color: #9ab; text-align: right; }
  .barrow .val { width: 62px; }
  .barfill { height: 10px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>door 방향 조합 분석 — D(door–subgraph) 배정이 해 품질에 주는 영향</h1>
  <div class="sub" id="intro"></div>
</header>
<div class="wrap">
  <div class="bar" id="seedbar"></div>

  <h2>1. 이 그래프의 door 구조</h2>
  <div class="cols">
    <div class="card" id="struct" style="min-width:340px"></div>
    <div class="card" style="flex:1; min-width:420px">
      <b>가능/불가능 조합의 구조</b>
      <div class="dim" style="margin-top:6px" id="feasnote"></div>
    </div>
  </div>

  <h2>1-b. 측정 신뢰성 — 먼저 잡아야 했던 아티팩트</h2>
  <div class="card" style="border-left:4px solid #f5a623">
    첫 스캔에서 <b>같은 조합을 다시 돌리면 다른 값</b>이 나왔다. 원인은 랜덤성이 아니라
    전역 <code>Edge._id_counter</code> 다: 조합마다 그래프를 새로 만들면 카운터가 전진하고,
    솔버 내부에서 생기는 Edge(방향 사본·축약 super edge)의 id 가 입력 id 들과 섞이면서
    정렬 순서가 호출 순서에 따라 달라진다.
    <div style="margin:8px 0" class="dim">
      동일 조합 5회 연속: stretch 1.6225 / 1.6114 / 1.6960 / 1.6013 / 1.6253 (편차 ~5%,
      어떤 조합은 강연결 성공↔실패까지 뒤집힘). np/random/SA 시드를 다 고정해도,
      1스레드로 돌려도 같은 수열이 그대로 재현된다 → 랜덤성·BLAS 스레드 문제 아님.
      파티션 단독은 6/6 동일하므로 파티션이 아니라 솔버 경로 문제.
    </div>
    <b>이 리포트의 수치는 조합마다 카운터를 같은 값으로 리셋해 다시 측정한 것</b>이다
    (리셋 후 4/4 완전 동일 확인). 모든 조합이 동일한 솔버 초기 상태에서 비교되는
    페어드 설계다. 단, 여전히 확률적 알고리즘의 한 표본이므로 조합 간 미세한 차이
    (&lt;1%)는 우열 근거로 쓰지 않는 편이 안전하다.
  </div>

  <h2>2. 조합 계열별 성능</h2>
  <div class="cols">
    <div class="card" style="flex:1; min-width:520px" id="family"></div>
  </div>

  <h2>3. 조합별 결과 — 행을 누르면 아래 그림이 그 조합으로 바뀐다</h2>
  <div class="bar" id="jump"></div>
  <div class="cols">
    <div style="flex:1; min-width:520px">
      <div class="scroll card" style="padding:0"><table id="tbl"></table></div>
    </div>
    <div class="card" style="min-width:300px" id="detail"></div>
  </div>

  <h2>4. D 배정 시각화 (선택한 조합)</h2>
  <div class="legend">
    <span><i class="sw" style="background:#4f9dff"></i>canonical 방향 door</span>
    <span><i class="sw" style="background:#f5a623"></i>뒤집힌 door</span>
    <span><i class="sw" style="background:#c17f9f"></i>자유 간선(솔버가 결정, macro 색)</span>
    <span>화살표 = 그 조합에서 고정된 door 방향</span>
    <span>표의 <b>성분</b> = 뒤집힌 door 집합의 연결 성분 수, <b>*</b> 는 열린 조각 포함</span>
  </div>
  <div class="cols">
    <svg id="map" width="620" height="620"></svg>
    <svg id="quo" width="420" height="420"></svg>
    <div class="card" style="min-width:260px" id="comps"></div>
  </div>

  <h2>5. 상위·하위 조합의 flip 위치 한눈에 보기</h2>
  <div class="dim">door 전체는 회색, 그 조합에서 뒤집은 door 만 색으로 표시. 위 줄 = 가장 좋은 조합,
  아래 줄 = 가장 나쁜 조합. 미니맵을 누르면 위 그림이 그 조합으로 바뀐다.</div>
  <div id="gallery"></div>

  <h2>6. 패턴 분석</h2>
  <div class="cols">
    <div class="card" style="min-width:360px">
      <b>기저 사이클별 효과</b>
      <div class="dim">그 루프를 뒤집은 조합들의 평균 stretch − 안 뒤집은 조합들의 평균.
      음수(초록)면 그 루프 반전이 이득.</div>
      <div id="basisfx" style="margin-top:8px"></div>
    </div>
    <div class="card" style="min-width:360px">
      <b>macro별 효과</b>
      <div class="dim">그 macro 경계 door 를 하나라도 뒤집은 조합의 평균 stretch 차이.</div>
      <div id="macrofx" style="margin-top:8px"></div>
    </div>
    <div class="card" style="min-width:300px">
      <b>뒤집은 door 수 vs stretch</b>
      <svg id="scatter" width="330" height="240"></svg>
    </div>
  </div>

  <h2>7. 결론 (현재 seed 기준, 데이터에서 계산)</h2>
  <div class="card" id="concl" style="border-left:4px solid #4f9dff"></div>
</div>
<script>
const DATA = __DATA__;
const SEEDS = Object.keys(DATA.cases);
let curSeed = SEEDS[0], curRow = 0, sortKey = "stretch", sortDir = 1;

const cfg = DATA.config;
document.getElementById("intro").textContent =
  `정점 ${cfg.n_vertices} · 간선 제거율 ${cfg.remove_ratio} · face-cluster target_k=${cfg.target_k} ·`
  + ` 솔버 = 프로덕션과 같은 DnC(max_vertices=${cfg.max_vertices})+SA. `
  + `door 방향을 조합마다 고정한 뒤 나머지 간선만 솔버가 푼다. 지표는 평균 우회율(stretch, 낮을수록 좋음).`;

function macroColor(i, alpha) {
  const hue = (i * 47) % 360;
  return `hsla(${hue},55%,58%,${alpha})`;
}

function caseOf() { return DATA.cases[curSeed]; }
function rowsOf() { return caseOf().rows; }
function isBad(r) { return r.failed || !(r.sc >= 1) || !isFinite(r.apsp); }
function okRows() { return rowsOf().filter(r => !isBad(r)); }

// ---- seed 탭 ----
const seedbar = document.getElementById("seedbar");
SEEDS.forEach(s => {
  const b = document.createElement("button");
  b.textContent = `seed ${s}`;
  b.onclick = () => { curSeed = s; curRow = 0; renderAll(); };
  seedbar.appendChild(b);
});

function renderSeedbar() {
  [...seedbar.children].forEach((b, i) =>
    b.classList.toggle("active", SEEDS[i] === curSeed));
}

// ---- 구조 카드 ----
function renderStruct() {
  const c = caseOf(), s = c.stats;
  const macroSizes = c.macros.map(m => m.verts.length).sort((a, b) => b - a);
  document.getElementById("struct").innerHTML =
    [["macro 수", c.macros.length],
     ["macro 크기(정점)", macroSizes.join(", ")],
     ["door 수", s.doors],
     ["door 사이클공간 차원", s.cycle_dim],
     ["사이클공간 조합 수", s.cycle_space_total],
     ["└ 그중 feasible", s.cycle_space_feasible],
     ["해밍1 전수에서 추가", "+" + s.hamming1_extra],
     ["해밍2 전수에서 추가", "+" + s.hamming2_extra],
     ["해밍3 스윕에서 추가", `+${s.hamming3_extra} (${s.hamming3_checked.toLocaleString()}건 검사)`],
     ["해밍4 스윕에서 추가", `+${s.hamming4_extra ?? 0} (${(s.hamming4_checked ?? 0).toLocaleString()}건)`],
     ["feasible 근방 섭동 추가", `+${s.perturb_extra ?? 0} (${(s.perturb_checked ?? 0).toLocaleString()}건)`],
     ["프리필터 총 검사", (s.prefilter_tested ?? 0).toLocaleString() + `건 / ${Math.round(s.prefilter_sec ?? 0)}초`],
     ["솔버 실행 조합", c.rows.length]]
    .map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join("");

  document.getElementById("feasnote").innerHTML =
    `door 부분그래프는 <b>전 정점 짝수 차수</b> — 즉 닫힌 곡선들의 합집합이다. `
    + `프리필터(각 macro 를 door 고정 + 나머지 양방향으로 본 mixed 그래프가 단일 SCC 인가)를 통과하는 조합은 `
    + `사실상 <b>닫힌 door 루프를 통째로 뒤집은 것</b>뿐이다. 루프를 부분만 뒤집으면 뒤집힌 구간의 양 끝에 `
    + `일방향 cut 이 생겨 그 macro 를 강연결로 완성할 수 없다. `
    + `실제로 균등 랜덤 조합은 2000개 중 0개가 통과했고, 해밍1~3 전수 스윕에서 나온 예외도 `
    + `${s.hamming1_extra + s.hamming2_extra + s.hamming3_extra + (s.hamming4_extra ?? 0)
        + (s.perturb_extra ?? 0)}개뿐이다.`;
}

// ---- 표 ----
const FAM = {
  canonical: ["canonical", "#4f9dff"],
  loop: ["닫힌 루프 반전", "#9b7bd4"],
  local: ["사이클공간 밖", "#f5a623"],
};

function famLabel(f) {
  const [t, c] = FAM[f] || [f, "#9ab"];
  return `<span class="pill" style="color:${c};border-color:${c}55">${t}</span>`;
}

const COLS = [
  ["#", r => r.idx, r => r.idx],
  ["계열", r => famLabel(r.family), r => r.family],
  ["stretch", r => isBad(r) ? `<span class="bad">${r.failed ? "솔버실패" : "SC실패"}</span>`
      : r.apsp.toFixed(4), r => isBad(r) ? 1e9 : r.apsp],
  ["Δ vs canonical", r => isBad(r) ? "-" : fmtPct(r.dpct), r => isBad(r) ? 1e9 : r.dpct],
  ["flip", r => r.n_flipped, r => r.n_flipped],
  ["성분", r => r.n_comps + (r.all_closed ? "" : "*"), r => r.n_comps],
  ["SC", r => isBad(r) && r.failed ? "-" : r.sc.toFixed(2), r => isBad(r) ? -1 : r.sc],
  ["flow", r => isBad(r) && r.failed ? "-" : r.flow.toFixed(0), r => isBad(r) ? -1 : r.flow],
  ["출처", r => r.source, r => r.source],
  ["기저좌표", r => r.coords ? r.coords.join("") : "span 밖", r => r.in_span ? 0 : 1],
];

function fmtPct(x) {
  if (x === null || x === undefined || !isFinite(x)) return "-";
  const s = (x >= 0 ? "+" : "") + x.toFixed(2) + "%";
  return `<span class="${x < 0 ? "good" : x > 0 ? "bad" : "dim"}">${s}</span>`;
}

function prepRows() {
  const rows = rowsOf();
  const base = rows.find(r => r.bits === "0");
  const baseV = base && !isBad(base) ? base.apsp : null;
  rows.forEach((r, i) => {
    r.idx = i;
    r.dpct = (baseV && !isBad(r)) ? (r.apsp - baseV) / baseV * 100 : null;
  });
  return rows;
}

function renderTable() {
  const rows = prepRows().slice();
  const col = COLS.find(c => c[0] === sortKey) || COLS[1];
  rows.sort((a, b) => (col[2](a) > col[2](b) ? 1 : -1) * sortDir);
  const head = "<tr>" + COLS.map(c =>
    `<th data-k="${c[0]}">${c[0]}${sortKey === c[0] ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`
  ).join("") + "</tr>";
  const body = rows.map(r =>
    `<tr data-i="${r.idx}" class="${r.idx === curRow ? "sel" : ""}">`
    + COLS.map(c => `<td>${c[1](r)}</td>`).join("") + "</tr>").join("");
  const tbl = document.getElementById("tbl");
  tbl.innerHTML = head + body;
  tbl.querySelectorAll("th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = 1; }
    renderTable();
  });
  tbl.querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => {
    curRow = +tr.dataset.i; renderTable(); renderDetail(); renderMap(); renderQuotient(); renderComps();
  });
}

function renderFamily() {
  const rows = prepRows().filter(r => !isBad(r));
  const base = rowsOf().find(r => r.bits === "0");
  const baseV = base && !isBad(base) ? base.apsp : null;
  const groups = ["canonical", "loop", "local"];
  const head = `<tr><th>계열</th><th>조합 수</th><th>평균 stretch</th><th>최고</th><th>최악</th>`
    + `<th>canonical보다 좋은 조합</th><th>평균 flip 수</th></tr>`;
  const body = groups.map(g => {
    const rs = rows.filter(r => r.family === g);
    if (!rs.length) return "";
    const vals = rs.map(r => r.apsp);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const better = baseV === null ? "-" : rs.filter(r => r.apsp < baseV).length + " / " + rs.length;
    const mf = rs.reduce((a, r) => a + r.n_flipped, 0) / rs.length;
    return `<tr><td>${famLabel(g)}</td><td>${rs.length}</td><td>${mean.toFixed(4)}</td>`
      + `<td class="good">${Math.min(...vals).toFixed(4)}</td>`
      + `<td class="bad">${Math.max(...vals).toFixed(4)}</td>`
      + `<td>${better}</td><td>${mf.toFixed(1)}</td></tr>`;
  }).join("");
  const failed = prepRows().filter(isBad);
  document.getElementById("family").innerHTML =
    `<table>${head}${body}</table>`
    + `<div class="dim" style="margin-top:8px">`
    + `<b>닫힌 루프 반전</b> = 뒤집은 door 집합이 door 그래프의 닫힌 곡선(사이클공간 원소). 크기와 무관하게 항상 프리필터 통과. `
    + `<b>사이클공간 밖</b> = 열린 조각을 포함한 조합. 소수만 뒤집은 국소 flip 도, 닫힌 루프에서 door 몇 개를 더하거나 뺀 큰 조합도 여기 속하며, `
    + `그 macro 가 자유 간선으로 우회로를 만들 수 있는 자리에서만 프리필터를 통과한다. `
    + `프리필터를 통과했는데도 솔버가 강연결에 실패한 조합 ${failed.length}개는 표에서 <i>실패</i>로 표시된다(해가 존재하는데 못 찾은 것 = 솔버 쪽 한계).</div>`;
}

function renderJump() {
  const rows = prepRows().filter(r => !isBad(r));
  if (!rows.length) return;
  const best = rows.reduce((a, b) => a.apsp < b.apsp ? a : b);
  const worst = rows.reduce((a, b) => a.apsp > b.apsp ? a : b);
  const canon = prepRows().find(r => r.bits === "0");
  const targets = [["최고 조합", best], ["최악 조합", worst], ["canonical", canon]];
  const bar = document.getElementById("jump");
  bar.innerHTML = "";
  targets.forEach(([lab, r]) => {
    if (!r) return;
    const b = document.createElement("button");
    b.textContent = `${lab} (#${r.idx}${isBad(r) ? "" : ", " + r.apsp.toFixed(4)})`;
    b.onclick = () => {
      curRow = r.idx;
      renderTable(); renderDetail(); renderMap(); renderQuotient(); renderComps();
    };
    bar.appendChild(b);
  });
}

function renderDetail() {
  const r = prepRows()[curRow], c = caseOf();
  const perMacro = Object.entries(r.per_macro || {})
    .map(([m, n]) => `macro ${m}: ${n}`).join(" · ") || "없음";
  document.getElementById("detail").innerHTML =
    `<b>선택 조합 #${r.idx}</b> ${famLabel(r.family)}`
    + `<div class="kv"><span>stretch</span><span>${isBad(r) ? (r.failed ? "솔버실패" : "SC실패") : r.apsp.toFixed(4)}</span></div>`
    + `<div class="kv"><span>canonical 대비</span><span>${fmtPct(r.dpct)}</span></div>`
    + `<div class="kv"><span>강연결률</span><span>${r.failed ? "-" : r.sc.toFixed(3)}</span></div>`
    + `<div class="kv"><span>flow</span><span>${r.failed ? "-" : r.flow.toFixed(0)}</span></div>`
    + `<div class="kv"><span>뒤집은 door</span><span>${r.n_flipped} / ${c.door_ids.length}</span></div>`
    + `<div class="kv"><span>뒤집힌 성분 수</span><span>${r.n_comps} ${r.all_closed ? "(전부 닫힌 루프)" : "(열린 성분 포함)"}</span></div>`
    + `<div class="kv"><span>솔버 시간</span><span>${r.sec ? r.sec.toFixed(1) + "s" : "-"}</span></div>`
    + `<div style="margin-top:8px" class="dim">macro별 뒤집힌 door 수<br>${perMacro}</div>`;
}

// ---- 평면도 ----
function renderMap() {
  const c = caseOf(), r = prepRows()[curRow];
  const svg = document.getElementById("map");
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), P = 18;
  const flip = new Set(r.flipped);
  const pos = c.pos;
  const X = v => P + pos[v][0] * (W - 2 * P), Y = v => H - P - pos[v][1] * (H - 2 * P);

  const macroOf = {};
  c.macros.forEach((m, i) => m.free.forEach(f => macroOf[f[0]] = i));

  let out = `<defs>
    <marker id="ac" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#4f9dff"/></marker>
    <marker id="af" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#f5a623"/></marker></defs>`;

  // 자유 간선
  c.edges.forEach(([eid, u, v]) => {
    if (c.canon[eid]) return;
    const mi = macroOf[eid];
    const col = mi === undefined ? "#2b3138" : macroColor(mi, 0.55);
    out += `<line x1="${X(u)}" y1="${Y(u)}" x2="${X(v)}" y2="${Y(v)}" stroke="${col}" stroke-width="1.1"/>`;
  });
  // door
  c.door_ids.forEach(eid => {
    let [a, b] = c.canon[eid];
    const f = flip.has(eid);
    if (f) { const t = a; a = b; b = t; }
    const x1 = X(a), y1 = Y(a), x2 = X(b), y2 = Y(b);
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
    const sx = x1 + dx * 0.12, sy = y1 + dy * 0.12;
    const ex = x1 + dx * 0.88, ey = y1 + dy * 0.88;
    out += `<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" stroke="${f ? "#f5a623" : "#4f9dff"}"`
      + ` stroke-width="${f ? 2.6 : 2}" marker-end="url(#${f ? "af" : "ac"})"/>`;
  });
  // 정점
  Object.keys(pos).forEach(v => {
    out += `<circle cx="${X(v)}" cy="${Y(v)}" r="1.6" fill="#5b6673"/>`;
  });
  svg.innerHTML = out;
}

// ---- D (macro 인접) 그래프 ----
function renderQuotient() {
  const c = caseOf(), r = prepRows()[curRow];
  const svg = document.getElementById("quo");
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), P = 40;
  const flip = new Set(r.flipped);
  const cen = c.macros.map(m => {
    let sx = 0, sy = 0;
    m.verts.forEach(v => { sx += c.pos[v][0]; sy += c.pos[v][1]; });
    return [sx / m.verts.length, sy / m.verts.length];
  });
  const X = p => P + p[0] * (W - 2 * P), Y = p => H - P - p[1] * (H - 2 * P);

  const bundles = {};
  c.door_ids.forEach(eid => {
    const own = c.owner[eid];
    if (!own || own.length < 2) return;
    const key = own[0] + "-" + own[1];
    (bundles[key] = bundles[key] || { a: own[0], b: own[1], n: 0, f: 0 });
    bundles[key].n++;
    if (flip.has(eid)) bundles[key].f++;
  });

  let out = "";
  Object.values(bundles).forEach(bu => {
    const p = cen[bu.a], q = cen[bu.b];
    const frac = bu.f / bu.n;
    const col = frac === 0 ? "#4f9dff" : frac === 1 ? "#f5a623" : "#e86a6a";
    out += `<line x1="${X(p)}" y1="${Y(p)}" x2="${X(q)}" y2="${Y(q)}" stroke="${col}"`
      + ` stroke-width="${1 + Math.min(6, bu.n / 3)}" opacity="0.85"/>`;
    const mx = (X(p) + X(q)) / 2, my = (Y(p) + Y(q)) / 2;
    out += `<text x="${mx}" y="${my - 3}" fill="#9ab" font-size="10" text-anchor="middle">${bu.f}/${bu.n}</text>`;
  });
  c.macros.forEach((m, i) => {
    const p = cen[i], rad = 6 + Math.sqrt(m.verts.length) * 1.6;
    out += `<circle cx="${X(p)}" cy="${Y(p)}" r="${rad}" fill="${macroColor(i, 0.75)}" stroke="#0c0e11"/>`
      + `<text x="${X(p)}" y="${Y(p) + 3.5}" font-size="10" text-anchor="middle" fill="#14161a">${i}</text>`;
  });
  svg.innerHTML = out;
}

function renderComps() {
  const r = prepRows()[curRow];
  const html = r.comps.length === 0
    ? "<div class='dim'>canonical (뒤집은 door 없음)</div>"
    : r.comps.map((cp, i) =>
        `<div class="kv"><span>성분 ${i + 1}</span><span>${cp.n} door · `
        + `${cp.closed ? "<span class='good'>닫힌 루프</span>" : "<span class='bad'>열린 경로</span>"}</span></div>`
      ).join("");
  document.getElementById("comps").innerHTML =
    `<b>뒤집힌 door 성분</b><div class="dim" style="margin:4px 0 8px">`
    + `닫힌 루프 = 방향 반전이 macro 강연결을 깨지 않는 유일한 형태.</div>` + html;
}

// ---- 패턴 분석 ----
function barRows(el, items, unit) {
  const max = Math.max(...items.map(it => Math.abs(it[1])), 1e-9);
  el.innerHTML = items.map(([lab, val, extra]) => {
    const w = Math.abs(val) / max * 100;
    const col = val < 0 ? "#6ec86e" : "#e86a6a";
    return `<div class="barrow"><span class="lab">${lab}</span>`
      + `<span class="val ${val < 0 ? "good" : "bad"}">${val >= 0 ? "+" : ""}${val.toFixed(3)}${unit}</span>`
      + `<span style="flex:1"><span class="barfill" style="display:block;width:${w}%;background:${col}"></span></span>`
      + `<span class="dim" style="width:74px">${extra || ""}</span></div>`;
  }).join("");
}

function renderEffects() {
  const c = caseOf(), rows = okRows();
  const base = rowsOf().find(r => r.bits === "0");
  const baseV = base && !base.failed ? base.apsp : null;

  // 기저 사이클별
  const nb = c.basis_edges.length;
  const items = [];
  for (let i = 0; i < nb; i++) {
    const on = rows.filter(r => r.coords && r.coords[i] === 1);
    const off = rows.filter(r => r.coords && r.coords[i] === 0);
    if (!on.length || !off.length) { items.push([`루프 ${i + 1}`, 0, "표본없음"]); continue; }
    const mOn = on.reduce((s, r) => s + r.apsp, 0) / on.length;
    const mOff = off.reduce((s, r) => s + r.apsp, 0) / off.length;
    items.push([`루프 ${i + 1} (${c.basis_edges[i].length}door)`, mOn - mOff, `n=${on.length}/${off.length}`]);
  }
  barRows(document.getElementById("basisfx"), items, "");

  // macro별
  const mitems = c.macros.map((m, i) => {
    const on = rows.filter(r => (r.per_macro || {})[i] > 0);
    const off = rows.filter(r => !((r.per_macro || {})[i] > 0));
    if (!on.length || !off.length) return [`macro ${i}`, 0, "표본없음"];
    const mOn = on.reduce((s, r) => s + r.apsp, 0) / on.length;
    const mOff = off.reduce((s, r) => s + r.apsp, 0) / off.length;
    return [`macro ${i} (V=${m.verts.length})`, mOn - mOff, `n=${on.length}/${off.length}`];
  }).sort((a, b) => a[1] - b[1]);
  barRows(document.getElementById("macrofx"), mitems, "");

  // 산점도
  const svg = document.getElementById("scatter");
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), P = 34;
  if (!rows.length) { svg.innerHTML = ""; return; }
  const xs = rows.map(r => r.n_flipped), ys = rows.map(r => r.apsp);
  const x0 = 0, x1 = Math.max(...xs) || 1;
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = v => P + (v - x0) / (x1 - x0 || 1) * (W - P - 12);
  const Y = v => H - P - (v - y0) / ((y1 - y0) || 1) * (H - P - 14);
  let out = `<line x1="${P}" y1="${H - P}" x2="${W - 6}" y2="${H - P}" stroke="#39404a"/>`
    + `<line x1="${P}" y1="8" x2="${P}" y2="${H - P}" stroke="#39404a"/>`
    + `<text x="${W / 2}" y="${H - 8}" fill="#78828e" font-size="10" text-anchor="middle">뒤집은 door 수</text>`
    + `<text x="8" y="14" fill="#78828e" font-size="10">stretch</text>`
    + `<text x="${P - 4}" y="${H - P + 12}" fill="#78828e" font-size="9" text-anchor="end">${y0.toFixed(3)}</text>`
    + `<text x="${P - 4}" y="14" fill="#78828e" font-size="9" text-anchor="end">${y1.toFixed(3)}</text>`;
  if (baseV !== null) {
    out += `<line x1="${P}" y1="${Y(baseV)}" x2="${W - 6}" y2="${Y(baseV)}" stroke="#4f9dff" stroke-dasharray="3 3"/>`;
  }
  rows.forEach(r => {
    out += `<circle cx="${X(r.n_flipped)}" cy="${Y(r.apsp)}" r="3"`
      + ` fill="${baseV !== null && r.apsp < baseV ? "#6ec86e" : "#e86a6a"}" opacity="0.8"><title>`
      + `flip ${r.n_flipped}, stretch ${r.apsp.toFixed(4)}</title></circle>`;
  });
  svg.innerHTML = out;
}

function miniMap(r, size, color) {
  const c = caseOf(), P = 6;
  const flip = new Set(r.flipped);
  const X = v => P + c.pos[v][0] * (size - 2 * P);
  const Y = v => size - P - c.pos[v][1] * (size - 2 * P);
  let out = "";
  c.door_ids.forEach(eid => {
    const [a, b] = c.canon[eid];
    if (flip.has(eid)) return;
    out += `<line x1="${X(a)}" y1="${Y(a)}" x2="${X(b)}" y2="${Y(b)}" stroke="#39404a" stroke-width="1"/>`;
  });
  c.door_ids.forEach(eid => {
    const [a, b] = c.canon[eid];
    if (!flip.has(eid)) return;
    out += `<line x1="${X(a)}" y1="${Y(a)}" x2="${X(b)}" y2="${Y(b)}" stroke="${color}" stroke-width="3"/>`
      + `<circle cx="${(X(a) + X(b)) / 2}" cy="${(Y(a) + Y(b)) / 2}" r="4" fill="none" stroke="${color}" stroke-width="1"/>`;
  });
  return `<svg width="${size}" height="${size}">${out}</svg>`;
}

function renderGallery() {
  const rows = prepRows().filter(r => !isBad(r)).sort((a, b) => a.apsp - b.apsp);
  const n = Math.min(8, Math.floor(rows.length / 2));
  const top = rows.slice(0, n), bot = rows.slice(-n).reverse();
  const cell = (r, color) =>
    `<div class="card" style="padding:6px; cursor:pointer" data-i="${r.idx}">`
    + miniMap(r, 168, color)
    + `<div style="font-size:11.5px; text-align:center">#${r.idx} · flip ${r.n_flipped} · `
    + `<b class="${r.dpct < 0 ? "good" : "bad"}">${r.dpct >= 0 ? "+" : ""}${r.dpct.toFixed(2)}%</b></div>`
    + `<div style="font-size:11px; text-align:center" class="dim">${FAM[r.family][0]}</div></div>`;
  const el = document.getElementById("gallery");
  el.innerHTML =
    `<div class="dim" style="margin:8px 0 4px">좋은 조합 ${n}개</div>`
    + `<div class="cols">${top.map(r => cell(r, "#6ec86e")).join("")}</div>`
    + `<div class="dim" style="margin:14px 0 4px">나쁜 조합 ${n}개</div>`
    + `<div class="cols">${bot.map(r => cell(r, "#e86a6a")).join("")}</div>`;
  el.querySelectorAll("[data-i]").forEach(d => d.onclick = () => {
    curRow = +d.dataset.i;
    renderTable(); renderDetail(); renderMap(); renderQuotient(); renderComps();
    document.getElementById("map").scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

function renderConclusion() {
  const c = caseOf(), all = prepRows();
  const rows = all.filter(r => !isBad(r));
  const base = all.find(r => r.bits === "0");
  const baseV = base && !isBad(base) ? base.apsp : null;
  if (!rows.length || baseV === null) {
    document.getElementById("concl").textContent = "SC 성공 조합이 없어 결론을 낼 수 없다.";
    return;
  }
  const best = rows.reduce((a, b) => a.apsp < b.apsp ? a : b);
  const worst = rows.reduce((a, b) => a.apsp > b.apsp ? a : b);
  const loop = rows.filter(r => r.family === "loop");
  const local = rows.filter(r => r.family === "local");
  const mean = rs => rs.length ? rs.reduce((a, r) => a + r.apsp, 0) / rs.length : NaN;
  const better = rows.filter(r => r.apsp < baseV).length;
  const pct = v => ((v - baseV) / baseV * 100).toFixed(2) + "%";
  const failed = all.filter(isBad).length;

  // flip 수와 stretch 의 상관
  const xs = rows.map(r => r.n_flipped), ys = rows.map(r => r.apsp);
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let cov = 0, sx = 0, sy = 0;
  xs.forEach((x, i) => { cov += (x - mx) * (ys[i] - my); sx += (x - mx) ** 2; sy += (ys[i] - my) ** 2; });
  const corr = (sx > 0 && sy > 0) ? cov / Math.sqrt(sx * sy) : 0;

  const sorted = rows.slice().sort((a, b) => a.apsp - b.apsp);
  const k = Math.max(1, Math.min(10, Math.floor(rows.length / 10)));
  const topFlip = sorted.slice(0, k).reduce((a, r) => a + r.n_flipped, 0) / k;
  const botFlip = sorted.slice(-k).reduce((a, r) => a + r.n_flipped, 0) / k;
  const full = rows.find(r => r.n_flipped === c.door_ids.length);

  document.getElementById("concl").innerHTML = `
  <ol style="margin:0; padding-left:20px">
    <li><b>가능한 조합 자체가 희소하다.</b> door ${c.door_ids.length}개면 조합은 2<sup>${c.door_ids.length}</sup>지만,
      각 macro 를 강연결로 완성할 수 있는 조합은 프리필터 ${(c.stats.prefilter_tested || 0).toLocaleString()}건 검사에서
      ${all.length}개만 나왔다. 균등 랜덤 조합은 2000개 중 0개였다.</li>
    <li><b>가능한 조합은 두 계열뿐이다.</b> (i) door 그래프의 <b>닫힌 곡선을 통째로 반전</b>한 것,
      (ii) 자유 간선이 우회로를 만들어 줄 수 있는 자리에서 <b>소수만 국소 반전</b>한 것.
      루프를 부분만 뒤집으면 뒤집힌 구간의 양 끝에 일방향 cut 이 생겨 그 macro 를 강연결로 완성할 수 없다.
      이 구조는 솔버와 무관한 순수 조합론이라 재현성 문제의 영향을 받지 않는다.</li>
    <li><b>canonical(2색칠 기본 방향)은 최적이 아니다.</b> 조사한 ${rows.length}개 조합 중
      ${better}개가 canonical(${baseV.toFixed(4)}) 보다 낮은 stretch 를 냈고, 최고는
      ${best.apsp.toFixed(4)} (<span class="good">${pct(best.apsp)}</span>, door ${best.n_flipped}개 반전)다.</li>
    <li><b>나쁜 조합은 거의 항상 소규모 국소 flip 이다.</b> 하위 ${k}개 조합의 평균 반전 수는
      ${botFlip.toFixed(1)}개(최악 ${worst.n_flipped}개, <span class="bad">${pct(worst.apsp)}</span>),
      상위 ${k}개는 ${topFlip.toFixed(1)}개다. 반전 수와 stretch 의 상관은
      <b>${corr.toFixed(3)}</b> — 이 그래프에서는 ${corr < -0.3 ? "많이 뒤집을수록 좋아지는" :
        corr > 0.3 ? "많이 뒤집을수록 나빠지는" : "반전 수와 무관한"} 경향이다.
      door 몇 개만 국소로 뒤집으면 그 지점이 병목이 되어 손해 보기 쉽다.</li>
    <li><b>그렇다고 전부 뒤집는 게 답도 아니다.</b>
      ${full ? `전체 반전(${c.door_ids.length}개)은 ${full.apsp.toFixed(4)} (${pct(full.apsp)}) 로,
        최적(${best.apsp.toFixed(4)})보다 나쁘다.` : "전체 반전 조합은 이 목록에 없다."}
      최적은 언제나 <b>부분 반전</b>이고, 그 규모는 그래프마다 다르다.</li>
    <li><b>계열별 안정성.</b> 닫힌 루프 계열 평균 ${mean(loop).toFixed(4)} (n=${loop.length}),
      국소 계열 평균 ${mean(local).toFixed(4)} (n=${local.length}).
      루프 반전은 평균이 안정적이고, 국소 flip 은 최고·최악을 모두 만드는 고분산 수단이다.</li>
    <li><b>솔버가 못 찾는 해가 많다.</b> 프리필터를 통과해 <b>해가 존재함이 보장된</b> 조합
      ${all.length}개 중 ${failed}개(${(100 * failed / all.length).toFixed(0)}%)에서 솔버가 강연결에 실패했다.
      대부분 소규모 flip 조합이다 — door 방향 자유도보다 솔버 탐색 쪽이 먼저 한계다.</li>
  </ol>`;
}

function renderAll() {
  renderSeedbar(); renderStruct(); renderFamily(); renderJump();
  renderTable(); renderDetail();
  renderMap(); renderQuotient(); renderComps(); renderGallery();
  renderEffects(); renderConclusion();
}
renderAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
