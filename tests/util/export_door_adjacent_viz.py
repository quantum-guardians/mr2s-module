"""Door 인접 간선 색칠/제거 시각화 HTML 생성.

macro_viz_data.json (export_macro_viz.py 출력) 을 읽어
door_adjacent_removal.html 을 만든다.

    python tests/util/export_door_adjacent_viz.py
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("tests/util/macro_viz_data.json")
OUT = Path("tests/util/door_adjacent_removal.html")

TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Door 인접 간선 제거 — macro 독립성</title>
<style>
  body { margin: 0; font: 13px/1.5 system-ui, sans-serif; background: #14161a; color: #dde; }
  h1 { font-size: 16px; margin: 12px 16px 4px; }
  .sub { margin: 0 16px 8px; color: #9ab; }
  .wrap { display: flex; flex-wrap: wrap; gap: 12px; padding: 0 16px 16px; }
  .panel { background: #1c1f24; border: 1px solid #2c3038; border-radius: 8px; padding: 8px; }
  .panel h2 { font-size: 13px; margin: 4px 6px 6px; color: #cde; font-weight: 600; }
  svg { display: block; background: #101216; border-radius: 4px; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 6px 2px; color: #9ab; }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .sw { width: 18px; height: 4px; border-radius: 2px; display: inline-block; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  #stats { background: #1c1f24; border: 1px solid #2c3038; border-radius: 8px;
           padding: 10px 14px; margin: 0 16px 16px; max-width: 1424px; }
  #stats table { border-collapse: collapse; margin-top: 6px; }
  #stats td, #stats th { border: 1px solid #333a44; padding: 3px 10px; text-align: left; }
  #stats th { color: #9ab; font-weight: 600; }
  .ok { color: #6ec86e; font-weight: 600; }
  .bad { color: #e86a6a; font-weight: 600; }
  label { user-select: none; cursor: pointer; margin-right: 14px; }
</style>
</head>
<body>
<h1>Door-only: door 인접 간선 색칠 &rarr; 제거 후 macro 독립성</h1>
<p class="sub">seed=__SEED__ &middot; V=__NV__ E=__NE__ &middot; macro __NM__개 &middot; door __ND__개
(인접 두 macro 공유 경계만; 외평면 접촉 외곽선 제외).
door 인접 간선 = door 가 아니면서 door 끝점에 닿는 간선.</p>
<div class="wrap">
  <div class="panel">
    <h2>A. 원본 — door / 인접 간선 색칠</h2>
    <svg id="svgA" width="700" height="700"></svg>
    <div class="legend">
      <span><i class="sw" style="background:#e84a4a"></i>door (__ND__)</span>
      <span><i class="sw" style="background:#f5a623"></i>door 인접 (<b id="nAdj"></b>)</span>
      <span><i class="sw" style="background:#4a5260"></i>일반 (내부)</span>
      <span><i class="dot" style="background:#e84a4a"></i>door 끝점</span>
    </div>
  </div>
  <div class="panel">
    <h2>B. 제거 후 — 남은 간선 macro 색, 연결요소</h2>
    <div style="margin:0 6px 6px">
      <label><input type="checkbox" id="rmAdj" checked> 인접 간선 제거</label>
      <label><input type="checkbox" id="rmDoor" checked> door 도 제거</label>
      <label><input type="checkbox" id="showGhost" checked> 제거된 간선 흐리게 표시</label>
    </div>
    <svg id="svgB" width="700" height="700"></svg>
    <div class="legend" id="legB"></div>
  </div>
  <div class="panel">
    <h2>C. 정점 관점 — 공유 정점(&ge;2 macro) = separator</h2>
    <div style="margin:0 6px 6px">
      <label><input type="checkbox" id="vcut"> 공유 정점 제거 (vertex cut)</label>
      <label><input type="checkbox" id="ghostC" checked> 제거분 흐리게</label>
    </div>
    <svg id="svgC" width="700" height="700"></svg>
    <div class="legend" id="legC"></div>
  </div>
</div>
<div id="stats"></div>
<div id="statsC" style="background:#1c1f24;border:1px solid #2c3038;border-radius:8px;
     padding:10px 14px;margin:0 16px 16px;max-width:1424px"></div>
<script>
const DATA = __DATA__;

const PAL = ["#4f9dff","#5ad07a","#c78bff","#ffd24a","#ff8a5c","#37d3d3",
             "#ff6ea8","#a3e635","#94a3b8","#f472b6"];
const W = 700, PAD = 18;
function xy(v) {
  const p = DATA.verts[v];
  return [PAD + p[0] * (W - 2 * PAD), PAD + (1 - p[1]) * (W - 2 * PAD)];
}
function line(svg, a, b, stroke, w, dash, op) {
  const [x1, y1] = xy(a), [x2, y2] = xy(b);
  const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
  el.setAttribute("x1", x1); el.setAttribute("y1", y1);
  el.setAttribute("x2", x2); el.setAttribute("y2", y2);
  el.setAttribute("stroke", stroke); el.setAttribute("stroke-width", w);
  if (dash) el.setAttribute("stroke-dasharray", dash);
  if (op !== undefined) el.setAttribute("opacity", op);
  svg.appendChild(el);
  return el;
}
function circ(svg, v, r, fill, op) {
  const [cx, cy] = xy(v);
  const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  el.setAttribute("cx", cx); el.setAttribute("cy", cy);
  el.setAttribute("r", r); el.setAttribute("fill", fill);
  if (op !== undefined) el.setAttribute("opacity", op);
  svg.appendChild(el);
}

// ---- 분류 ----
// door = 인접 두 macro 가 공유하는 경계 간선만. 외평면(outer face)과 맞닿는
// 외곽선은 directed 로 나왔어도 door 가 아니다 (owners 1개).
const ek = (u, v) => (u < v ? u + "," + v : v + "," + u);
const DOORS = DATA.doors.filter(d => d.macros.length >= 2);
const doorKeys = new Set(DOORS.map(d => ek(d.u, d.v)));
const doorVerts = new Set();
DOORS.forEach(d => { doorVerts.add(d.u); doorVerts.add(d.v); });

// edge -> {u,v,kind: door|adj|free}
const edges = DATA.edges.map(e => {
  const isDoor = doorKeys.has(ek(e.u, e.v));
  const kind = isDoor ? "door"
    : (doorVerts.has(e.u) || doorVerts.has(e.v)) ? "adj" : "free";
  return { u: e.u, v: e.v, kind };
});
const nAdj = edges.filter(e => e.kind === "adj").length;
document.getElementById("nAdj").textContent = nAdj;

// edge key -> 소유 macro 들
const key2macros = {};
DATA.macros.forEach(m => m.edges.forEach(e => {
  const k = ek(e.u, e.v);
  (key2macros[k] = key2macros[k] || []).push(m.id);
}));
// 정점 -> macro 들 (macro.verts 기준)
const vert2macros = {};
DATA.macros.forEach(m => m.verts.forEach(v => {
  (vert2macros[v] = vert2macros[v] || []).push(m.id);
}));

// ---- Panel A ----
{
  const svg = document.getElementById("svgA");
  const order = { free: 0, adj: 1, door: 2 };
  [...edges].sort((a, b) => order[a.kind] - order[b.kind]).forEach(e => {
    if (e.kind === "free") line(svg, e.u, e.v, "#4a5260", 1.4);
    else if (e.kind === "adj") line(svg, e.u, e.v, "#f5a623", 2);
    else line(svg, e.u, e.v, "#e84a4a", 2.4);
  });
  Object.keys(DATA.verts).forEach(v => {
    circ(svg, v, doorVerts.has(+v) ? 3 : 1.8,
         doorVerts.has(+v) ? "#e84a4a" : "#7a8494");
  });
}

// ---- Panel B (토글 재계산) ----
function renderB() {
  const rmAdj = document.getElementById("rmAdj").checked;
  const rmDoor = document.getElementById("rmDoor").checked;
  const ghost = document.getElementById("showGhost").checked;
  const svg = document.getElementById("svgB");
  svg.innerHTML = "";

  const kept = edges.filter(e =>
    !(rmAdj && e.kind === "adj") && !(rmDoor && e.kind === "door"));
  const removed = edges.filter(e => !kept.includes(e));

  // union-find (남은 간선만)
  const par = {};
  const find = x => (par[x] === undefined || par[x] === x)
    ? (par[x] = x) : (par[x] = find(par[x]));
  kept.forEach(e => { const a = find(e.u), b = find(e.v); if (a !== b) par[a] = b; });

  if (ghost) removed.forEach(e =>
    line(svg, e.u, e.v, e.kind === "door" ? "#e84a4a" : "#f5a623", 1, "3,4", 0.22));

  kept.forEach(e => {
    const ms = key2macros[ek(e.u, e.v)] || [];
    const col = ms.length === 1 ? PAL[ms[0] % PAL.length] : "#ffffff";
    line(svg, e.u, e.v, col, 1.8);
  });

  // 정점: 남은 간선이 없는 고립 정점은 흐린 회색 링
  const touched = new Set();
  kept.forEach(e => { touched.add(e.u); touched.add(e.v); });
  let nIso = 0;
  Object.keys(DATA.verts).forEach(v => {
    if (touched.has(+v)) circ(svg, v, 1.8, "#aab4c4");
    else { circ(svg, v, 2.6, "#3a3f48"); nIso++; }
  });

  // 연결요소 -> 참여 macro 집합 (간선 소유 macro 기준)
  const compMacros = {}, compEdges = {};
  kept.forEach(e => {
    const c = find(e.u);
    compEdges[c] = (compEdges[c] || 0) + 1;
    (compMacros[c] = compMacros[c] || new Set());
    (key2macros[ek(e.u, e.v)] || []).forEach(m => compMacros[c].add(m));
  });
  const comps = Object.keys(compMacros);

  // macro -> 걸쳐 있는 연결요소 수
  const macroComps = {};
  DATA.macros.forEach(m => macroComps[m.id] = new Set());
  kept.forEach(e => {
    const c = find(e.u);
    (key2macros[ek(e.u, e.v)] || []).forEach(m => macroComps[m].add(c));
  });

  // 범례
  document.getElementById("legB").innerHTML =
    DATA.macros.map(m =>
      `<span><i class="sw" style="background:${PAL[m.id % PAL.length]}"></i>macro ${m.id}</span>`
    ).join("") +
    `<span><i class="sw" style="background:#ffffff"></i>공유 간선</span>` +
    `<span><i class="dot" style="background:#3a3f48"></i>고립 정점 (${nIso})</span>`;

  // 통계표
  const rows = DATA.macros.map(m => {
    const cs = macroComps[m.id];
    // 이 macro 의 컴포넌트들이 타 macro 와 섞였는가
    let mixed = false;
    cs.forEach(c => { if (compMacros[c].size > 1) mixed = true; });
    const indep = cs.size === 1 && !mixed;
    return `<tr><td>macro ${m.id}</td><td>${m.edges.length}</td><td>${cs.size}</td>
      <td class="${indep ? "ok" : "bad"}">${indep ? "독립 O"
        : (mixed ? "타 macro 와 연결" : cs.size + "조각 분리")}</td></tr>`;
  }).join("");

  const compList = comps
    .map(c => ({ n: compEdges[c], ms: [...compMacros[c]].sort((a,b)=>a-b) }))
    .sort((a, b) => b.n - a.n)
    .map(o => `{간선 ${o.n}, macro [${o.ms.join(",")}]}`).join(" &middot; ");

  document.getElementById("stats").innerHTML =
    `<b>제거:</b> 인접 ${rmAdj ? nAdj : 0}개 + door ${rmDoor ? DOORS.length : 0}개
     &rarr; <b>남은 간선</b> ${kept.length} / ${edges.length},
     <b>연결요소</b> ${comps.length}개, <b>고립 정점</b> ${nIso}개
     <table><tr><th>macro</th><th>간선 수(원본)</th><th>걸친 연결요소 수</th><th>판정</th></tr>
     ${rows}</table>
     <div style="margin-top:6px;color:#9ab"><b>연결요소 상세:</b> ${compList}</div>`;
}
["rmAdj", "rmDoor", "showGhost"].forEach(id =>
  document.getElementById(id).addEventListener("change", renderB));
renderB();

// ---- Panel C (정점 관점) ----
const sharedVerts = new Set(
  Object.keys(vert2macros).filter(v => vert2macros[v].length >= 2).map(Number));

function renderC() {
  const cut = document.getElementById("vcut").checked;
  const ghost = document.getElementById("ghostC").checked;
  const svg = document.getElementById("svgC");
  svg.innerHTML = "";

  const kept = cut
    ? edges.filter(e => !sharedVerts.has(e.u) && !sharedVerts.has(e.v))
    : edges;
  const removed = cut
    ? edges.filter(e => sharedVerts.has(e.u) || sharedVerts.has(e.v)) : [];

  if (ghost) removed.forEach(e => line(svg, e.u, e.v, "#5a6270", 1, "3,4", 0.25));

  kept.forEach(e => {
    const ms = key2macros[ek(e.u, e.v)] || [];
    const col = ms.length === 1 ? PAL[ms[0] % PAL.length] : "#ffffff";
    line(svg, e.u, e.v, col, cut ? 1.8 : 1.4, undefined, cut ? 1 : 0.75);
  });

  // 정점: 공유 정점은 macro 수에 따라 크기/색, 내부 정점은 소속 macro 색
  Object.keys(DATA.verts).forEach(vs => {
    const v = +vs;
    const ms = vert2macros[v] || [];
    if (sharedVerts.has(v)) {
      const deg = ms.length; // 2,3,4...
      const col = deg >= 4 ? "#ff4ad4" : deg === 3 ? "#e84a4a" : "#f5a623";
      circ(svg, v, deg >= 3 ? 5 : 3.6, col, cut ? 0.45 : 1);
    } else {
      circ(svg, v, 2.2, ms.length === 1 ? PAL[ms[0] % PAL.length] : "#7a8494");
    }
  });

  // union-find (kept 간선만) -> macro 내부 컴포넌트
  const par = {};
  const find = x => (par[x] === undefined || par[x] === x)
    ? (par[x] = x) : (par[x] = find(par[x]));
  kept.forEach(e => { const a = find(e.u), b = find(e.v); if (a !== b) par[a] = b; });
  const compM = {}, compV = {};
  kept.forEach(e => {
    const c = find(e.u);
    (compV[c] = compV[c] || new Set()).add(e.u); compV[c].add(e.v);
    (compM[c] = compM[c] || new Set());
    [e.u, e.v].forEach(w => (vert2macros[w] || []).forEach(m => compM[c].add(m)));
  });
  const comps = Object.keys(compM);
  const pure = comps.every(c => compM[c].size === 1);

  // macro 별 내부 통계
  const macroShared = {}, macroInner = {}, macroComps = {};
  DATA.macros.forEach(m => {
    macroShared[m.id] = m.verts.filter(v => sharedVerts.has(v)).length;
    macroInner[m.id] = m.verts.length - macroShared[m.id];
    macroComps[m.id] = new Set();
  });
  kept.forEach(e => {
    const c = find(e.u);
    (key2macros[ek(e.u, e.v)] || []).forEach(m => macroComps[m].add(c));
  });

  document.getElementById("legC").innerHTML =
    `<span><i class="dot" style="background:#f5a623"></i>공유 2-macro</span>` +
    `<span><i class="dot" style="background:#e84a4a"></i>공유 3-macro</span>` +
    `<span><i class="dot" style="background:#ff4ad4"></i>공유 4-macro</span>` +
    DATA.macros.map(m =>
      `<span><i class="dot" style="background:${PAL[m.id % PAL.length]}"></i>macro ${m.id} 내부</span>`
    ).join("");

  const rows = DATA.macros.map(m =>
    `<tr><td>macro ${m.id}</td><td>${m.verts.length}</td><td>${macroShared[m.id]}</td>
     <td>${macroInner[m.id]}</td><td>${cut ? macroComps[m.id].size : "-"}</td></tr>`
  ).join("");

  document.getElementById("statsC").innerHTML =
    `<b>정점 관점:</b> 공유 정점 ${sharedVerts.size} / ${DATA.n_vertices}` +
    (cut ? ` &middot; 컷 적용 &rarr; 남은 간선 ${kept.length}, 컴포넌트 ${comps.length},
        전부 단일 macro? <span class="${pure ? "ok" : "bad"}">${pure ? "O" : "X"}</span>` : "") +
    `<table><tr><th>macro</th><th>정점</th><th>공유(경계)</th><th>내부</th>
     <th>컷 후 컴포넌트</th></tr>${rows}</table>`;
}
["vcut", "ghostC"].forEach(id =>
  document.getElementById(id).addEventListener("change", renderC));
renderC();
</script>
</body>
</html>
"""


def main() -> None:
    data = json.loads(SRC.read_text())
    html = (
        TEMPLATE
        .replace("__SEED__", str(data["seed"]))
        .replace("__NV__", str(data["n_vertices"]))
        .replace("__NE__", str(data["n_edges"]))
        .replace("__NM__", str(data["n_macros"]))
        .replace("__ND__", str(sum(1 for d in data["doors"] if len(d["macros"]) >= 2)))
        .replace("__DATA__", json.dumps(data))
    )
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
