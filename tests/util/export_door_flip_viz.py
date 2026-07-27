"""door_flip_viz_data.json → 비교 시각화 HTML (door_flip_compare.html).

    python tests/util/export_door_flip_viz.py
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("tests/util/door_flip_viz_data.json")
OUT = Path("tests/util/door_flip_compare.html")

TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>경계 간선 방향 교정 — 전/후 비교</title>
<style>
  body { margin: 0; font: 14px/1.6 system-ui, sans-serif; background: #14161a; color: #dde; }
  header { padding: 16px 24px 8px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: #9ab; font-size: 13px; }
  #seedbar { display: flex; gap: 6px; flex-wrap: wrap; padding: 10px 24px; }
  #seedbar button { background: #1c1f24; color: #dde; border: 1px solid #333a44;
    border-radius: 8px; padding: 6px 10px; cursor: pointer; font: 12px system-ui;
    text-align: center; line-height: 1.3; }
  #seedbar button.active { border-color: #4f9dff; background: #23303f; }
  #seedbar button b { color: #6ec86e; }
  .wrap { display: flex; gap: 16px; padding: 8px 24px 24px; flex-wrap: wrap; }
  svg { background: #101216; border-radius: 10px; }
  .side { max-width: 330px; display: flex; flex-direction: column; gap: 12px; }
  .card { background: #1c1f24; border: 1px solid #2c3038; border-radius: 10px;
          padding: 12px 16px; font-size: 13px; }
  .card h2 { font-size: 13px; margin: 0 0 8px; color: #cde; }
  .big { font-size: 22px; font-weight: 700; color: #6ec86e; }
  .row { display: flex; justify-content: space-between; margin: 2px 0; }
  .row span:first-child { color: #9ab; }
  .lg { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
  .sw { width: 26px; height: 4px; border-radius: 2px; flex: none; }
  .fc { width: 14px; height: 14px; border-radius: 3px; flex: none; }
  .dim { color: #78828e; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>경계 간선 방향 교정 — 기본 해 vs 교정 후</h1>
  <div class="sub">정점 200 · 간선 제거율 0.4 · 그래프 10개.
  기본 해(면 순회 방향 복사)를 그대로 두고, <b>경계 간선의 방향만 뒤집어</b>
  평균 우회율이 낮아지면 채택한 결과. 뒤집은 간선만 주황 화살표로 표시.</div>
</header>
<div id="seedbar"></div>
<div class="wrap">
  <svg id="svg" width="720" height="720">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff9b3d"/>
      </marker>
    </defs>
    <g id="gFaces"></g><g id="gEdges"></g><g id="gFlips"></g>
  </svg>
  <div class="side">
    <div class="card" id="stats"></div>
    <div class="card">
      <h2>범례 · 용어</h2>
      <div class="lg"><i class="fc" style="background:#4f9dff33"></i>
        <div><b>구역</b> — 그래프를 나눠 푸는 묶음. 배경색이 같으면 같은 구역.</div></div>
      <div class="lg"><i class="sw" style="background:#3d5a80"></i>
        <div><b>경계 간선</b> — 이웃한 두 구역이 함께 쓰는 간선 (파란 계열).</div></div>
      <div class="lg"><i class="sw" style="background:#ff9b3d;height:5px"></i>
        <div><b>뒤집은 경계 간선</b> — 교정에서 방향을 바꾼 간선.
        화살표 = <u>바꾼 뒤</u> 방향.</div></div>
      <div class="lg"><i class="sw" style="background:#3a4150"></i>
        <div><b>일반 간선</b> — 구역 내부 간선 (방향 유지).</div></div>
      <div class="dim" style="margin-top:8px"><b>평균 우회율</b>: 모든 정점 쌍에 대해
      "일방통행 거리 ÷ 양방향 거리"의 평균. 1.0이면 우회가 전혀 없다는 뜻.
      낮을수록 좋다. 값이 유한하면 어디서 어디로든 갈 수 있음(강연결)이 보장된
      상태다.</div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const PAL = ["#4f9dff","#5ad07a","#c78bff","#ffd24a","#ff8a5c","#37d3d3",
             "#ff6ea8","#a3e635","#94a3b8","#f472b6","#7dd3fc","#fca5a5"];
const W = 720, PAD = 22;
let current = 0;

function xy(verts, v) {
  const p = verts[v];
  return [PAD + p[0] * (W - 2 * PAD), PAD + (1 - p[1]) * (W - 2 * PAD)];
}
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent, title) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (title) {
    const t = document.createElementNS(NS, "title");
    t.textContent = title;
    e.appendChild(t);
  }
  parent.appendChild(e);
  return e;
}

function render(idx) {
  current = idx;
  const g = DATA.graphs[idx];
  const gF = document.getElementById("gFaces");
  const gE = document.getElementById("gEdges");
  const gX = document.getElementById("gFlips");
  gF.innerHTML = gE.innerHTML = gX.innerHTML = "";

  for (const f of g.faces) {
    if (f.macro < 0) continue;
    const pts = f.verts.map(v => xy(g.verts, v).join(",")).join(" ");
    el("polygon", { points: pts, fill: PAL[f.macro % PAL.length],
                    opacity: 0.10 }, gF);
  }
  for (const e of g.edges) {
    const [x1, y1] = xy(g.verts, e.u), [x2, y2] = xy(g.verts, e.v);
    if (e.flipped) continue;
    el("line", { x1, y1, x2, y2,
      stroke: e.door ? "#3d5a80" : "#3a4150",
      "stroke-width": e.door ? 2 : 1.2 }, gE,
      (e.door ? "경계 간선" : "일반 간선")
      + ` #${e.eid}  방향 ${e.after[0]}→${e.after[1]}`);
  }
  for (const e of g.edges) {
    if (!e.flipped) continue;
    const [x1, y1] = xy(g.verts, e.after[0]), [x2, y2] = xy(g.verts, e.after[1]);
    // 화살표가 정점에 묻히지 않게 살짝 안쪽으로
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
    const sx = x1 + dx * 0.12, sy = y1 + dy * 0.12;
    const ex = x1 + dx * 0.88, ey = y1 + dy * 0.88;
    el("line", { x1: sx, y1: sy, x2: ex, y2: ey, stroke: "#ff9b3d",
      "stroke-width": 3.2, "marker-end": "url(#arrow)" }, gX,
      `뒤집은 경계 간선 #${e.eid}  ${e.base[0]}→${e.base[1]} 이던 것을 `
      + `${e.after[0]}→${e.after[1]} 로 변경`);
  }

  const imp = 100 * (g.stretch_base - g.stretch_after) / g.stretch_base;
  document.getElementById("stats").innerHTML = `
    <h2>그래프 ${g.seed}</h2>
    <div class="big">우회율 ${imp.toFixed(2)}% 감소</div>
    <div class="row"><span>평균 우회율 (교정 전)</span><span>${g.stretch_base.toFixed(4)}</span></div>
    <div class="row"><span>평균 우회율 (교정 후)</span><span>${g.stretch_after.toFixed(4)}</span></div>
    <div class="row"><span>뒤집은 경계 간선</span><span>${g.n_flip}개 / 전체 ${g.n_door}개</span></div>
    <div class="dim" style="margin-top:6px">교정 규칙: 경계 간선을 하나씩 뒤집어 보고,
    평균 우회율이 낮아질 때만 채택. 어디로든 갈 수 있는 성질(강연결)이 깨지는
    뒤집기는 자동으로 버려진다.</div>`;

  document.querySelectorAll("#seedbar button").forEach((b, i) =>
    b.classList.toggle("active", i === idx));
}

const bar = document.getElementById("seedbar");
DATA.graphs.forEach((g, i) => {
  const imp = 100 * (g.stretch_base - g.stretch_after) / g.stretch_base;
  const b = document.createElement("button");
  b.innerHTML = `그래프 ${g.seed}<br><b>−${imp.toFixed(2)}%</b>`;
  b.onclick = () => render(i);
  bar.appendChild(b);
});
render(0);
</script>
</body>
</html>
"""


def main() -> None:
    data = json.loads(SRC.read_text())
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
