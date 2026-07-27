"""door_combo_enum.json → 실험 보고서 HTML 생성.

    python tests/util/export_door_combo_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("tests/util/door_combo_enum.json")
OUT = Path("tests/util/door_combo_report.html")

STYLE = """
  body { margin: 0; font: 14px/1.7 system-ui, sans-serif; background: #14161a;
         color: #dde; max-width: 1080px; padding: 24px 32px 48px; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 34px; color: #cde;
       border-bottom: 1px solid #2c3038; padding-bottom: 6px; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  td, th { border: 1px solid #333a44; padding: 4px 10px; text-align: right; }
  th { color: #9ab; font-weight: 600; } td:first-child, th:first-child { text-align: left; }
  .ok { color: #6ec86e; font-weight: 600; } .warn { color: #f5a623; font-weight: 600; }
  .bad { color: #e86a6a; font-weight: 600; } .dim { color: #78828e; }
  code { background: #1c1f24; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  .box { background: #1c1f24; border: 1px solid #2c3038; border-radius: 8px;
         padding: 12px 18px; margin: 14px 0; }
  .concl { border-left: 4px solid #4f9dff; }
"""


def fmt_arm(a: dict) -> tuple[str, str]:
    if a.get("skip"):
        return (f'<span class="dim">skip(2^{a["bits"]})</span>', "")
    return (str(a["n"]), "" if a["best"] is None else f'{a["best"]:.0f}')


def macro_rows(cfg: dict) -> str:
    rows = []
    for m in cfg["macros"]:
        fc_n, fc_b = fmt_arm(m["fc"])
        do_n, do_b = fmt_arm(m["do"])
        al_n, al_b = fmt_arm(m["all"])
        q1 = ""
        if not m["fc"].get("skip") and not m["do"].get("skip"):
            same_set = m["fc"]["n"] == m["do"]["n"]
            same_best = m["fc"]["best"] == m["do"]["best"]
            q1 = ('<span class="ok">집합·최적 동일</span>' if same_set and same_best
                  else ('<span class="ok">최적 동일</span>' if same_best
                        else '<span class="bad">개선 존재</span>'))
        q2 = ""
        if not m["do"].get("skip") and not m["all"].get("skip") \
                and m["do"]["best"] is not None and m["all"]["best"] is not None:
            if m["all"]["best"] < m["do"]["best"]:
                gain = 100 * (m["do"]["best"] - m["all"]["best"]) / m["do"]["best"]
                q2 = f'<span class="warn">−{gain:.1f}%</span>'
            else:
                q2 = '<span class="ok">0%</span>'
        rows.append(
            f"<tr><td>macro {m['macro']}</td><td>{m['nv']}</td>"
            f"<td>{m['n_door']}</td><td>{m['n_outer']}</td><td>{m['n_intern']}</td>"
            f"<td>{fc_n}</td><td>{do_n}</td><td>{al_n}</td>"
            f"<td>{fc_b}</td><td>{do_b}</td><td>{al_b}</td>"
            f"<td>{q1}</td><td>{q2}</td></tr>"
        )
    return "\n".join(rows)


def config_section(cfg: dict) -> str:
    return f"""
<h2>그래프 V={cfg['n_vertices']} E={cfg['n_edges']} (target_k={cfg['target_k']})</h2>
<table>
<tr><th>macro</th><th>|V|</th><th>door</th><th>외곽</th><th>내부</th>
<th>|S_fc|</th><th>|S_do|</th><th>|S_all|</th>
<th>best fc</th><th>best do</th><th>best all</th>
<th>Q1: 외곽 자유 이득</th><th>Q2: door 자유 이득(상한)</th></tr>
{macro_rows(cfg)}
</table>
"""


def main() -> None:
    data = json.loads(SRC.read_text())
    sections = "\n".join(config_section(c) for c in data["configs"])
    html = f"""<!doctype html>
<html lang="ko">
<head><meta charset="utf-8"><title>Door 배향 해공간 전수열거 보고서</title>
<style>{STYLE}</style></head>
<body>
<h1>Door 배향 해공간 전수열거 — S_fc ⊂ S_do ⊂ S_all</h1>
<p class="dim">seed={data['seed']} &middot; remove_ratio={data['remove_ratio']} &middot;
열거 상한 2^{data['enum_cap']} &middot; 생성: tests/util/door_combo_enum.py</p>

<h2>1. 배경과 가정</h2>
<div class="box">
<p><b>선행 실측 (100-seed 솔버 벤치, V=200, DnC+SA):</b>
face-cycle(FC: door+외곽선 전부 face 순회 방향으로 고정)과
door-only(DO: 진짜 door 만 고정, 외곽선은 자유 변수)가
SC 100/100 동률, stretch −0.04%, flow −0.14% — 통계적 완전 동률. DO 만 시간 +16%.</p>
<p><b>가정(질문) 두 가지:</b></p>
<p><b>Q1.</b> DO 의 해공간은 FC 를 포함하고 더 크다 (S_fc ⊂ S_do).
동률이 나온 것은 (a) 더 좋은 해가 있는데 SA 가 못 찾은 것인가,
(b) 애초에 더 좋은 해가 없는 것인가?</p>
<p><b>Q2.</b> door 방향 조합까지 바꾸면(S_all) 성능 개선 여지가 있는가?</p>
</div>

<h2>2. 실험 설계</h2>
<div class="box">
<p>SA 를 완전히 배제하고 macro 부분그래프 단위로 배향을 <b>전수열거</b>한다.
강연결(SC)인 배향만 유효해로 간주하고, 각 유효해의 macro-국소 APSP 합
(모든 순서쌍 방향 최단경로 hop 합, 낮을수록 좋음)의 최소값을 팔 별로 비교한다.</p>
<table>
<tr><th>팔</th><th>고정(canonical=face 순회 방향)</th><th>자유(전수열거)</th><th>대응</th></tr>
<tr><td>S_fc</td><td>door + 외곽</td><td>내부</td><td>face-cycle</td></tr>
<tr><td>S_do</td><td>door</td><td>외곽 + 내부</td><td>door-only</td></tr>
<tr><td>S_all</td><td>없음</td><td>전부 (door 포함)</td><td>door 조합 탐색 상한</td></tr>
</table>
<p><b>한계 (해석 시 필수):</b>
① 2^n 폭발 때문에 자유 간선 ≤ {data['enum_cap']}개인 팔만 열거 — 큰 macro 는 skip,
결론은 열거 가능한 macro 기준의 부분 증거.
② S_all 은 macro 를 단독으로 떼어 door 를 뒤집으므로 이웃 macro 와의 정합성
(door 방향은 양쪽 macro 가 공유)을 무시한다 → S_all 이득은 <b>개선 상한(낙관치)</b>.
③ 지표가 macro-국소 hop APSP 라 전역 stretch 와 동일하지 않다 — 방향성 판단용.</p>
</div>

<h2>3. 결과</h2>
{sections}

<h2>4. 결론</h2>
<div class="box concl">
<p><b>Q1 답: (b) — 동률은 구조적이다. SA 탓이 아니다.</b>
열거 가능한 모든 macro 에서 best(S_do) = best(S_fc) 였고, 심지어 외곽 자유도가
있는 macro(외곽 &gt; 0)에서도 <b>유효 해집합 크기 자체가 동일</b>했다
(|S_do| = |S_fc|). 즉 외곽선을 뒤집는 어떤 조합도 강연결을 유지하지 못한다 —
외곽선은 형식상 자유 변수일 뿐 실질적으로는 죽은 자유도다. (macro 외곽은 쌍대
컷 구조라 부분 뒤집기가 SC 를 깨는 것과 정합.) 100-seed 벤치의 동률은 "SA 가
못 찾은" 게 아니라 "찾을 것이 없어서"다.</p>
<p><b>Q2 답: 상한 기준 여지 있음, 단 낙관치.</b>
door 가 지배적인 작은 macro 에서 door 조합 개방 시 macro-국소 APSP 최적이
3~5% 내려갔다 (V=200 macro5: 212→202, V=100 macro5: 871→841). 유효 해집합도
크게 늘었다 (12→216 등). 그러나 이는 이웃 정합성을 무시한 상한이며, door 는
양쪽 macro 공유라 실제로는 결합 최적화(이웃 macro 와 동시 조정)가 필요하다.
또한 큰 macro 에서의 여지는 미측정.</p>
<p><b>실용 판단:</b> 외곽선 자유화(door-only)는 이득 없음이 구조적으로 확정 —
품질 동일, 시간 +16% 손해. 개선을 원하면 방향은 door 조합의 결합 탐색이며,
기대 이득 상한은 국소 APSP 기준 한 자릿수 %.</p>
</div>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
