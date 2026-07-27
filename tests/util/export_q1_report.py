"""door_outer_flip_scan.json + door_q1_stage2.json → Q1 완전 판정 보고서 HTML.

    python tests/util/export_q1_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

SCAN = Path("tests/util/door_outer_flip_scan.json")
STAGE2 = Path("tests/util/door_q1_stage2.json")
OUT = Path("tests/util/door_q1_report.html")

STYLE = """
  body { margin: 0 auto; font: 14px/1.7 system-ui, sans-serif; background: #14161a;
         color: #dde; max-width: 1100px; padding: 24px 32px 48px; }
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


def main() -> None:
    scan = json.loads(SCAN.read_text())
    st2 = json.loads(STAGE2.read_text())
    st2_by = {(r["seed"], r["macro"]): r for r in st2["rows"]}

    # --- stage A 표 ---
    rows_a = []
    n_macros = n_ptrue = 0
    n_balanced = n_simple = 0
    for g in scan["graphs"]:
        for m in g["macros"]:
            n_macros += 1
            circ = m["outline"]
            n_balanced += circ["balanced"]
            n_simple += circ["simple_cycles"]
            if m["p_flip_exists"]:
                n_ptrue += 1
            p = ('<span class="warn">TRUE (S_do ⊋ S_fc)</span>'
                 if m["p_flip_exists"] else '<span class="ok">false (S_do = S_fc)</span>')
            o = ("단순 사이클" if circ["simple_cycles"]
                 else ("폐곡선(비단순)" if circ["balanced"]
                       else '<span class="bad">불균형!</span>'))
            rows_a.append(
                f"<tr><td>seed {g['seed']} / macro {m['macro']}</td>"
                f"<td>{m['nv']}</td><td>{m['n_door']}</td><td>{m['n_outer']}</td>"
                f"<td>{m['n_intern']}</td><td>{o}</td>"
                f"<td>{m['n_feasible_outer']:,}</td><td>{p}</td></tr>")

    # --- stage 2 표 ---
    rows_b = []
    exact_gains, budget_gains = [], []
    for r in st2["rows"]:
        g = r["gain_pct"]
        (exact_gains if r["mode"] == "exact" else budget_gains).append(g or 0.0)
        gtxt = ("" if g is None else
                (f'<span class="warn">DO 개선 −{g:.2f}%</span>' if g > 1e-9
                 else ('<span class="ok">동일</span>' if abs(g) <= 1e-9
                       else f'<span class="dim">FC 우세 {-g:.2f}%</span>')))
        sweep = "" if r["outer_sweep_best"] is None else f"{r['outer_sweep_best']:.0f}"
        rows_b.append(
            f"<tr><td>seed {r['seed']} / macro {r['macro']}</td><td>{r['nv']}</td>"
            f"<td>{r['n_outer']}</td><td>{r['n_intern']}</td>"
            f"<td>{r['mode']}</td><td>{r['best_fc']:.0f}</td>"
            f"<td>{r['best_do']:.0f}</td><td>{sweep}</td><td>{gtxt}</td></tr>")

    max_exact = max((g for g in exact_gains), default=0.0)
    max_budget = max((g for g in budget_gains), default=0.0)

    html = f"""<!doctype html>
<html lang="ko">
<head><meta charset="utf-8"><title>Q1 완전 판정 — 외곽 자유도와 best 개선</title>
<style>{STYLE}</style></head>
<body>
<h1>Q1 완전 판정: face 방향 카피(canonical) 대비 외곽 자유화의 실제 이득</h1>
<p class="dim">V={scan['n_vertices']} &middot; target_k={scan['target_k']} &middot;
remove_ratio={scan['remove_ratio']} &middot; 그래프 5개 (seed {scan['seeds']}) &middot;
생성: door_outer_flip_scan.py, door_q1_stage2.py (멀티프로세싱 14 proc)</p>

<h2>1. 가정과 질문</h2>
<div class="box">
<p><b>용어.</b> canonical = FaceClusterPartition 이 face 순회 순서에서 카피해 오는
경계 간선 방향. S_fc = door·외곽 canonical 고정 + 내부 자유의 강연결 배향 집합.
S_do = door 만 canonical 고정 + 외곽·내부 자유의 강연결 배향 집합. S_fc ⊆ S_do.</p>
<p><b>검증 항목 1.</b> canonical 을 카피하면 macro 외곽선이 방향 폐곡선(원형)이 되는가?</p>
<p><b>검증 항목 2 (Q1).</b> 100-seed 솔버 벤치의 FC≈DO 동률이 "S_do 에 더 좋은 해가
없어서"인가, "있는데 SA 가 못 찾아서"인가? 앞선 소 macro 전수는 S_do = S_fc
(집합 자체 동일)를 시사했으나 큰 macro 는 미검증이었다.</p>
</div>

<h2>2. 평가 기준과 방법 (전 macro 손실 없는 판정)</h2>
<div class="box">
<p><b>1단계 — 집합 판정 (exact, 전 macro).</b> 명제 P(macro): canonical 아닌
외곽 조합 중 어떤 내부 배향으로든 강연결이 되는 것이 존재하는가.
내부 자유도는 전수 대신 <b>Boesch–Tindell 정리</b>로 정확 판정한다: macro 는
bridgeless(면 사이클 논증 + 실측 0 bridge)이므로 "door·외곽 고정 + 내부 양방향"
mixed graph 가 강연결이면 내부 배향이 존재한다. 계산은 동치 축약 후 전수:
① 내부 연결성분 supernode 축약(내부는 양방향이라 SC 판정 동치),
② 외곽 직렬 체인 병합(중간 정점 source/sink 금지로 2상태뿐),
③ 양끝이 같은 supernode 인 외곽 간선은 self-loop → 방향 무관 항상 유효.
이로써 외곽 32비트(2^32)도 실질 변수 ≤19개로 줄어 전 macro 전수 완료.</p>
<p><b>2단계 — best 비교 (P=TRUE macro 만).</b>
총 자유비트(외곽+내부) ≤ 2^{st2['exact_cap']} 이면 <b>전수 = 참값(exact)</b>.
초과 big macro 는 <b>동일 예산 페어드 탐색(budgeted)</b>: 두 팔 모두 greedy
mixed-SC 구성 초기해 + 단일비트 flip 언덕오르기, restart {st2['n_restarts']}회 ×
평가 {st2['eval_budget']:,}회. 차이는 탐색 변수 범위뿐(FC=내부만, DO=내부+외곽).
보조로 FC 최적해의 내부를 고정하고 외곽 조합만 바꾸는 sweep 교차검사.
목적 함수는 macro-국소 APSP 합(hop) — 전역 stretch 와 다르므로 방향성 지표.</p>
</div>

<h2>3. 결과 — 1단계: 외곽 폐곡선 여부와 집합 판정</h2>
<table>
<tr><th>macro</th><th>|V|</th><th>door</th><th>외곽</th><th>내부</th>
<th>canonical 외곽 형태</th><th>유효 외곽 조합 수</th><th>P (비-canonical 유효 존재)</th></tr>
{''.join(rows_a)}
</table>
<div class="box">
<p><b>항목 1 답: 폐곡선 맞음.</b> 전 {n_macros}개 macro 에서 canonical outline 은
모든 outline 정점이 in-deg = out-deg (balanced {n_balanced}/{n_macros}) —
방향 폐곡선(들)로 분해된다. {n_simple}개는 모든 정점 in=out=1 인 단순 사이클,
나머지는 한 정점을 두 번 지나는 비단순 폐곡선.</p>
<p><b>집합 판정: 소 macro 결론은 일반화되지 않았다.</b> {n_macros}개 중
{n_ptrue}개 macro 에서 P=TRUE — 내부 간선이 조밀한 큰 macro 는 canonical 아닌
유효 외곽 조합이 다수 존재(최대 4.2M개). 주요 원인은 양끝이 내부로 이미 연결된
외곽 간선(self-loop 형): 뒤집어도 강연결이 안 깨진다. door 지배적 소 macro 는
전부 P=false (어제의 |S_do| = |S_fc| 관찰과 정합 — 검증 통과).</p>
</div>

<h2>4. 결과 — 2단계: best(S_do) vs best(S_fc)</h2>
<table>
<tr><th>macro</th><th>|V|</th><th>외곽</th><th>내부</th><th>판정 방식</th>
<th>best FC</th><th>best DO</th><th>outer sweep</th><th>DO 이득</th></tr>
{''.join(rows_b)}
</table>

<h2>5. 결론</h2>
<div class="box concl">
<p><b>항목 1:</b> face 방향 카피 → 외곽은 항상 방향 폐곡선. 구조 보장 확인.</p>
<p><b>Q1 최종 답: "못 찾은 해"는 실존한다 — 단, 작고, 공간이 커진 만큼 찾기는
더 어려워진다.</b> 근거 세 겹:</p>
<p>① <b>exact(참값)</b>: seed 42 macro 2 에서 best(S_do)=1271 &lt; best(S_fc)=1330
(−{max_exact:.1f}%). 외곽 자유화가 이론상 더 좋은 해를 품는 것이 전수로 확정.
(나머지 exact 2건은 동일.)</p>
<p>② <b>outer sweep(구성적 증거)</b>: budgeted macro 에서도 FC 최적해의 내부를
고정한 채 외곽 조합만 바꿔 개선이 나온 사례 2건 (seed 2 macro 2: 229,095→228,569;
seed 3 macro 3: 3,420→3,321, −2.9%). 즉 FC 가 도달한 지점에서도 외곽 flip 만으로
더 내려갈 길이 있었다.</p>
<p>③ <b>budgeted 혼조가 동률의 메커니즘을 설명</b>: 동일 예산에서 DO 가 이긴
macro (+0.2~+2.0%)와 진 macro (−0.9~−5.3%)가 공존. 공간이 커지면 좋은 해도
늘지만 같은 예산으론 탐색이 어려워진다 — 100-seed 솔버 벤치의 동률(±0.04%)과
DO 시간 +16% 는 정확히 이 트레이드오프의 표현이다.</p>
<p><b>실용 판단.</b> 외곽 자유화를 "SA 변수 추가"로 쓰는 것은 이득 없음이 재확인
(여지는 수 % 국소, 탐색 난이도와 시간 비용이 상쇄). 여지를 캐는 싼 길은 후처리:
FC 해를 받아 self-loop 형 외곽 간선/외곽 조합만 국소 열거로 flip 최적화
(outer sweep 이 그 프로토타입 — macro 당 수 초). 별도 후속 실험 거리.</p>
</div>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
