"""Render a bilingual showcase HTML: both teacher prompts + sample results + the
blind-audit verdicts for the chosen passages.

Usage:
    python dataset/prompt_showcase.py
Writes outputs/graphrag_pilot/showcase.html.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract import render_teacher_prompt  # noqa: E402
from kg_contract.graphrag_prompts import GRAPH_EXTRACTION_PROMPT  # noqa: E402
from kg_contract.prompts import TEACHER_SECTIONS_PROSE  # noqa: E402

from dataset.graphrag_pilot import ENTITY_TYPES  # noqa: E402

OUT = Path("outputs/graphrag_pilot")
PILOT_IDS = ["wikipedia-01557", "wikipedia-02266"]


def _load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def _audit(audits: list[dict], cid: str) -> dict:
    for a in audits:
        if a["id"] == cid:
            return a
    return {}


def _chip(label: str, grade: str) -> str:
    cls = {"GOOD": "good", "FAIR": "fair", "POOR": "poor", "LOW": "good", "MODERATE": "fair", "HIGH": "poor"}.get(grade, "")
    return f"<span class='chip {cls}'>{html.escape(label)}: {html.escape(grade)}</span>"


def _twenty_block(row: dict, aud: dict) -> str:
    rows = "".join(
        f"<li><span class='en'>{html.escape(t['source']['title'])}</span> "
        f"<b>[{html.escape(t['relation']['type'])},{t['relation']['weight']}]</b> "
        f"<span class='en'>{html.escape(t['target']['title'])}</span></li>"
        for t in row.get("triplets", [])
    )
    a = aud.get("twenty", {})
    chips = "".join(
        _chip(k, v) for k, v in a.items() if k in ("faithfulness", "completeness", "noise")
    )
    return f"""
<div class='col'>
  <h3>20-set teacher — {len(row.get('triplets', []))} triplets</h3>
  <ul class='trips'>{rows or '<li class=muted>无</li>'}</ul>
  <div class='audit'><b>AI 盲评：</b>{chips}
    <p class='num'>数字保留：{html.escape(a.get('numeric_preserved', '-'))}</p>
    <p class='num'>数字丢失：{html.escape(a.get('numeric_dropped', '-'))}</p>
  </div>
</div>"""


def _graphrag_block(row: dict, aud: dict, fixed_row: dict | None = None) -> str:
    primary = fixed_row or row
    ents = "".join(
        f"<li><span class='en'>{html.escape(e['title'])}</span> <i>({html.escape(e['type'])})</i> — "
        f"{html.escape(e['description'][:90])}{'…' if len(e['description']) > 90 else ''}</li>"
        for e in primary.get("entities", [])
    )
    rels = "".join(
        f"<li><span class='en'>{html.escape(r['source'])}</span> <b>[{r['strength']:.0f}]</b> "
        f"{html.escape(r['description'][:70])}{'…' if len(r['description']) > 70 else ''} "
        f"→ <span class='en'>{html.escape(r['target'])}</span></li>"
        for r in primary.get("relationships", [])
    )
    a = aud.get("graphrag", {})
    chips = "".join(
        _chip(k, v) for k, v in a.items() if k in ("faithfulness", "completeness", "noise")
    )
    fix_note = ""
    if fixed_row is not None:
        fix_note = (
            f"<p class='fix'>修复后（当前方案）：{len(fixed_row.get('entities', []))} 实体 / "
            f"{len(fixed_row.get('relationships', []))} 关系 — 修复前（盲评原版）："
            f"{len(row.get('entities', []))} 实体 / {len(row.get('relationships', []))} 关系</p>"
        )
    return f"""
<div class='col'>
  <h3>GraphRAG teacher — {len(primary.get('entities', []))} 实体 / {len(primary.get('relationships', []))} 关系</h3>
  {fix_note}
  <details open><summary>实体（{len(primary.get('entities', []))}）</summary><ul class='trips'>{ents or '<li class=muted>无</li>'}</ul></details>
  <details><summary>关系（{len(primary.get('relationships', []))}）</summary><ul class='trips'>{rels or '<li class=muted>无</li>'}</ul></details>
  <div class='audit'><b>AI 盲评（修复前原版）：</b>{chips}
    <p class='num'>数字保留：{html.escape(a.get('numeric_preserved', '-'))}</p>
    <p class='num'>数字丢失：{html.escape(a.get('numeric_dropped', '-'))}</p>
  </div>
</div>"""


def _passage_card(cid: str, twenty: dict, graphrag: dict, fixed: dict | None, aud: dict, trans: dict) -> str:
    pas = twenty["text"]
    pas_zh = trans["passages"].get(cid, "")
    note = aud.get("note", "")
    return f"""
<section class='card'>
  <h2>{html.escape(cid)}</h2>
  <details class='passage' open>
    <summary>Passage (EN · {len(pas.split())} 词)</summary>
    <pre class='en'>{html.escape(pas)}</pre>
    {f'<pre class="zh">{html.escape(pas_zh)}</pre>' if pas_zh else ''}
  </details>
  <p class='vnote'><b>盲评注：</b>{html.escape(note)}</p>
  <div class='cols'>
    {_twenty_block(twenty, aud)}
    {_graphrag_block(graphrag, aud, fixed)}
  </div>
</section>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=PILOT_IDS)
    args = parser.parse_args(argv)

    graphrag = _load_rows(OUT / "graphrag.jsonl")
    twenty = _load_rows(OUT / "twenty.jsonl")
    fixed = _load_rows(OUT / "graphrag_fixed.jsonl")
    audits = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    trans_path = OUT / "translations_zh.json"
    if not trans_path.exists():
        trans_path = Path("outputs/prompt_dev/translations_zh.json")
    trans = json.loads(trans_path.read_text(encoding="utf-8")) if trans_path.exists() else {}

    twenty_prompt = render_teacher_prompt(TEACHER_SECTIONS_PROSE)
    graphrag_prompt = GRAPH_EXTRACTION_PROMPT.format(entity_types=ENTITY_TYPES, input_text="<passage text>")

    cards = "".join(
        _passage_card(cid, twenty[cid], graphrag[cid], fixed.get(cid), _audit(audits, cid), trans)
        for cid in args.ids if cid in twenty and cid in graphrag
    )

    html_out = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Teacher prompt 展示 — prompt · 结果 · AI 审计</title>
<style>
  body{{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
       margin:0;padding:24px;background:#f5f6f8;color:#1c1e21;}}
  h1{{font-size:20px;margin:0 0 4px;}} p.sub{{color:#666;font-size:13px;margin:0 0 18px;}}
  .prompts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;}}
  @media (max-width:1000px){{.prompts{{grid-template-columns:1fr;}}}}
  .pcard{{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:14px 16px;}}
  .pcard h2{{font-size:14px;margin:0 0 8px;}} .pcard h2 span{{color:#666;font-weight:400;font-size:12px;}}
  details pre{{white-space:pre-wrap;background:#fafbfc;border:1px solid #eee;border-radius:6px;padding:10px;
       font-size:12px;line-height:1.5;max-height:340px;overflow:auto;font-family:ui-monospace,Consolas,monospace;}}
  .card{{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:16px 20px;margin-bottom:18px;}}
  .card h2{{font-size:15px;margin:0 0 8px;}}
  .passage pre{{white-space:pre-wrap;background:#fafbfc;border:1px solid #eee;border-radius:6px;padding:10px;font-size:13px;line-height:1.55;margin:4px 0;}}
  .vnote{{color:#57606a;font-size:13px;margin:8px 0;}}
  .cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
  @media (max-width:1000px){{.cols{{grid-template-columns:1fr;}}}}
  .col{{background:#fafbfc;border:1px solid #eee;border-radius:8px;padding:12px 14px;}}
  .col h3{{font-size:13px;margin:0 0 8px;color:#24292f;}}
  ul.trips{{margin:0;padding-left:18px;font-size:12.5px;line-height:1.55;}}
  ul.trips li{{margin-bottom:2px;}}
  .en{{font-family:ui-monospace,Consolas,monospace;}}
  .muted{{color:#999;}}
  summary{{cursor:pointer;font-size:12.5px;color:#57606a;}}
  .audit{{margin-top:8px;background:#fff;border:1px solid #e1e4e8;border-radius:6px;padding:8px 10px;font-size:12.5px;}}
  .audit p{{margin:4px 0 0;}}
  .chip{{display:inline-block;border-radius:10px;padding:1px 8px;font-size:11.5px;font-weight:600;margin:2px 4px 2px 0;}}
  .good{{background:#dafbe1;color:#1a7f37;}} .fair{{background:#fff8c5;color:#9a6700;}} .poor{{background:#ffebe9;color:#cf222e;}}
  .num{{color:#57606a;}} .fix{{color:#1a7f37;font-size:12px;margin:4px 0;}}
</style></head><body>
<h1>Teacher prompt 展示 — prompt · 结果 · AI 盲评</h1>
<p class="sub">Phase-0 试点：同 10 条语料，20-set teacher vs GraphRAG teacher。审计为 blind subagent 三轴判定。</p>
<div class="prompts">
  <div class="pcard">
    <h2>20-set teacher <span>（当前 kg_contract/prompts.py）</span></h2>
    <details><summary>展开完整 prompt（5 节）</summary><pre>{html.escape(twenty_prompt)}</pre></details>
  </div>
  <div class="pcard">
    <h2>GraphRAG teacher <span>（microsoft/graphrag 官方 prompt，entity_types={html.escape(ENTITY_TYPES)}）</span></h2>
    <details><summary>展开完整 prompt（含 3 个官方示例）</summary><pre>{html.escape(graphrag_prompt)}</pre></details>
  </div>
</div>
{cards}
</body></html>"""

    out = OUT / "showcase.html"
    out.write_text(html_out, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
