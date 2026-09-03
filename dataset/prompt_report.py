"""Render teacher-prompt dev runs as a bilingual (EN/ZH) HTML review report.

Reads raw run rows from outputs/prompt_dev/iterN.jsonl and
heldout_iterN.jsonl, plus translations + gold from
outputs/prompt_dev/translations_zh.json, and writes a self-contained
outputs/prompt_dev/review.html.

Usage:
    python dataset/prompt_report.py --iter 7 --heldout-iter 8
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

OUT_DIR = Path("outputs/prompt_dev")


def _load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["case_id"]] = r
    return rows


def _zh(titles: dict, rels: dict, s: str) -> str:
    return titles.get(s, s)


def _triplet_en(t: dict) -> str:
    return f"{t['source']['title']} -[{t['relation']['type']},{t['relation']['weight']}]- {t['target']['title']}"


def _triplet_zh(t: dict, titles: dict, rels: dict) -> str:
    s = _zh(titles, rels, t["source"]["title"])
    r = rels.get(t["relation"]["type"], t["relation"]["type"])
    o = _zh(titles, rels, t["target"]["title"])
    return f"{s} -[{r},{t['relation']['weight']}]- {o}"


def _card(case_id: str, row: dict, data: dict, gold: list, verdict: dict) -> str:
    titles = data["entities"]
    rels = data["relations"]
    passage_zh = data["passages"].get(case_id, "")
    rows_html = []
    for i, t in enumerate(row["triplets"], 1):
        rows_html.append(
            f"<tr><td class='num'>{i}</td>"
            f"<td class='en'>{html.escape(_triplet_en(t))}</td>"
            f"<td class='zh'>{html.escape(_triplet_zh(t, titles, rels))}</td></tr>"
        )
    viols = ""
    if row["semantic_violations"]:
        items = [
            f"<li>{html.escape(v['rule'])}：{html.escape(v['detail'])}</li>"
            for v in row["semantic_violations"]
        ]
        viols = f"<div class='viol'><b>语义校验违例：</b><ul>{''.join(items)}</ul></div>"
    gold_html = "".join(f"<li>{html.escape(g)}</li>" for g in gold)
    v_verdict = verdict.get("verdict", "") if verdict else ""
    v_note = verdict.get("note", "") if verdict else ""
    iter_label = row.get("iter", "?")
    return f"""
<section class='card'>
  <header>
    <h2>{html.escape(case_id)} <span class='diff'>{html.escape(row['difficulty'])}</span>
        <span class='meta'>iter {iter_label} · {len(row['triplets'])} 三元组 · 违例 {len(row['semantic_violations'])}</span>
        <span class='verdict'>{html.escape(v_verdict)}</span></h2>
    <p class='note'>{html.escape(v_note)}</p>
  </header>
  <div class='cols'>
    <div class='passage'>
      <h3>原文 Passage (EN)</h3>
      <pre class='en'>{html.escape(row['passage'])}</pre>
      <h3>译文 (ZH)</h3>
      <pre class='zh'>{html.escape(passage_zh)}</pre>
    </div>
    <div class='side'>
      <h3>抽取的三元组</h3>
      <table><thead><tr><th>#</th><th>EN</th><th>中文</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody></table>
      {viols}
      <h3>关键事实清单（gold，参考）</h3>
      <ul class='gold'>{gold_html}</ul>
    </div>
  </div>
</section>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter", type=int, default=7)
    parser.add_argument("--heldout-iter", type=int, default=8)
    parser.add_argument("--cases", nargs="*", default=["wikipedia-01557", "wikipedia-02266"])
    args = parser.parse_args(argv)

    data = json.loads((OUT_DIR / "translations_zh.json").read_text(encoding="utf-8"))
    tuning = _load_rows(OUT_DIR / f"iter{args.iter}.jsonl")
    heldout = _load_rows(OUT_DIR / f"heldout_iter{args.heldout_iter}.jsonl")

    cards = []
    for case_id in args.cases:
        row = heldout.get(case_id) or tuning.get(case_id)
        if row is None:
            sys.stderr.write(f"warning: {case_id} not found in iter{args.iter} or heldout_iter{args.heldout_iter}\n")
            continue
        cards.append(_card(case_id, row, data, data["gold"].get(case_id, []), data["verdicts"].get(case_id, {})))

    html_out = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Teacher prompt 抽检 — 困难 + 留出集</title>
<style>
  body{{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
       margin:0;padding:24px;background:#f5f6f8;color:#1c1e21;}}
  h1{{font-size:20px;margin:0 0 4px;}}
  p.sub{{color:#666;font-size:13px;margin:0 0 20px;}}
  .card{{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:18px 22px;margin-bottom:22px;
        box-shadow:0 1px 2px rgba(0,0,0,.04);}}
  .card header{{border-bottom:1px solid #eee;padding-bottom:10px;margin-bottom:14px;}}
  h2{{font-size:16px;margin:0 0 6px;}}
  .diff{{background:#2f81f7;color:#fff;border-radius:4px;padding:1px 8px;font-size:12px;font-weight:600;}}
  .meta{{color:#57606a;font-size:12px;font-weight:400;margin-left:8px;}}
  .verdict{{float:right;font-size:13px;font-weight:700;color:#bf8700;}}
  .note{{color:#57606a;font-size:13px;margin:0;}}
  .cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
  @media (max-width:900px){{.cols{{grid-template-columns:1fr;}}}}
  h3{{font-size:13px;color:#57606a;margin:14px 0 6px;text-transform:none;}}
  pre{{white-space:pre-wrap;word-break:break-word;background:#fafbfc;border:1px solid #eee;
      border-radius:6px;padding:12px;font-size:13px;line-height:1.55;max-height:340px;overflow:auto;margin:0;}}
  pre.en{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}}
  pre.zh{{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  td,th{{border-bottom:1px solid #eee;padding:5px 6px;text-align:left;vertical-align:top;}}
  th{{color:#57606a;font-size:12px;font-weight:600;}}
  td.num{{color:#999;width:26px;}}
  td.en{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}}
  td.zh{{color:#444;}}
  .viol{{background:#fff8e6;border:1px solid #f0d78c;border-radius:6px;padding:8px 12px;margin-top:10px;font-size:13px;}}
  .viol ul{{margin:4px 0 0;padding-left:18px;}}
  .gold{{columns:2;font-size:13px;color:#444;padding-left:18px;margin:0;}}
  .gold li{{margin-bottom:2px;}}
  @media (max-width:900px){{.gold{{columns:1;}}}}
</style></head><body>
<h1>Teacher prompt 抽检 — 困难 &amp; 留出集</h1>
<p class="sub">盲评收敛后（iter7 prompt）· 数据源 outputs/prompt_dev/iter7.jsonl &amp; heldout_iter8.jsonl · 双语对照</p>
{''.join(cards)}
</body></html>"""

    out = OUT_DIR / "review.html"
    out.write_text(html_out, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
