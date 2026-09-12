"""预注册题库 — LightRAG e2e（ADR-0009 Leg B）。

全自然语言信息需求，按"信息需求类型"分类（不是图算子），预注册冻结。
题面用自然称谓，答案元素锚在 gold 实体上（用于 T1 检索层机械打分）。

    python eval/lightrag_questions.py --write    # 校验 + 写出 questions.json
    python eval/lightrag_questions.py --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "lightrag_e2e"
SLICE = OUT_DIR / "slice.json"
GOLD = ROOT / "dataset" / "data" / "graphrag_labels_full.jsonl"

# 类型：single 单实体事实 / link 跨段关联 / agg 聚合 / theme 全局主题
NATURAL_QUESTIONS = [
    # space (Apollo)
    dict(qid="S1", type="single", cluster="space",
         question="Who commanded the Apollo 8 mission?",
         answer_terms=["BORMAN", "APOLLO 8"],
         reference="Frank Borman commanded Apollo 8."),
    dict(qid="S2", type="link", cluster="space",
         question="What was the relationship between the backup crew and the prime crew of Apollo 8?",
         answer_terms=["BACKUP CREW", "PRIME CREW", "NEIL ARMSTRONG", "BUZZ ALDRIN", "LOVELL"],
         reference="The backup crew was announced with the prime crew; when Lovell moved to the prime crew, Aldrin became CMP and Haise joined as backup LMP."),
    dict(qid="S3", type="agg", cluster="space",
         question="What were the main milestones of the Apollo 8 mission?",
         answer_terms=["APOLLO 8", "EARTH", "MOON", "LUNAR ORBIT INSERTION", "FIRST TELEVISION PICTURES"],
         reference="Apollo 8 entered the Moon's sphere of influence and lunar orbit, and sent the first television pictures of Earth."),
    dict(qid="S4", type="single", cluster="space",
         question="Which spacecraft and rocket were used for the first crewed Moon landing?",
         answer_terms=["APOLLO 11", "SATURN V", "EAGLE", "COLUMBIA"],
         reference="Apollo 11 launched on a Saturn V; the Lunar Module Eagle landed while the Command Module Columbia stayed in lunar orbit."),
    # kurosawa
    dict(qid="K1", type="single", cluster="kurosawa",
         question="Which Kurosawa film was remade as A Fistful of Dollars?",
         answer_terms=["YOJIMBO", "A FISTFUL OF DOLLARS", "SERGIO LEONE"],
         reference="Yojimbo was remade by Sergio Leone as A Fistful of Dollars."),
    dict(qid="K2", type="link", cluster="kurosawa",
         question="How did George Lucas help Kurosawa get his later film produced?",
         answer_terms=["LUCAS", "KUROSAWA", "20TH CENTURY FOX", "FRANCIS FORD COPPOLA"],
         reference="Lucas used his influence at 20th Century Fox and recruited Francis Ford Coppola as co-producer."),
    dict(qid="K3", type="agg", cluster="kurosawa",
         question="Which later directors cite Kurosawa as an influence?",
         answer_terms=["WES ANDERSON", "ZACK SNYDER", "AKIRA KUROSAWA", "WESTERN CINEMA"],
         reference="Directors such as Wes Anderson and Zack Snyder cite Kurosawa as an influence."),
    # afroasiatic
    dict(qid="F1", type="single", cluster="afroasiatic",
         question="Why is the name Hamito-Semitic now considered outdated?",
         answer_terms=["HAMITO-SEMITIC", "SEMITIC", "HAMITIC", "TABLE OF NATIONS"],
         reference="The name came from Noah's sons Shem and Ham in the Table of Nations, a genealogy that does not reflect the languages' actual origins."),
    dict(qid="F2", type="agg", cluster="afroasiatic",
         question="Which branches are included in the Afroasiatic language family?",
         answer_terms=["AFROASIATIC", "OMOTIC", "CHADIC", "SEMITIC", "BERBER", "CUSHITIC"],
         reference="Semitic, Berber, Cushitic, Chadic, Omotic and Egyptian."),
    dict(qid="F3", type="theme", cluster="afroasiatic",
         question="What is known about the Proto-Afroasiatic language?",
         answer_terms=["PROTO-AFROASIATIC", "JOSEPH GREENBERG", "PREFIX CONJUGATION", "SUFFIX CONJUGATION"],
         reference="Greenberg grouped the family; reconstructed features include prefix and suffix conjugation patterns."),
    # andorra
    dict(qid="A1", type="agg", cluster="andorra",
         question="How is education organized in Andorra?",
         answer_terms=["EDUCATION", "ANDORRAN SCHOOLS", "FRENCH SCHOOLS", "SPANISH SCHOOLS", "UNIVERSITY OF ANDORRA"],
         reference="There are three school systems (Andorran, French, Spanish) plus the University of Andorra."),
    dict(qid="A2", type="single", cluster="andorra",
         question="Which company runs telecommunications in Andorra?",
         answer_terms=["ANDORRA TELECOM", "SOM", "OPTICAL FIBER LINK TO ALL HOMES"],
         reference="Andorra Telecom (SOM) is the exclusive national operator."),
    dict(qid="A3", type="link", cluster="andorra",
         question="Which historical institutions shaped Andorra's government?",
         answer_terms=["CONSELL DE LA TERRA", "CASA DE LA VALL", "TRIBUNAL DE CORTS", "CO-PRINCES"],
         reference="Institutions such as the Consell de la Terra, Casa de la Vall, Tribunal de Corts and the co-princes."),
    # egypt
    dict(qid="E1", type="single", cluster="egypt",
         question="What crops and animals did the ancient Egyptians depend on?",
         answer_terms=["EMMER", "BARLEY", "CATTLE", "FLAX", "PAPYRUS", "NILE RIVER"],
         reference="Emmer, barley, flax and papyrus, along with cattle and other livestock."),
    dict(qid="E2", type="link", cluster="egypt",
         question="What did genetic studies find about ancient Egyptians?",
         answer_terms=["SCHUENEMANN", "2017 STUDY", "MTDNA", "Y-DNA", "MODERN EGYPTIANS"],
         reference="Schuenemann et al. (2017) analysed mummy mtDNA and Y-DNA and found continuity with modern Egyptians."),
    dict(qid="E3", type="theme", cluster="egypt",
         question="How did religion shape ancient Egyptian society?",
         answer_terms=["GODS", "PRIESTS", "PANTHEON", "CULT TEMPLES", "PHARAOH"],
         reference="A continually changing pantheon was served by priests in cult temples acting on the king's behalf."),
]


def _norm(title) -> str:
    return " ".join(str(title).split()).upper()


def _gold_titles():
    slice_ids = {p["id"] for p in json.loads(SLICE.read_text(encoding="utf-8"))["passages"]}
    titles = set()
    for line in GOLD.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["id"] in slice_ids:
            for e in rec.get("entities") or []:
                t = _norm(e.get("title", ""))
                if t:
                    titles.add(t)
    return titles


def run_write():
    gold = _gold_titles()
    missing = {}
    for q in NATURAL_QUESTIONS:
        bad = [t for t in q["answer_terms"] if _norm(t) not in gold]
        if bad:
            missing[q["qid"]] = bad
    if missing:
        raise SystemExit(f"answer_terms not in gold slice (gate 1 fails): {missing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "questions.json").write_text(
        json.dumps(NATURAL_QUESTIONS, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    from collections import Counter

    by_type = Counter(q["type"] for q in NATURAL_QUESTIONS)
    by_cluster = Counter(q["cluster"] for q in NATURAL_QUESTIONS)
    print(f"预注册 {len(NATURAL_QUESTIONS)} 题 | types={dict(by_type)} | clusters={dict(by_cluster)}")
    for q in NATURAL_QUESTIONS:
        print(f"  [{q['qid']}/{q['type']}/{q['cluster']}] {q['question']}")
    return 0


def run_list():
    qs = json.loads((OUT_DIR / "questions.json").read_text(encoding="utf-8"))
    for q in qs:
        print(f"[{q['qid']}] ({q['type']}/{q['cluster']}) {q['question']}")
        print(f"     terms: {q['answer_terms']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.write:
        return run_write()
    if args.list:
        return run_list()
    ap.print_help()


if __name__ == "__main__":
    raise SystemExit(main() or 0)
